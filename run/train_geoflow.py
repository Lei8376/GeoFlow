"""GeoFlow trainer (geometry-conditioned flow matching).

Auto-detects dataset (matterport vs scannet) from `args.data_root` like
`run/validation.py` does, and loads the matching `config/fusion_*.yaml`,
`{dataset}_train.txt` scene list, and dataloader.

Typical launch (DDP, 2x RTX 3080/4090):

    export PYTHONPATH=.
    sh run/train_geoflow.sh --exp_dir=out/geoflow_matterport \
        --config=config/geoflow_matterport.yaml --nproc=2 --gpus=0,1

Only the GeoFlow velocity field + geometry-condition encoder are optimized;
X-Decoder, Sonata, and (when supplied) the GeoPurify affinity student stay
frozen. One optimizer step == one scene; scenes are sharded across DDP ranks by
``DistributedSceneBatchSampler``.
"""
import os
import time
import random
import logging
import argparse
import re
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

import MinkowskiEngine as ME
from util import config
from util.util import AverageMeter
from xdecoder.utils.arguments import load_opt_from_config_files
from models.geoflow_module import SonataXGeoFlowTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_logger():
    logger = logging.getLogger("geoflow-logger")
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s %(filename)s:%(lineno)d] %(message)s"))
    logger.addHandler(h)
    return logger


def get_parser():
    p = argparse.ArgumentParser("geoflow")
    p.add_argument("--config", type=str, default="config/geoflow_matterport.yaml")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = config.load_cfg_from_cfg_file(a.config)
    cfg_opts = []
    scene_opts = []
    raw_opts = list(a.opts or [])
    if raw_opts:
        assert len(raw_opts) % 2 == 0
        for key, value in zip(raw_opts[0::2], raw_opts[1::2]):
            if key.startswith("scene_config."):
                scene_opts.append(f"{key[len('scene_config.'):]}={value}")
            else:
                cfg_opts.extend([key, value])
    if cfg_opts:
        cfg = config.merge_cfg_from_list(cfg, cfg_opts)
    cfg.scene_config_opts = scene_opts
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    return cfg


def detect_dataset_name(data_root: str) -> str:
    dr = data_root.lower()
    if "matterport" in dr:
        return "matterport"
    if "scannet" in dr:
        return "scannet"
    raise ValueError(f"Cannot infer dataset from data_root: {data_root}")


def infer_epoch_from_path(path: str):
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def load_geoflow_checkpoint(args, model, optimizer, scheduler, scaler, loader_len,
                            grad_accum, device, logger, is_main):
    resume = getattr(args, "resume", None)
    if not resume:
        return int(getattr(args, "start_epoch", 0)), 0
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"GeoFlow checkpoint not found: {resume}")

    if is_main:
        logger.info(f"=> loading GeoFlow checkpoint '{resume}'")
    checkpoint = torch.load(resume, map_location=device, weights_only=False)
    core = model.module if getattr(args, "distributed", False) else model

    if isinstance(checkpoint, dict) and "velocity" in checkpoint:
        core.velocity.load_state_dict(checkpoint["velocity"])
        cond_state = checkpoint.get("cond_encoder")
        if cond_state is not None and core.cond_encoder is not None:
            core.cond_encoder.load_state_dict(cond_state)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        core.velocity.load_state_dict(checkpoint["model_state_dict"])
    else:
        core.velocity.load_state_dict(checkpoint)

    ckpt_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if ckpt_epoch is None:
        ckpt_epoch = infer_epoch_from_path(resume)
    start_epoch = int(ckpt_epoch) + 1 if ckpt_epoch is not None else int(args.start_epoch)

    if bool(getattr(args, "resume_weights_only", False)):
        start_epoch = int(getattr(args, "start_epoch", 0))
        global_step = start_epoch * loader_len
        if is_main:
            logger.info(
                f"=> loaded checkpoint '{resume}' (weights only by resume_weights_only=True); "
                f"will start from epoch {start_epoch}"
            )
        return start_epoch, global_step

    optimizer_state = None
    scheduler_state = None
    scaler_state = None
    global_step = start_epoch * loader_len
    if isinstance(checkpoint, dict):
        optimizer_state = checkpoint.get("optimizer_state_dict") or checkpoint.get("optimizer")
        scheduler_state = checkpoint.get("scheduler_state_dict") or checkpoint.get("scheduler")
        scaler_state = checkpoint.get("scaler_state_dict") or checkpoint.get("scaler")
        global_step = int(checkpoint.get("global_step", global_step))

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif ckpt_epoch is not None:
        completed_updates = start_epoch * (loader_len // max(grad_accum, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(completed_updates):
                scheduler.step()
    if scaler_state is not None and bool(getattr(args, "use_amp", False)):
        scaler.load_state_dict(scaler_state)

    if is_main:
        restored = []
        if optimizer_state is not None:
            restored.append("optimizer")
        if scheduler_state is not None:
            restored.append("scheduler")
        if scaler_state is not None and bool(getattr(args, "use_amp", False)):
            restored.append("amp scaler")
        restored_msg = ", ".join(restored) if restored else "weights only"
        logger.info(f"=> loaded checkpoint '{resume}' ({restored_msg}); "
                    f"will start from epoch {start_epoch}")
    return start_epoch, global_step


def main():
    args = get_parser()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = world_size > 1
    args.rank, args.world_size = rank, world_size
    is_main = (rank == 0)

    cudnn.benchmark = True
    if getattr(args, "manual_seed", None) is not None:
        s = args.manual_seed + rank
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://",
                                world_size=world_size, rank=rank,
                                timeout=timedelta(minutes=int(getattr(args, "ddp_timeout_minutes", 120))))

    dataset_name = detect_dataset_name(args.data_root)
    if is_main:
        print(f"[GeoFlow] detected dataset = {dataset_name}")

    #print(f"dataset_name:{dataset_name}")
    if dataset_name == "matterport":
        from dataset.data_loader_matterport import (
            ScannetLoaderFull, scene_based_collate_fn)
        # data_loader_matterport doesn't export the distributed sampler; fall
        # back to the one in data_loader_ablation (it only looks at the
        # samples_list, so it works for either dataset).
        from dataset.data_loader_ablation import DistributedSceneBatchSampler
    else:
        from dataset.data_loader_ablation import (
            ScannetLoaderFull, DistributedSceneBatchSampler, scene_based_collate_fn)

    scene_config = OmegaConf.merge(
        OmegaConf.load(f"./config/fusion_{dataset_name}.yaml"),
        OmegaConf.from_dotlist(getattr(args, "scene_config_opts", [])),
    )
    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])

    logger = get_logger() if is_main else None
    writer = SummaryWriter(args.save_path) if is_main else None

    model = SonataXGeoFlowTrainer(args, xdecoder_cfg, scene_config, device, False).to(device)

    base_lr = args.lr_3d
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=base_lr, weight_decay=args.weight_decay)

    use_amp = bool(getattr(args, "use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.distributed:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)

    if not hasattr(args, "input_color"):
        args.input_color = False

    train_scene_file = getattr(args, "train_scene_file", None) or f"{dataset_name}_train.txt"
    with open(train_scene_file) as f:
        scene_ids = [ln.strip() for ln in f if ln.strip()]
    if is_main:
        logger.info(f"[GeoFlow] {len(scene_ids)} training scenes from {train_scene_file}")

    train_data = ScannetLoaderFull(
        datapath_prefix=args.data_root, datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split, label_2d=args.label_2d,
        caption_path=getattr(args, "caption_path", None),
        entity_path=getattr(args, "entity_path", None),
        scannet200=getattr(args, "scannet200", False), val_keep=args.val_keep,
        voxel_size=args.voxel_size,
        split="train", aug=args.aug, memcache_init=args.use_shm, loop=args.loop,
        input_color=args.input_color, scene_config=scene_config, specific_ids=scene_ids)
    if len(train_data.samples) == 0:
        raise RuntimeError(
            f"No training views were created for {dataset_name}. "
            f"Check data_root/data_root_2d and fusion_{dataset_name}.yaml scene.scene_path."
        )

    sampler = DistributedSceneBatchSampler(
        train_data.samples, world_size, rank, shuffle=True,
        max_indices_per_scene=int(getattr(args, "geoflow_train_max_views", 0)))
    loader = torch.utils.data.DataLoader(
        train_data, num_workers=args.workers, pin_memory=True,
        collate_fn=scene_based_collate_fn, batch_sampler=sampler, worker_init_fn=worker_init_fn)

    warmup_iters = max(args.warmup_epochs * len(loader), 1)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iters)
    main_iters = max((args.epochs - args.warmup_epochs) * len(loader), 1)
    cosine = CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=base_lr * 1e-3)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_iters])

    grad_accum = int(getattr(args, "grad_accum_steps", 1))
    args.start_epoch, global_step = load_geoflow_checkpoint(
        args, model, optimizer, scheduler, scaler, len(loader), grad_accum,
        device, logger, is_main)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        meters = {
            "loss": AverageMeter(),
            "loss_fm": AverageMeter(),
            "loss_manifold": AverageMeter(),
            "cos_hat_x1": AverageMeter(),
            "conf_mean": AverageMeter(),
        }
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)
            if isinstance(out, dict):
                loss = out["loss"]
                aux = out
            else:
                loss = out
                aux = {"loss_fm": loss.detach(),
                       "loss_manifold": torch.zeros((), device=loss.device),
                       "cos_hat_x1": torch.zeros((), device=loss.device)}
            scaler.scale(loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            meters["loss"].update(loss.item())
            for k in ("loss_fm", "loss_manifold", "cos_hat_x1", "conf_mean"):
                v = aux.get(k)
                if isinstance(v, torch.Tensor):
                    meters[k].update(v.item())
            global_step += 1
            if is_main and i % args.print_freq == 0:
                conf_msg = (
                    f"  conf {meters['conf_mean'].avg:.3f}"
                    if meters["conf_mean"].count > 0 else ""
                )
                logger.info(
                    f"Epoch[{epoch}][{i}/{len(loader)}] "
                    f"loss {meters['loss'].avg:.4f}  "
                    f"fm {meters['loss_fm'].avg:.4f}  "
                    f"manif {meters['loss_manifold'].avg:.4f}  "
                    f"cos {meters['cos_hat_x1'].avg:.3f}  "
                    f"lr {scheduler.get_last_lr()[0]:.7f}"
                    f"{conf_msg}"
                )
                writer.add_scalar("loss_iter", loss.item(), global_step)
                writer.add_scalar("loss_fm_iter", meters["loss_fm"].avg, global_step)
                writer.add_scalar("loss_manifold_iter", meters["loss_manifold"].avg, global_step)
                writer.add_scalar("cos_hat_x1_iter", meters["cos_hat_x1"].avg, global_step)
        if is_main:
            writer.add_scalar("loss_train", meters["loss"].avg, epoch + 1)
            writer.add_scalar("loss_fm_train", meters["loss_fm"].avg, epoch + 1)
            writer.add_scalar("loss_manifold_train", meters["loss_manifold"].avg, epoch + 1)
            writer.add_scalar("cos_hat_x1_train", meters["cos_hat_x1"].avg, epoch + 1)
            conf_msg = (
                f" conf {meters['conf_mean'].avg:.3f}"
                if meters["conf_mean"].count > 0 else ""
            )
            logger.info(f"Epoch {epoch} mean loss {meters['loss'].avg:.4f} "
                        f"fm {meters['loss_fm'].avg:.4f} "
                        f"manif {meters['loss_manifold'].avg:.4f} "
                        f"cos {meters['cos_hat_x1'].avg:.3f}"
                        f"{conf_msg}")
            core = model.module if args.distributed else model
            sd = {"velocity": core.velocity.state_dict(),
                  "cond_encoder": core.cond_encoder.state_dict()
                  if core.cond_encoder is not None else None,
                  "epoch": epoch,
                  "global_step": global_step,
                  "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(),
                  "scaler_state_dict": scaler.state_dict()}
            torch.save(sd, os.path.join(args.save_path, "model", "geoflow_last.pth"))
            save_freq = max(1, int(getattr(args, "save_freq", 5)))
            if (epoch + 1) % save_freq == 0 or epoch == args.epochs - 1:
                torch.save(sd, os.path.join(args.save_path, "model", f"geoflow_epoch_{epoch}.pth"))
        if args.distributed:
            dist.barrier()

    if is_main:
        writer.close(); logger.info("==> GeoFlow training done.")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
