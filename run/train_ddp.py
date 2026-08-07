"""GeoPurify DDP trainer for multi-GPU (e.g. 2x RTX 4090).

Launch with torchrun, which sets RANK / LOCAL_RANK / WORLD_SIZE:

    export PYTHONPATH=.
    torchrun --standalone --nproc_per_node=2 run/train_ddp.py \
        --config=config/geopurify_scannet.yaml save_path out/scannet_ddp

This path is intentionally separate from run/train.py (the single-GPU
mp.spawn path) so the two launch mechanisms never interfere. The model,
dataset, scheduler and loss are identical; only the parallelism wrapper and
the scene-sharding sampler differ.
"""
import os
import time
import random
import logging
import argparse

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
from models.affinity_module import SonataXAffinityTrainer
from dataset.data_loader_ablation import (
    ScannetLoaderFull,
    DistributedSceneBatchSampler,
    scene_based_collate_fn,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_logger():
    logger = logging.getLogger("main-logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"))
    logger.addHandler(handler)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description="geopurify-ddp")
    parser.add_argument("--config", type=str, default="config/geopurify_scannet.yaml")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    return cfg


def main():
    args = get_parser()

    # torchrun-provided environment.
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = world_size > 1
    args.multiprocessing_distributed = False  # we do NOT use mp.spawn here
    args.rank = rank
    args.world_size = world_size
    args.ngpus_per_node = world_size

    is_main = (rank == 0)

    cudnn.benchmark = True
    if getattr(args, "manual_seed", None) is not None:
        seed = args.manual_seed + rank
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method="env://",
                                world_size=world_size, rank=rank)

    scene_config = OmegaConf.load("./config/fusion_scannet.yaml")
    scene_config = OmegaConf.merge(scene_config, OmegaConf.from_cli())
    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])

    logger = get_logger() if is_main else None
    writer = SummaryWriter(args.save_path) if is_main else None

    model = SonataXAffinityTrainer(args, xdecoder_cfg, scene_config, device, False).to(device)

    # Differential LR groups (paper Table 7).
    student = model.affinity_student
    pg = student.get_param_groups()
    base_lr = args.lr_3d
    optimizer = torch.optim.AdamW([
        {"params": pg["input"],  "lr": base_lr * 0.1},
        {"params": pg["middle"], "lr": base_lr},
        {"params": pg["output"], "lr": base_lr * 5.0},
    ], weight_decay=args.weight_decay)

    if args.distributed:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    if not hasattr(args, "input_color"):
        args.input_color = False

    with open("scannet_train.txt") as f:
        scene_ids_train = [ln.strip() for ln in f if ln.strip()]

    train_data = ScannetLoaderFull(
        datapath_prefix=args.data_root, datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split, label_2d=args.label_2d,
        caption_path=args.caption_path, entity_path=args.entity_path,
        scannet200=args.scannet200, val_keep=args.val_keep, voxel_size=args.voxel_size,
        split="train", aug=args.aug, memcache_init=args.use_shm, loop=args.loop,
        input_color=args.input_color, scene_config=scene_config,
        specific_ids=scene_ids_train,
    )

    sampler = DistributedSceneBatchSampler(
        train_data.samples, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, num_workers=args.workers, pin_memory=True,
        collate_fn=scene_based_collate_fn, batch_sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(warmup_iters, 1))
    main_iters = (args.epochs - args.warmup_epochs) * len(train_loader)
    cosine = CosineAnnealingLR(optimizer, T_max=max(main_iters, 1), eta_min=base_lr * 1e-3)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[max(warmup_iters, 1)])

    grad_accum = int(getattr(args, "grad_accum_steps", 1))
    use_amp = bool(getattr(args, "use_amp", False))  # off by default (sparse conv)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(batch_data)
                if isinstance(loss, dict):
                    loss = loss["loss"]
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_meter.update(loss.item() * grad_accum)
            if is_main and i % args.print_freq == 0:
                lr_now = scheduler.get_last_lr()[1]
                logger.info(f"Epoch [{epoch}][{i}/{len(train_loader)}] "
                            f"Loss {loss.item()*grad_accum:.4f} LR {lr_now:.7f}")
                writer.add_scalar("loss_iter", loss.item() * grad_accum,
                                  epoch * len(train_loader) + i)

        if is_main:
            writer.add_scalar("loss_train", loss_meter.avg, epoch + 1)
            logger.info(f"Epoch {epoch} mean loss {loss_meter.avg:.4f}")
            student_sd = (model.module.affinity_student if args.distributed
                          else model.affinity_student).state_dict()
            ckpt = {"epoch": epoch, "model_state_dict": student_sd,
                    "optimizer_state_dict": optimizer.state_dict()}
            torch.save(ckpt, os.path.join(args.save_path, "model",
                                          "affinity_predictor_last.pth"))
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                torch.save(ckpt, os.path.join(args.save_path, "model",
                                              f"affinity_predictor_epoch_{epoch}.pth"))
        if args.distributed:
            dist.barrier()

    if is_main:
        writer.close()
        logger.info("==> DDP training done.")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
