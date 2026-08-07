"""GeoDiff trainer (geometry-conditioned diffusion). torchrun DDP, 2x RTX 4090.

    export PYTHONPATH=.
    torchrun --standalone --nproc_per_node=2 run/train_geodiff.py \
        --config=config/geodiff_scannet.yaml save_path out/geodiff_scannet

Only the GeoDiff denoiser + geometry-condition encoder are optimized; X-Decoder
and Sonata stay frozen. One optimizer step == one scene; scenes are sharded
across ranks by DistributedSceneBatchSampler.
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
from models.geodiff_module import SonataXGeoDiffTrainer
from dataset.data_loader_ablation import (
    ScannetLoaderFull, DistributedSceneBatchSampler, scene_based_collate_fn)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_logger():
    logger = logging.getLogger("geodiff-logger")
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s %(filename)s:%(lineno)d] %(message)s"))
    logger.addHandler(h)
    return logger


def get_parser():
    p = argparse.ArgumentParser("geodiff")
    p.add_argument("--config", type=str, default="config/geodiff_scannet.yaml")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = config.load_cfg_from_cfg_file(a.config)
    if a.opts:
        cfg = config.merge_cfg_from_list(cfg, a.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    return cfg


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
                                world_size=world_size, rank=rank)

    scene_config = OmegaConf.merge(OmegaConf.load("./config/fusion_scannet.yaml"),
                                   OmegaConf.from_cli())
    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])

    logger = get_logger() if is_main else None
    writer = SummaryWriter(args.save_path) if is_main else None

    model = SonataXGeoDiffTrainer(args, xdecoder_cfg, scene_config, device, False).to(device)

    base_lr = args.lr_3d
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=base_lr, weight_decay=args.weight_decay)

    if args.distributed:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            if "model_state_dict" in checkpoint:
                if args.distributed:
                    model.module.denoiser.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.denoiser.load_state_dict(checkpoint["model_state_dict"])
            else:
                if args.distributed:
                    model.module.denoiser.load_state_dict(checkpoint)
                else:
                    model.denoiser.load_state_dict(checkpoint)

            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"] + 1
            else:
                import re
                match = re.search(r"epoch_(\d+)", args.resume)
                if match:
                    args.start_epoch = int(match.group(1)) + 1
                else:
                    args.start_epoch = 0
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "tensorboard_scalars" in checkpoint:
                previous_scalars = checkpoint["tensorboard_scalars"]
                if main_process():
                    for scalar_name, scalar_data in previous_scalars.items():
                        for step, value in scalar_data.items():
                            writer.add_scalar(scalar_name, value, step)
            if main_process():
                logger.info(
                    "=> loaded checkpoint '{}' (will start from epoch {})".format(
                        args.resume, args.start_epoch
                    )
                )
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
            args.start_epoch = 0
    else:
        args.start_epoch = 0
    if not hasattr(args, "input_color"):
        args.input_color = False
    with open("scannet_train.txt") as f:
        scene_ids = [ln.strip() for ln in f if ln.strip()]

    train_data = ScannetLoaderFull(
        datapath_prefix=args.data_root, datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split, label_2d=args.label_2d,
        caption_path=args.caption_path, entity_path=args.entity_path,
        scannet200=args.scannet200, val_keep=args.val_keep, voxel_size=args.voxel_size,
        split="train", aug=args.aug, memcache_init=args.use_shm, loop=args.loop,
        input_color=args.input_color, scene_config=scene_config, specific_ids=scene_ids)

    sampler = DistributedSceneBatchSampler(train_data.samples, world_size, rank, shuffle=True)
    loader = torch.utils.data.DataLoader(
        train_data, num_workers=args.workers, pin_memory=True,
        collate_fn=scene_based_collate_fn, batch_sampler=sampler, worker_init_fn=worker_init_fn)

    warmup_iters = max(args.warmup_epochs * len(loader), 1)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_iters)
    main_iters = max((args.epochs - args.warmup_epochs) * len(loader), 1)
    cosine = CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=base_lr * 1e-3)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_iters])

    grad_accum = int(getattr(args, "grad_accum_steps", 1))
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            loss = model(batch)
            if isinstance(loss, dict):
                loss = loss["loss"]
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            meter.update(loss.item())
            if is_main and i % args.print_freq == 0:
                logger.info(f"Epoch[{epoch}][{i}/{len(loader)}] loss {loss.item():.4f} "
                            f"lr {scheduler.get_last_lr()[0]:.7f}")
                writer.add_scalar("loss_iter", loss.item(), epoch * len(loader) + i)
        if is_main:
            writer.add_scalar("loss_train", meter.avg, epoch + 1)
            logger.info(f"Epoch {epoch} mean loss {meter.avg:.4f}")
            core = model.module if args.distributed else model
            sd = {"denoiser": core.denoiser.state_dict(),
                  "cond_encoder": core.cond_encoder.state_dict()
                  if core.cond_encoder is not None else None,
                  "epoch": epoch}
            torch.save(sd, os.path.join(args.save_path, "model", "geodiff_last.pth"))
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                torch.save(sd, os.path.join(args.save_path, "model", f"geodiff_epoch_{epoch}.pth"))
        if args.distributed:
            dist.barrier()

    if is_main:
        writer.close(); logger.info("==> GeoDiff training done.")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
