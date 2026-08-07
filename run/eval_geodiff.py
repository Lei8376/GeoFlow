"""GeoDiff evaluation: reverse-diffusion purification -> text dot-product -> mIoU.

Runs one split per invocation (GPU chosen via CUDA_VISIBLE_DEVICES), writes raw
per-class stats to eval_raw_split{idx}.npz, and run/merge_eval.py aggregates.

    export PYTHONPATH=.
    CUDA_VISIBLE_DEVICES=0 python run/eval_geodiff.py \
        --config=config/geodiff_scannet.yaml --split_idx=0 --split_total=1 \
        save_path out/geodiff_scannet resume out/geodiff_scannet/model/geodiff_last.pth
"""
import os
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.neighbors import KDTree
from omegaconf import OmegaConf

from util import config
from util.util import AverageMeter, intersectionAndUnionGPU
from xdecoder.utils.arguments import load_opt_from_config_files
from models.geodiff_module import SonataXGeoDiffTrainer


def get_logger():
    lg = logging.getLogger("geodiff-eval")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    lg.addHandler(h)
    return lg


def get_parser():
    p = argparse.ArgumentParser("geodiff-eval")
    p.add_argument("--config", type=str, default="config/geodiff_scannet.yaml")
    p.add_argument("--split_idx", type=int, default=0)
    p.add_argument("--split_total", type=int, default=1)
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    a = p.parse_args()
    cfg = config.load_cfg_from_cfg_file(a.config)
    if a.opts:
        cfg = config.merge_cfg_from_list(cfg, a.opts)
    cfg.split_idx, cfg.split_total = a.split_idx, a.split_total
    os.makedirs(cfg.save_path, exist_ok=True)
    return cfg


def get_batch_scenes(scenes, idx, total):
    n = len(scenes); b = n // total; r = n % total
    s = idx * b + min(idx, r)
    e = s + b + (1 if idx < r else 0)
    return scenes[s:e]


def main():
    args = get_parser()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    logger = get_logger()

    dataset_name = "matterport" if "matterport" in args.data_root.lower() else "scannet"
    if dataset_name == "matterport":
        from dataset.data_loader_matterport import (
            ScannetLoaderFull, SceneBatchSampler, scene_based_collate_fn)
    else:
        from dataset.data_loader_ablation import (
            ScannetLoaderFull, SceneBatchSampler, scene_based_collate_fn)

    device = torch.device("cuda:0")
    scene_config = OmegaConf.merge(OmegaConf.load(f"./config/fusion_{dataset_name}.yaml"),
                                   OmegaConf.from_cli())
    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])

    model = SonataXGeoDiffTrainer(args, xdecoder_cfg, scene_config, device, False).to(device)

    ckpt = torch.load(args.resume, map_location=device)
    model.denoiser.load_state_dict(ckpt["denoiser"])
    if ckpt.get("cond_encoder") is not None:
        if model.cond_encoder is None:
            model._maybe_build_cond_encoder(model._cond_in_dim or 0)
        model.cond_encoder.load_state_dict(ckpt["cond_encoder"])
    model.eval()
    logger.info(f"Loaded GeoDiff checkpoint {args.resume}")

    with open(f"{dataset_name}_evaluation.txt") as f:
        scenes = [ln.strip() for ln in f if ln.strip()]
    scenes = get_batch_scenes(scenes, args.split_idx, args.split_total)
    logger.info(f"Split {args.split_idx+1}/{args.split_total}: {len(scenes)} scenes")

    if not hasattr(args, "input_color"):
        args.input_color = False
    val_data = ScannetLoaderFull(
        datapath_prefix=args.data_root, datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split, label_2d=args.label_2d,
        caption_path=args.caption_path, scannet200=args.scannet200, val_keep=args.val_keep,
        voxel_size=args.voxel_size, split="test" if dataset_name == "matterport" else "val",
        aug=False, memcache_init=args.use_shm, eval_all=True, input_color=args.input_color,
        scene_config=scene_config, specific_ids=scenes)
    sampler = SceneBatchSampler(val_data.samples, shuffle=False)
    loader = torch.utils.data.DataLoader(
        val_data, num_workers=args.infer_workers, pin_memory=True,
        collate_fn=scene_based_collate_fn, batch_sampler=sampler)

    inter_m, union_m, target_m = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            scene_coords, _, _, scene_label = batch[0], batch[1], batch[2], batch[3]
            out = model.evaluate_scene(batch)
            feats = F.normalize(out["scene_features"], dim=-1)
            txt = F.normalize(out["text_features"], dim=-1)
            logits = out["logit_scale"] * (feats @ txt.t())
            pred = logits.argmax(1)

            # fill points that received no 2D feature via nearest seen neighbour
            unseen = (feats.abs().sum(1) == 0).to(scene_coords.device)
            if unseen.any():
                seen = ~unseen
                sc = scene_coords[seen][:, 1:4].cpu()
                uc = scene_coords[unseen][:, 1:4].cpu()
                if sc.shape[0] > 0:
                    kd = KDTree(sc); _, idx = kd.query(uc, k=1)
                    seen_idx = torch.where(seen)[0]
                    pred[torch.where(unseen)[0]] = pred[seen_idx[idx.flatten()]]

            inter, union, tgt = intersectionAndUnionGPU(
                pred.to(scene_label.device), scene_label.detach(),
                args.test_classes, args.test_ignore_label)
            inter_m.update(inter.cpu().numpy())
            union_m.update(union.cpu().numpy())
            target_m.update(tgt.cpu().numpy())
            if i % 10 == 0:
                iou = inter_m.sum / (union_m.sum + 1e-10)
                logger.info(f"[{i}/{len(loader)}] running mIoU={np.mean(iou)*100:.2f}")

    out_npz = os.path.join(args.save_path, f"eval_raw_split{args.split_idx}.npz")
    np.savez(out_npz, intersection_all=inter_m.sum, union_all=union_m.sum, target_all=target_m.sum)
    iou = inter_m.sum / (union_m.sum + 1e-10)
    acc = inter_m.sum / (target_m.sum + 1e-10)
    logger.info(f"[split {args.split_idx}] mIoU={np.mean(iou)*100:.2f} mAcc={np.mean(acc)*100:.2f}")
    logger.info(f"raw stats -> {out_npz}")


if __name__ == "__main__":
    main()
