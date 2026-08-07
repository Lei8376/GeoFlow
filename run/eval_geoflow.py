"""GeoDiff evaluation: reverse-diffusion purification -> text dot-product -> mIoU.

Runs one split per invocation (GPU chosen via CUDA_VISIBLE_DEVICES), writes raw
per-class stats to eval_raw_split{idx}.npz, and run/merge_eval.py aggregates.

    export PYTHONPATH=.
    CUDA_VISIBLE_DEVICES=0 python run/eval_geoflow.py \
        --config=config/geoflow_scannet.yaml --split_idx=0 --split_total=1 \
        save_path out/geoflow_scannet resume out/geoflow_scannet/model/geoflow_last.pth
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
from models.geoflow_module import SonataXGeoFlowTrainer


def get_logger():
    lg = logging.getLogger("geoflow-eval")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    lg.addHandler(h)
    return lg


def get_parser():
    p = argparse.ArgumentParser("geoflow-eval")
    p.add_argument("--config", type=str, default="config/geoflow_scannet.yaml")
    p.add_argument("--split_idx", type=int, default=0)
    p.add_argument("--split_total", type=int, default=1)
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
    cfg.split_idx, cfg.split_total = a.split_idx, a.split_total
    os.makedirs(cfg.save_path, exist_ok=True)
    return cfg


def get_batch_scenes(scenes, idx, total):
    n = len(scenes); b = n // total; r = n % total
    s = idx * b + min(idx, r)
    e = s + b + (1 if idx < r else 0)
    return scenes[s:e]


def save_eval_checkpoint(path, inter, union, target, scenes_processed):
    tmp = f"{path}.tmp"
    np.savez(
        tmp,
        intersection_all=inter,
        union_all=union,
        target_all=target,
        scenes_processed=np.int64(scenes_processed),
    )
    os.replace(f"{tmp}.npz", path)


def save_timing_checkpoint(
    path,
    ode_time_sec,
    ode_voxels,
    timed_scenes,
    scenes_processed,
    ode_steps,
    ode_solver,
):
    tmp = f"{path}.tmp"
    np.savez(
        tmp,
        ode_time_sec=np.float64(ode_time_sec),
        ode_voxels=np.int64(ode_voxels),
        timed_scenes=np.int64(timed_scenes),
        scenes_processed=np.int64(scenes_processed),
        ode_steps=np.int64(ode_steps),
        ode_solver=np.asarray([str(ode_solver)]),
    )
    os.replace(f"{tmp}.npz", path)


def main():
    args = get_parser()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    logger = get_logger()
    out_npz = os.path.join(args.save_path, f"eval_raw_split{args.split_idx}.npz")
    out_partial = os.path.join(args.save_path, f"eval_raw_split{args.split_idx}.partial.npz")
    timing_npz = os.path.join(args.save_path, f"timing_raw_split{args.split_idx}.npz")
    timing_partial = os.path.join(args.save_path, f"timing_raw_split{args.split_idx}.partial.npz")
    timing_tsv = os.path.join(args.save_path, f"timing_split{args.split_idx}.tsv")
    if os.path.isfile(out_npz):
        logger.info(f"Final split stats already exist; skip split {args.split_idx}: {out_npz}")
        return

    dataset_name = "matterport" if "matterport" in args.data_root.lower() else "scannet"
    if dataset_name == "matterport":
        from dataset.data_loader_matterport import (
            ScannetLoaderFull, SceneBatchSampler, scene_based_collate_fn)
    else:
        from dataset.data_loader_ablation import (
            ScannetLoaderFull, SceneBatchSampler, scene_based_collate_fn)

    device = torch.device("cuda:0")
    scene_config = OmegaConf.merge(
        OmegaConf.load(f"./config/fusion_{dataset_name}.yaml"),
        OmegaConf.from_dotlist(getattr(args, "scene_config_opts", [])))
    xdecoder_cfg = load_opt_from_config_files(["./config/xdecoder_focall_lang.yaml"])

    model = SonataXGeoFlowTrainer(args, xdecoder_cfg, scene_config, device, False).to(device)

    if not args.resume or not os.path.isfile(args.resume):
        raise FileNotFoundError(
            f"GeoFlow checkpoint not found at {args.resume}; pass `resume <path>` on the CLI.")
    # Checkpoints saved by train_geoflow.py include optimizer/scheduler/scaler
    # states. Keep those on CPU during eval; only model weights are copied to GPU
    # by load_state_dict below.
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.velocity.load_state_dict(ckpt["velocity"])
    if ckpt.get("cond_encoder") is not None:
        if model.cond_encoder is None:
            model._maybe_build_cond_encoder(model._cond_in_dim or 0)
        model.cond_encoder.load_state_dict(ckpt["cond_encoder"])
    ckpt_epoch = ckpt.get("epoch", "?")
    del ckpt
    torch.cuda.empty_cache()
    model.eval()
    logger.info(f"Loaded GeoFlow checkpoint {args.resume} (epoch {ckpt_epoch})")
    if model.affinity_student is not None:
        logger.info("GeoFlow trainer also has a frozen GeoPurify student loaded "
                    "(used only at training time; inference uses the velocity ODE only).")

    with open(f"{dataset_name}_evaluation.txt") as f:
        scenes = [ln.strip() for ln in f if ln.strip()]
    scenes = get_batch_scenes(scenes, args.split_idx, args.split_total)
    logger.info(f"Split {args.split_idx+1}/{args.split_total}: {len(scenes)} scenes")

    if not hasattr(args, "input_color"):
        args.input_color = False
    val_data = ScannetLoaderFull(
        datapath_prefix=args.data_root, datapath_prefix_2d=args.data_root_2d,
        category_split=args.category_split, label_2d=args.label_2d,
        caption_path=getattr(args, "caption_path", None),
        entity_path=getattr(args, "entity_path", None),
        scannet200=getattr(args, "scannet200", False), val_keep=args.val_keep,
        voxel_size=args.voxel_size, split="test" if dataset_name == "matterport" else "val",
        aug=False, memcache_init=args.use_shm, eval_all=True, input_color=args.input_color,
        scene_config=scene_config, specific_ids=scenes)
    sampler = SceneBatchSampler(val_data.samples, shuffle=False)
    loader = torch.utils.data.DataLoader(
        val_data, num_workers=args.infer_workers, pin_memory=True,
        collate_fn=scene_based_collate_fn, batch_sampler=sampler)

    inter_m, union_m, target_m = AverageMeter(), AverageMeter(), AverageMeter()
    resume_from = 0
    ode_time_sum = 0.0
    ode_voxels_sum = 0
    timed_scenes = 0
    timing_ode_steps = int(getattr(args, "geoflow_ode_steps", 0))
    timing_ode_solver = str(getattr(args, "geoflow_ode_solver", "unknown"))
    if os.path.isfile(out_partial):
        partial = np.load(out_partial)
        inter_m.sum = partial["intersection_all"].astype(np.float64)
        union_m.sum = partial["union_all"].astype(np.float64)
        target_m.sum = partial["target_all"].astype(np.float64)
        inter_m.count = union_m.count = target_m.count = 1
        inter_m.avg = inter_m.sum
        union_m.avg = union_m.sum
        target_m.avg = target_m.sum
        resume_from = int(partial["scenes_processed"]) if "scenes_processed" in partial.files else 0
        logger.info(f"Resuming split {args.split_idx} from {out_partial}; "
                    f"skip first {resume_from}/{len(loader)} scene batches")
        if os.path.isfile(timing_partial):
            timing = np.load(timing_partial)
            ode_time_sum = float(timing["ode_time_sec"]) if "ode_time_sec" in timing.files else 0.0
            ode_voxels_sum = int(timing["ode_voxels"]) if "ode_voxels" in timing.files else 0
            timed_scenes = int(timing["timed_scenes"]) if "timed_scenes" in timing.files else 0
            if "ode_steps" in timing.files:
                timing_ode_steps = int(timing["ode_steps"])
            if "ode_solver" in timing.files:
                timing_ode_solver = str(timing["ode_solver"][0])
        elif resume_from > 0:
            logger.warning(
                f"Partial metric stats exist but timing checkpoint is missing: {timing_partial}; "
                "timing summary for this split will only include newly processed scenes.")

    if resume_from == 0 or not os.path.isfile(timing_tsv):
        with open(timing_tsv, "w", encoding="utf-8") as handle:
            handle.write(
                "split_idx\tscene_index\tscene_name\tode_steps\tode_solver\t"
                "ode_voxels\tode_time_sec\tode_ms_per_scene\tode_ms_per_step\t"
                "ode_voxels_per_sec\n"
            )

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i < resume_from:
                del batch
                continue
            if batch is None:
                save_eval_checkpoint(
                    out_partial, inter_m.sum, union_m.sum, target_m.sum, i + 1)
                continue
            scene_coords, _, _, scene_label = batch[0], batch[1], batch[2], batch[3]
            out = model.evaluate_scene(batch)
            ode_time = float(out.get("ode_time_sec", 0.0))
            ode_voxels = int(out.get("ode_voxels", 0))
            ode_steps = int(out.get("ode_steps", timing_ode_steps or 0))
            ode_solver = str(out.get("ode_solver", timing_ode_solver))
            scene_name = scenes[i] if i < len(scenes) else f"scene_index_{i}"
            ode_time_sum += ode_time
            ode_voxels_sum += ode_voxels
            timed_scenes += 1
            timing_ode_steps = ode_steps
            timing_ode_solver = ode_solver
            ms_per_scene = ode_time * 1000.0
            ms_per_step = (ode_time / max(ode_steps, 1)) * 1000.0
            voxels_per_sec = ode_voxels / max(ode_time, 1e-12)
            with open(timing_tsv, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{args.split_idx}\t{i}\t{scene_name}\t{ode_steps}\t{ode_solver}\t"
                    f"{ode_voxels}\t{ode_time:.9f}\t{ms_per_scene:.6f}\t"
                    f"{ms_per_step:.6f}\t{voxels_per_sec:.3f}\n"
                )

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

            save_eval_checkpoint(
                out_partial, inter_m.sum, union_m.sum, target_m.sum, i + 1)
            save_timing_checkpoint(
                timing_partial,
                ode_time_sum,
                ode_voxels_sum,
                timed_scenes,
                i + 1,
                timing_ode_steps,
                timing_ode_solver,
            )
            del batch, out, feats, txt, logits, pred
            torch.cuda.empty_cache()
            import gc as _gc
            _gc.collect()
            try:
                import ctypes as _ctypes
                _ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    save_eval_checkpoint(out_npz, inter_m.sum, union_m.sum, target_m.sum, len(loader))
    save_timing_checkpoint(
        timing_npz,
        ode_time_sum,
        ode_voxels_sum,
        timed_scenes,
        len(loader),
        timing_ode_steps,
        timing_ode_solver,
    )
    iou = inter_m.sum / (union_m.sum + 1e-10)
    acc = inter_m.sum / (target_m.sum + 1e-10)
    logger.info(f"[split {args.split_idx}] per-class IoU (%): "
                + " ".join(f"{v*100:.1f}" for v in iou))
    logger.info(f"[split {args.split_idx}] per-class Acc (%): "
                + " ".join(f"{v*100:.1f}" for v in acc))
    logger.info(f"[split {args.split_idx}] mIoU={np.mean(iou)*100:.2f} "
                f"mAcc={np.mean(acc)*100:.2f} "
                f"allAcc={inter_m.sum.sum()/(target_m.sum.sum()+1e-10)*100:.2f}")
    logger.info(f"raw stats -> {out_npz}")
    logger.info(
        f"[split {args.split_idx}] ODE inference time only: "
        f"{ode_time_sum:.6f}s over {timed_scenes} scenes "
        f"({ode_time_sum / max(timed_scenes, 1):.6f}s/scene), "
        f"voxels={ode_voxels_sum}")
    logger.info(f"timing stats -> {timing_npz}")


if __name__ == "__main__":
    main()
