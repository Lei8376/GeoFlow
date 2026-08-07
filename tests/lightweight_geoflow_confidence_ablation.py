#!/usr/bin/env python3
"""Paired lightweight ablation for GeoFlow confidence-weighted supervision.

Runs two short finetune branches from the same checkpoint:
  1. baseline_x1: current pseudo-clean target supervision
  2. confw_x1: same target, but confidence-weighted flow/manifold losses

Both branches use identical train epochs, eval split list, and inference knobs.
The output CSV is intended for direction checks before full 312-split eval.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


LABELS_SCANNET20 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
]


VARIANTS = {
    "baseline_x1": (
        "geoflow_confidence_weighting",
        "False",
    ),
    "confw_x1": (
        "geoflow_confidence_weighting",
        "True",
        "geoflow_confidence_min",
        "0.05",
        "geoflow_confidence_power",
        "1.0",
        "geoflow_confidence_feature_w",
        "0.7",
        "geoflow_confidence_view_w",
        "0.3",
        "geoflow_confidence_view_ref",
        "3.0",
    ),
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def parse_int_csv(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def checkpoint_epoch(path, python_bin):
    code = (
        "import torch, sys; "
        "ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "print(int(ck.get('epoch', -1)))"
    )
    out = subprocess.check_output([python_bin, "-c", code, str(path)], text=True)
    return int(out.strip())


def split_stat_path(exp_dir, split_idx):
    final = exp_dir / f"eval_raw_split{split_idx}.npz"
    partial = exp_dir / f"eval_raw_split{split_idx}.partial.npz"
    if final.is_file():
        return final
    if partial.is_file():
        return partial
    return None


def merge_split_list(exp_dir, split_list):
    inter = union = target = None
    used = []
    missing = []
    for split_idx in split_list:
        path = split_stat_path(exp_dir, split_idx)
        if path is None:
            missing.append(split_idx)
            continue
        data = np.load(path)
        if inter is None:
            inter = np.zeros_like(data["intersection_all"], dtype=np.float64)
            union = np.zeros_like(data["union_all"], dtype=np.float64)
            target = np.zeros_like(data["target_all"], dtype=np.float64)
        inter += data["intersection_all"]
        union += data["union_all"]
        target += data["target_all"]
        used.append(split_idx)
    if inter is None:
        raise FileNotFoundError(f"No eval stats found in {exp_dir}")
    iou = inter / (union + 1e-10)
    acc = inter / (target + 1e-10)
    return {
        "used": used,
        "missing": missing,
        "iou": iou,
        "acc": acc,
        "miou": float(np.mean(iou) * 100.0),
        "macc": float(np.mean(acc) * 100.0),
        "allacc": float(inter.sum() / (target.sum() + 1e-10) * 100.0),
    }


def run_shell(shell_cmd, root, env, dry_run=False):
    print(shell_cmd)
    if dry_run:
        return
    subprocess.run(["bash", "-o", "pipefail", "-lc", shell_cmd], cwd=root, env=env, check=True)


def run_cmd(cmd, root, env, dry_run=False):
    print(" ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=root, env=env, check=True)


def write_variant_report(path, variant_name, result):
    lines = [
        f"variant={variant_name}",
        f"used_splits={result['used']}",
        f"missing_splits={result['missing']}",
        "per-class IoU:",
    ]
    for label, value in zip(LABELS_SCANNET20, result["iou"] * 100.0):
        lines.append(f"  {label}: {value:.2f}")
    lines.extend(
        [
            f"mIoU={result['miou']:.2f}",
            f"mAcc={result['macc']:.2f}",
            f"allAcc={result['allacc']:.2f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Paired confidence-weighting ablation for GeoFlow."
    )
    parser.add_argument(
        "--base",
        default=(
            "out/geoflow_ft_k96_a20_i8_m030_allviews_v180224_target256_loss1024_from_m080_last/"
            "model/geoflow_last.pth"
        ),
    )
    parser.add_argument("--config", default="config/geoflow_scannet.yaml")
    parser.add_argument(
        "--out-root",
        default="out/geoflow_confidence_ablation_from_ep113_splits48_55_90_224_274",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--split-list", default="48,55,90,224,274")
    parser.add_argument("--split-total", type=int, default=312)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--variants", default="baseline_x1,confw_x1")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")

    base = (root / args.base).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)

    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    for name in requested:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant: {name}. Available: {sorted(VARIANTS)}")

    split_list = parse_int_csv(args.split_list)
    out_root = (root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    repo_pythonpath = ".:third_party/sonata:third_party/X-Decoder:third_party/detectron2"
    env["PYTHONPATH"] = (
        repo_pythonpath
        if not env.get("PYTHONPATH")
        else f"{repo_pythonpath}:{env['PYTHONPATH']}"
    )

    base_epoch = checkpoint_epoch(base, args.python_bin)
    end_epoch = base_epoch + args.train_epochs + 1
    print(f"base={base}")
    print(f"base_epoch={base_epoch}, train_epochs={args.train_epochs}, epochs_arg={end_epoch}")
    print(f"split_list={split_list}")

    common_train_opts = (
        "geoflow_train_max_views",
        "0",
        "geoflow_train_max_voxels",
        "180224",
        "geodiff_target_chunk_v",
        "256",
        "geoflow_loss_chunk_v",
        "1024",
        "save_freq",
        "1",
        "use_student_target",
        "False",
        "geoflow_sigma_min",
        "0.0",
        "geoflow_manifold_loss_w",
        "0.3",
        "geodiff_target_knn",
        "96",
        "geodiff_target_alpha",
        "20.0",
        "geodiff_target_iters",
        "8",
    )

    summary_rows = []
    for variant_name in requested:
        branch = out_root / variant_name
        train_dir = branch / "train"
        model_dir = train_dir / "model"
        eval_dir = branch / f"eval_splits_{'_'.join(map(str, split_list))}"
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        resume_path = base
        if not args.skip_train:
            local_base = model_dir / base.name
            if not local_base.is_file() and not args.dry_run:
                shutil.copy2(base, local_base)
            elif args.dry_run:
                print(f"copy {base} -> {local_base}")
            resume_path = local_base
            train_cmd = [
                args.python_bin,
                "-u",
                "run/train_geoflow.py",
                "--config",
                args.config,
                "save_path",
                str(train_dir),
                "resume",
                str(resume_path),
                "epochs",
                str(end_epoch),
                *common_train_opts,
                *VARIANTS[variant_name],
            ]
            shell_cmd = (
                f"{' '.join(str(x) for x in train_cmd)} "
                f"2>&1 | tee -a {train_dir / 'train_light.log'}"
            )
            print(f"\n==> Train {variant_name}")
            run_shell(shell_cmd, root, {**env, "CUDA_VISIBLE_DEVICES": args.gpu}, args.dry_run)

        ckpt = model_dir / "geoflow_last.pth"
        if args.skip_train:
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing checkpoint for --skip-train: {ckpt}")
        elif not args.dry_run and not ckpt.is_file():
            raise FileNotFoundError(f"Training did not write checkpoint: {ckpt}")

        if not args.skip_eval:
            print(f"\n==> Eval {variant_name}")
            for split_idx in split_list:
                if split_stat_path(eval_dir, split_idx) is not None:
                    print(f"[skip] {variant_name} split {split_idx}: stats already exist")
                    continue
                cmd = [
                    args.python_bin,
                    "-u",
                    "run/eval_geoflow.py",
                    f"--config={args.config}",
                    f"--split_idx={split_idx}",
                    f"--split_total={args.split_total}",
                    "save_path",
                    str(eval_dir),
                    "resume",
                    str(ckpt),
                    "use_student_target",
                    "False",
                    "geoflow_eval_max_voxels",
                    "0",
                    "geoflow_ode_steps",
                    "8",
                ]
                log_path = eval_dir / f"geoflow-infer-split{split_idx}.log"
                shell_cmd = f"{' '.join(str(x) for x in cmd)} 2>&1 | tee {log_path}"
                run_shell(shell_cmd, root, {**env, "CUDA_VISIBLE_DEVICES": args.gpu}, args.dry_run)

            if not args.dry_run:
                result = merge_split_list(eval_dir, split_list)
                write_variant_report(eval_dir / "range_eval_summary.txt", variant_name, result)
                row = {
                    "variant": variant_name,
                    "base_epoch": base_epoch,
                    "train_epochs": args.train_epochs,
                    "epochs_arg": end_epoch,
                    "split_list": ",".join(map(str, split_list)),
                    "used_splits": len(result["used"]),
                    "missing_splits": ",".join(map(str, result["missing"])),
                    "mIoU": f"{result['miou']:.4f}",
                    "mAcc": f"{result['macc']:.4f}",
                    "allAcc": f"{result['allacc']:.4f}",
                    "picture_iou": f"{result['iou'][10] * 100.0:.4f}",
                    "counter_iou": f"{result['iou'][11] * 100.0:.4f}",
                    "desk_iou": f"{result['iou'][12] * 100.0:.4f}",
                    "sink_iou": f"{result['iou'][17] * 100.0:.4f}",
                    "checkpoint": str(ckpt),
                    "eval_dir": str(eval_dir),
                }
                summary_rows.append(row)
                print(
                    f"{variant_name}: mIoU={result['miou']:.2f} "
                    f"mAcc={result['macc']:.2f} allAcc={result['allacc']:.2f} "
                    f"picture={result['iou'][10] * 100.0:.2f} "
                    f"counter={result['iou'][11] * 100.0:.2f} "
                    f"desk={result['iou'][12] * 100.0:.2f} "
                    f"sink={result['iou'][17] * 100.0:.2f}"
                )

    if summary_rows:
        summary_path = out_root / "confidence_ablation_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
