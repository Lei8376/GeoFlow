#!/usr/bin/env python3
"""Lightweight GeoFlow finetune sweep from a fixed checkpoint.

This is an experiment driver, not a unit test. It starts from an existing
GeoFlow checkpoint, runs short finetune branches, evaluates each branch on a
fixed split range, and writes a compact CSV so checkpoint/target choices can be
compared before launching full 312-split evaluation.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Variant:
    name: str
    resume_mode: str
    train_opts: tuple
    eval_opts: tuple = ()


DEFAULT_VARIANTS = [
    Variant(
        name="continue_k96_a20_i10",
        resume_mode="full",
        train_opts=(
            "lr_3d", "0.00005",
            "lr_others", "0.000025",
            "geodiff_target_knn", "96",
            "geodiff_target_alpha", "20.0",
            "geodiff_target_iters", "10",
        ),
    ),
    Variant(
        name="reset_low_lr_k96_a20_i10_sigma001",
        resume_mode="weights",
        train_opts=(
            "lr_3d", "0.00002",
            "lr_others", "0.00001",
            "geoflow_sigma_min", "0.01",
            "geoflow_manifold_loss_w", "0.05",
            "geodiff_target_knn", "96",
            "geodiff_target_alpha", "20.0",
            "geodiff_target_iters", "10",
        ),
    ),
    Variant(
        name="reset_low_lr_k80_a30_i6_sigma001",
        resume_mode="weights",
        train_opts=(
            "lr_3d", "0.00002",
            "lr_others", "0.00001",
            "geoflow_sigma_min", "0.01",
            "geoflow_manifold_loss_w", "0.05",
            "geodiff_target_knn", "80",
            "geodiff_target_alpha", "30.0",
            "geodiff_target_iters", "6",
        ),
    ),
    Variant(
        name="reset_low_lr_k96_a30_i6_sigma001",
        resume_mode="weights",
        train_opts=(
            "lr_3d", "0.00002",
            "lr_others", "0.00001",
            "geoflow_sigma_min", "0.01",
            "geoflow_manifold_loss_w", "0.05",
            "geodiff_target_knn", "96",
            "geodiff_target_alpha", "30.0",
            "geodiff_target_iters", "6",
        ),
    ),
]


def repo_root():
    return Path(__file__).resolve().parents[1]


def parse_csv_names(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def checkpoint_epoch(path, python_bin):
    code = (
        "import torch, sys; "
        "ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "print(int(ck.get('epoch', -1)))"
    )
    out = subprocess.check_output([python_bin, "-c", code, str(path)], text=True)
    return int(out.strip())


def ensure_weights_only(src, dst, root, python_bin, dry_run=False):
    if dst.is_file():
        return
    cmd = [python_bin, "run/strip_geoflow_weights.py", str(src), str(dst)]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=root, check=True)


def merge_range(exp_dir, start, end):
    inter = union = target = None
    used = []
    missing = []
    for split_idx in range(start, end + 1):
        final = exp_dir / f"eval_raw_split{split_idx}.npz"
        partial = exp_dir / f"eval_raw_split{split_idx}.partial.npz"
        if final.is_file():
            path = final
        elif partial.is_file():
            path = partial
        else:
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
        "used_splits": used,
        "missing_splits": missing,
        "iou": iou,
        "acc": acc,
        "miou": float(np.mean(iou) * 100.0),
        "macc": float(np.mean(acc) * 100.0),
        "allacc": float(inter.sum() / (target.sum() + 1e-10) * 100.0),
    }


def write_eval_report(path, variant, result):
    lines = [
        f"variant={variant.name}",
        f"resume_mode={variant.resume_mode}",
        f"merged_splits={len(result['used_splits'])}",
        f"missing_splits={result['missing_splits']}",
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


def run_command(cmd, root, env, dry_run=False):
    print(" ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=root, env=env, check=True)


def remove_numbered_checkpoints(model_dir, dry_run=False):
    for path in model_dir.glob("geoflow_epoch_*.pth"):
        print(f"remove {path}")
        if not dry_run:
            path.unlink()


def copy_checkpoint(src, dst, dry_run=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"copy {src} -> {dst}")
    if not dry_run:
        shutil.copy2(src, dst)


def update_summary_csv(path, new_rows):
    if not new_rows:
        return
    fieldnames = list(new_rows[0].keys())
    rows = []
    if path.is_file():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))

    def row_key(row):
        return (
            row.get("variant"),
            row.get("train_epochs"),
            row.get("eval_start"),
            row.get("eval_end"),
        )

    row_map = {row_key(row): row for row in rows}
    for row in new_rows:
        row_map[row_key(row)] = row

    merged = list(row_map.values())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Short GeoFlow finetune/eval sweep from epoch102-style checkpoint."
    )
    parser.add_argument(
        "--base",
        default=(
            "out/geoflow_ft_k96_a20_i10_allviews_v180224_target256_loss1024_from_clean/"
            "model/geoflow_epoch_102.pth"
        ),
        help="Full GeoFlow checkpoint used as the starting point.",
    )
    parser.add_argument("--config", default="config/geoflow_scannet.yaml")
    parser.add_argument(
        "--out-root",
        default="out/geoflow_epoch102_light_finetune_sweep",
        help="Parent directory for branch training and eval outputs.",
    )
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--splits", type=int, default=312)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or all.",
    )
    parser.add_argument("--geoflow-train-max-views", default="0")
    parser.add_argument("--geoflow-train-max-voxels", default="180224")
    parser.add_argument("--geodiff-target-chunk-v", default="256")
    parser.add_argument("--geoflow-loss-chunk-v", default="1024")
    parser.add_argument("--ode-steps", default="8")
    parser.add_argument("--delete-numbered-ckpts", action="store_true", default=True)
    parser.add_argument("--keep-numbered-ckpts", dest="delete_numbered_ckpts", action="store_false")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")
    if args.start < 0 or args.end < args.start or args.end >= args.splits:
        raise ValueError(
            f"Invalid split range start={args.start}, end={args.end}, splits={args.splits}"
        )

    base = (root / args.base).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)

    requested = None if args.variants == "all" else set(parse_csv_names(args.variants))
    variants = [v for v in DEFAULT_VARIANTS if requested is None or v.name in requested]
    if not variants:
        raise ValueError(f"No variants selected by --variants={args.variants}")

    out_root = (root / args.out_root).resolve()
    weights_only = out_root / "base_weights_only.pth"
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    repo_pythonpath = ".:third_party/sonata:third_party/X-Decoder:third_party/detectron2"
    env["PYTHONPATH"] = (
        repo_pythonpath
        if not env.get("PYTHONPATH")
        else f"{repo_pythonpath}:{env['PYTHONPATH']}"
    )
    env["PYTHON_BIN"] = args.python_bin

    base_epoch = checkpoint_epoch(base, args.python_bin)
    if any(v.resume_mode == "weights" for v in variants):
        ensure_weights_only(base, weights_only, root, args.python_bin, dry_run=args.dry_run)

    summary_rows = []
    common_train_opts = (
        "geoflow_train_max_views", args.geoflow_train_max_views,
        "geoflow_train_max_voxels", args.geoflow_train_max_voxels,
        "geodiff_target_chunk_v", args.geodiff_target_chunk_v,
        "geoflow_loss_chunk_v", args.geoflow_loss_chunk_v,
        "save_freq", "9999",
    )

    for variant in variants:
        branch_dir = out_root / variant.name
        train_dir = branch_dir / "train"
        eval_dir = branch_dir / f"eval_splits{args.start}_{args.end}"
        train_dir.mkdir(parents=True, exist_ok=True)
        resume = base if variant.resume_mode == "full" else weights_only
        epochs = base_epoch + args.train_epochs + 1 if variant.resume_mode == "full" else args.train_epochs

        if not args.skip_train:
            if variant.resume_mode == "full":
                copy_checkpoint(base, train_dir / "model" / base.name, dry_run=args.dry_run)
                resume = train_dir / "model" / base.name
            train_cmd = [
                args.python_bin,
                "-u",
                "run/train_geoflow.py",
                "--config",
                args.config,
                "save_path",
                str(train_dir),
                "resume",
                str(resume),
                "epochs",
                str(epochs),
                *common_train_opts,
                *variant.train_opts,
            ]
            train_log = train_dir / "train_light.log"
            shell_cmd = f"{' '.join(str(x) for x in train_cmd)} 2>&1 | tee -a {train_log}"
            print(f"\n==> Train {variant.name}")
            print(shell_cmd)
            if not args.dry_run:
                subprocess.run(
                    ["bash", "-o", "pipefail", "-lc", shell_cmd],
                    cwd=root,
                    env={**env, "CUDA_VISIBLE_DEVICES": args.gpu},
                    check=True,
                )
                ckpt_after_train = train_dir / "model" / "geoflow_last.pth"
                if not ckpt_after_train.is_file():
                    raise FileNotFoundError(
                        f"Training finished but did not write checkpoint: {ckpt_after_train}"
                    )
                if args.delete_numbered_ckpts:
                    remove_numbered_checkpoints(train_dir / "model")

        ckpt = train_dir / "model" / "geoflow_last.pth"
        if args.skip_train and not ckpt.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint for --skip-train: {ckpt}")

        if not args.skip_eval:
            eval_cmd = [
                "bash",
                "run/val_geoflow_split_range.sh",
                f"--exp_dir={eval_dir}",
                f"--config={args.config}",
                f"--resume={ckpt}",
                f"--splits={args.splits}",
                f"--start={args.start}",
                f"--end={args.end}",
                f"--gpu={args.gpu}",
                "--no-merge",
                "use_student_target",
                "False",
                "geoflow_eval_max_voxels",
                "0",
                "geoflow_ode_steps",
                str(args.ode_steps),
                *variant.eval_opts,
            ]
            if args.fail_fast:
                eval_cmd.insert(-10, "--fail-fast")
            print(f"\n==> Eval {variant.name}")
            run_command(eval_cmd, root, env, dry_run=args.dry_run)
            if not args.dry_run:
                result = merge_range(eval_dir, args.start, args.end)
                write_eval_report(eval_dir / "range_eval_summary.txt", variant, result)
                summary_rows.append(
                    {
                        "variant": variant.name,
                        "resume_mode": variant.resume_mode,
                        "train_epochs": args.train_epochs,
                        "eval_start": args.start,
                        "eval_end": args.end,
                        "merged_splits": len(result["used_splits"]),
                        "missing_splits": " ".join(map(str, result["missing_splits"])),
                        "mIoU": f"{result['miou']:.4f}",
                        "mAcc": f"{result['macc']:.4f}",
                        "allAcc": f"{result['allacc']:.4f}",
                        "picture_iou": f"{result['iou'][10] * 100.0:.4f}",
                        "counter_iou": f"{result['iou'][11] * 100.0:.4f}",
                        "desk_iou": f"{result['iou'][12] * 100.0:.4f}",
                        "sink_iou": f"{result['iou'][17] * 100.0:.4f}",
                        "train_dir": str(train_dir),
                        "eval_dir": str(eval_dir),
                    }
                )
                print(
                    f"{variant.name}: mIoU={result['miou']:.2f} "
                    f"mAcc={result['macc']:.2f} allAcc={result['allacc']:.2f} "
                    f"picture={result['iou'][10] * 100.0:.2f}"
                )

    if summary_rows:
        summary_path = out_root / "finetune_sweep_summary.csv"
        update_summary_csv(summary_path, summary_rows)
        print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
