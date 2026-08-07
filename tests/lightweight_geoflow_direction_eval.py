#!/usr/bin/env python3
"""Lightweight GeoFlow direction evaluation on an explicit split list.

This script is intentionally inference-only. It runs a small, fixed set of
validation splits for one checkpoint while sweeping test-time knobs such as ODE
steps and optional output blending. Use it to decide which direction deserves a
real finetune/full evaluation from the epoch102 checkpoint.
"""

import argparse
import csv
import os
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

DEFAULT_RESUME = (
    "out/geoflow_ft_k96_a20_i10_allviews_v180224_target256_loss1024_from_clean/"
    "model/geoflow_epoch_102.pth"
)
DEFAULT_SPLIT_LIST = "48,55,90,224,274"
SMALL_CLASS_IDXS = [10, 11, 12, 17]  # picture, counter, desk, sink
CORE_CLASS_IDXS = [0, 1, 4, 6, 7]    # wall, floor, chair, table, door


def repo_root():
    return Path(__file__).resolve().parents[1]


def parse_int_csv(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("empty integer CSV")
    return values


def parse_float_csv(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float CSV")
    return values


def format_blend(blend):
    return str(blend).replace(".", "p").replace("-", "m")


def prepare_env(root, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    paths = [
        str(root),
        str(root / "third_party" / "sonata"),
        str(root / "third_party" / "X-Decoder"),
        str(root / "third_party" / "detectron2"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


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
        "used_splits": used,
        "missing_splits": missing,
        "iou": iou,
        "acc": acc,
        "miou": float(iou.mean() * 100.0),
        "macc": float(acc.mean() * 100.0),
        "allacc": float(inter.sum() / (target.sum() + 1e-10) * 100.0),
        "small_miou": float(iou[SMALL_CLASS_IDXS].mean() * 100.0),
        "core_miou": float(iou[CORE_CLASS_IDXS].mean() * 100.0),
    }


def write_report(path, setting, result):
    lines = [
        f"name={setting['name']}",
        f"resume={setting['resume']}",
        f"steps={setting['steps']}",
        f"blend={setting['blend']}",
        f"splits={setting['split_list']}",
        f"merged_splits={len(result['used_splits'])}",
        f"missing_splits={result['missing_splits']}",
        "",
        "per-class IoU:",
    ]
    for label, value in zip(LABELS_SCANNET20, result["iou"] * 100.0):
        lines.append(f"  {label}: {value:.2f}")
    lines.extend(
        [
            "",
            f"mIoU={result['miou']:.2f}",
            f"mAcc={result['macc']:.2f}",
            f"allAcc={result['allacc']:.2f}",
            f"small_mIoU={result['small_miou']:.2f}",
            f"core_mIoU={result['core_miou']:.2f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def update_summary_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    old_rows = []
    if path.is_file():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))

    def row_key(row):
        return (
            row.get("name"),
            row.get("steps"),
            row.get("blend"),
            row.get("split_list"),
        )

    merged = {row_key(row): row for row in old_rows}
    for row in rows:
        merged[row_key(row)] = row

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged.values())


def build_eval_command(args, exp_dir, split_idx, steps, blend):
    cmd = [
        args.python_bin,
        "-u",
        "run/eval_geoflow.py",
        "--config",
        args.config,
        "--split_idx",
        str(split_idx),
        "--split_total",
        str(args.split_total),
        "save_path",
        str(exp_dir),
        "resume",
        args.resume,
        "use_student_target",
        "False",
        "geoflow_eval_max_voxels",
        str(args.eval_max_voxels),
        "geoflow_ode_steps",
        str(steps),
        "geoflow_ode_solver",
        args.solver,
        "geoflow_renorm_each_step",
        str(args.renorm_each_step),
        "geoflow_eval_blend",
        str(blend),
    ]
    cmd.extend(args.opts)
    return cmd


def run_eval_setting(args, root, env, split_list, steps, blend):
    setting_name = f"{args.name}_steps{steps:02d}_blend{format_blend(blend)}"
    exp_dir = Path(args.out_root) / setting_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for split_idx in split_list:
        if split_stat_path(exp_dir, split_idx) is not None:
            print(f"[skip] {setting_name} split {split_idx}: stats already exist")
            continue

        cmd = build_eval_command(args, exp_dir, split_idx, steps, blend)
        print(" ".join(str(x) for x in cmd))
        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, cwd=root, env=env, check=True)
        except subprocess.CalledProcessError:
            failed.append(split_idx)
            print(f"[failed] {setting_name} split {split_idx}")
            if args.fail_fast:
                raise

    if args.dry_run:
        return None

    result = merge_split_list(exp_dir, split_list)
    if failed:
        failed_path = exp_dir / "failed_splits.txt"
        failed_path.write_text("\n".join(str(x) for x in failed) + "\n")

    setting = {
        "name": args.name,
        "resume": args.resume,
        "steps": steps,
        "blend": blend,
        "split_list": ",".join(str(x) for x in split_list),
    }
    write_report(exp_dir / "light_direction_report.txt", setting, result)
    print(
        f"[summary] {setting_name}: mIoU={result['miou']:.2f} "
        f"mAcc={result['macc']:.2f} allAcc={result['allacc']:.2f} "
        f"small={result['small_miou']:.2f} core={result['core_miou']:.2f} "
        f"used={len(result['used_splits'])}/{len(split_list)}"
    )
    return setting_name, result


def make_summary_row(args, setting_name, split_list, steps, blend, result):
    row = {
        "setting": setting_name,
        "name": args.name,
        "resume": args.resume,
        "steps": steps,
        "blend": blend,
        "split_total": args.split_total,
        "split_list": ",".join(str(x) for x in split_list),
        "used_splits": len(result["used_splits"]),
        "missing_splits": ",".join(str(x) for x in result["missing_splits"]),
        "mIoU": f"{result['miou']:.4f}",
        "mAcc": f"{result['macc']:.4f}",
        "allAcc": f"{result['allacc']:.4f}",
        "small_mIoU": f"{result['small_miou']:.4f}",
        "core_mIoU": f"{result['core_miou']:.4f}",
    }
    for label, value in zip(LABELS_SCANNET20, result["iou"] * 100.0):
        row[f"iou_{label.replace(' ', '_')}"] = f"{float(value):.4f}"
    return row


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Run explicit split-list GeoFlow eval sweeps for quick direction checks."
    )
    parser.add_argument("--resume", default=DEFAULT_RESUME)
    parser.add_argument("--name", default="epoch102")
    parser.add_argument("--config", default="config/geoflow_scannet.yaml")
    parser.add_argument("--out-root", default="out/geoflow_epoch102_direction_eval")
    parser.add_argument("--split-list", default=DEFAULT_SPLIT_LIST)
    parser.add_argument("--split-total", type=int, default=312)
    parser.add_argument("--steps", default="8,12,16,24")
    parser.add_argument("--blends", default="1.0")
    parser.add_argument("--solver", default="midpoint")
    parser.add_argument("--eval-max-voxels", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--renorm-each-step", default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    split_list = parse_int_csv(args.split_list)
    steps_list = parse_int_csv(args.steps)
    blend_list = parse_float_csv(args.blends)

    env = prepare_env(root, args.gpu)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    rows = []
    for steps in steps_list:
        for blend in blend_list:
            item = run_eval_setting(args, root, env, split_list, steps, blend)
            if item is None:
                continue
            setting_name, result = item
            rows.append(make_summary_row(args, setting_name, split_list, steps, blend, result))

    if rows:
        summary_path = Path(args.out_root) / "direction_eval_summary.csv"
        update_summary_csv(summary_path, rows)
        print(f"[summary_csv] {summary_path}")


if __name__ == "__main__":
    main()
