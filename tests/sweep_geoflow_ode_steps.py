#!/usr/bin/env python3
"""Run a small GeoFlow ODE-step sweep on a fixed split range.

This is an experiment driver, not a model test.  It calls the existing
``run/val_geoflow_split_range.sh`` with ``--no-merge`` for each requested ODE
step, then merges only the requested split range so a 40-scene smoke sweep does
not require all 312 ScanNet validation splits.
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


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_bools(text):
    vals = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.lower() in {"1", "true", "yes", "y"}:
            vals.append(True)
        elif item.lower() in {"0", "false", "no", "n"}:
            vals.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value: {item}")
    return vals


def bool_cli(value):
    return "True" if value else "False"


def repo_root():
    return Path(__file__).resolve().parents[1]


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
        raise FileNotFoundError(f"No eval stats found in {exp_dir} for splits {start}-{end}")

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


def write_step_report(path, step, renorm, solver, result):
    lines = [
        f"geoflow_ode_steps={step}",
        f"geoflow_ode_solver={solver}",
        f"geoflow_renorm_each_step={bool_cli(renorm)}",
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


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Sweep GeoFlow ODE steps on a small fixed split range."
    )
    parser.add_argument(
        "--resume",
        default="out/geoflow_scannet_planA_clean_maxviews/model/geoflow_last.pth",
        help="GeoFlow checkpoint to evaluate.",
    )
    parser.add_argument("--config", default="config/geoflow_scannet.yaml")
    parser.add_argument(
        "--out-root",
        default="out/geoflow_scannet_planA_ode_steps_sweep_splits0_39",
        help="Parent directory for per-setting eval outputs.",
    )
    parser.add_argument("--steps", default="4,8,12,16,24")
    parser.add_argument(
        "--renorms",
        default="True",
        help="Comma-separated values, e.g. True or True,False.",
    )
    parser.add_argument("--solver", default="midpoint")
    parser.add_argument("--splits", type=int, default=312)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python interpreter used by val_geoflow_split_range.sh.",
    )
    parser.add_argument(
        "--keep-student-target",
        action="store_true",
        help="Do not force use_student_target False. Default keeps this PlanA-only.",
    )
    parser.add_argument(
        "--extra-opt",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Extra config override passed to eval_geoflow.py.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start < 0 or args.end < args.start or args.end >= args.splits:
        raise ValueError(
            f"Invalid split range start={args.start}, end={args.end}, splits={args.splits}"
        )

    steps = parse_csv_ints(args.steps)
    renorms = parse_csv_bools(args.renorms)
    out_root = (root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    env = os.environ.copy()
    env["PYTHON_BIN"] = args.python_bin
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    for renorm in renorms:
        for step in steps:
            exp_name = (
                f"steps{step:02d}_solver-{args.solver}_renorm-{bool_cli(renorm).lower()}"
                f"_splits{args.start}_{args.end}"
            )
            exp_dir = out_root / exp_name
            cmd = [
                "bash",
                "run/val_geoflow_split_range.sh",
                f"--exp_dir={exp_dir}",
                f"--config={args.config}",
                f"--resume={args.resume}",
                f"--splits={args.splits}",
                f"--start={args.start}",
                f"--end={args.end}",
                f"--gpu={args.gpu}",
                "--no-merge",
            ]
            if args.fail_fast:
                cmd.append("--fail-fast")
            if not args.keep_student_target:
                cmd.extend(["use_student_target", "False"])
            cmd.extend(
                [
                    "geoflow_ode_steps",
                    str(step),
                    "geoflow_ode_solver",
                    args.solver,
                    "geoflow_renorm_each_step",
                    bool_cli(renorm),
                ]
            )
            for key, value in args.extra_opt:
                cmd.extend([key, value])

            print("\n==> Running", exp_name)
            print(" ".join(str(x) for x in cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=root, env=env, check=True)
                result = merge_range(exp_dir, args.start, args.end)
                report_path = exp_dir / "range_eval_summary.txt"
                write_step_report(report_path, step, renorm, args.solver, result)
                print(
                    f"steps={step:>2} renorm={bool_cli(renorm):<5} "
                    f"mIoU={result['miou']:.2f} mAcc={result['macc']:.2f} "
                    f"allAcc={result['allacc']:.2f} "
                    f"merged={len(result['used_splits'])}"
                )
                summary_rows.append(
                    {
                        "steps": step,
                        "solver": args.solver,
                        "renorm": bool_cli(renorm),
                        "start": args.start,
                        "end": args.end,
                        "merged_splits": len(result["used_splits"]),
                        "missing_splits": " ".join(map(str, result["missing_splits"])),
                        "mIoU": f"{result['miou']:.4f}",
                        "mAcc": f"{result['macc']:.4f}",
                        "allAcc": f"{result['allacc']:.4f}",
                        "exp_dir": str(exp_dir),
                    }
                )

    if not args.dry_run and summary_rows:
        summary_path = out_root / "sweep_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
