#!/bin/bash
# Resume-friendly GeoFlow evaluation over an arbitrary split range.
# Completed splits are skipped, partial splits are resumed, and failed/OOM
# splits are recorded then skipped so later splits can continue.
#
# Example:
#   bash run/val_geoflow_split_range.sh \
#     --exp_dir=out/geoflow_scannet_planA_max_split312 \
#     --config=config/geoflow_scannet.yaml \
#     --resume=out/geoflow_scannet_planA_max/model/geoflow_last.pth \
#     --splits=312 --start=0 --end=311 --gpu=0

exp_dir=
config=
resume=
splits=312
start=0
end=
gpu=0
merge=1
fail_fast=0
coverage=1
coverage_strict=0
evaluation_list=
extra_args=()

for arg in "$@"; do
    case "$arg" in
        --exp_dir=*) exp_dir="${arg#*=}" ;;
        --config=*) config="${arg#*=}" ;;
        --resume=*) resume="${arg#*=}" ;;
        --ckpt_name=*) ckpt_name="${arg#*=}" ;;
        --splits=*) splits="${arg#*=}" ;;
        --start=*) start="${arg#*=}" ;;
        --end=*) end="${arg#*=}" ;;
        --gpu=*) gpu="${arg#*=}" ;;
        --no-merge) merge=0 ;;
        --fail-fast) fail_fast=1 ;;
        --no-coverage) coverage=0 ;;
        --coverage-strict) coverage_strict=1 ;;
        --evaluation-list=*) evaluation_list="${arg#*=}" ;;
        *) extra_args+=("$arg") ;;
    esac
done

if [ -n "${ckpt_name:-}" ] && [ -z "$resume" ]; then
    resume="${exp_dir}/model/${ckpt_name}"
fi

if [ -z "$exp_dir" ] || [ -z "$config" ] || [ -z "$resume" ]; then
    echo "Usage: bash run/val_geoflow_split_range.sh --exp_dir=OUT --config=CONFIG --resume=CKPT [--splits=312] [--start=0] [--end=311] [--gpu=0] [--no-merge] [--fail-fast]"
    exit 1
fi

if [ -z "$end" ]; then
    end=$((splits - 1))
fi

if [ "$start" -lt 0 ] || [ "$end" -lt "$start" ] || [ "$end" -ge "$splits" ]; then
    echo "Invalid range: start=${start}, end=${end}, splits=${splits}"
    exit 1
fi

if [ ! -f "$resume" ]; then
    echo "Checkpoint not found: $resume"
    exit 1
fi

mkdir -p "$exp_dir"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export PYTHONPATH=".:third_party/sonata:third_party/X-Decoder:third_party/detectron2${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python}"
failed_file="${exp_dir}/geoflow-failed-splits.txt"
summary_file="${exp_dir}/geoflow-split-range-summary.txt"
skipped_done=0
resumed_partial=0
completed_now=0
failed_now=0

: > "$failed_file"

for split_idx in $(seq "$start" "$end"); do
    final_npz="${exp_dir}/eval_raw_split${split_idx}.npz"
    partial_npz="${exp_dir}/eval_raw_split${split_idx}.partial.npz"
    if [ -s "$final_npz" ]; then
        echo "[split ${split_idx}/${splits}] final stats exist, skip: ${final_npz}"
        skipped_done=$((skipped_done + 1))
        continue
    fi

    if [ -f "$final_npz" ]; then
        echo "[split ${split_idx}/${splits}] final stats file is empty; rerun: ${final_npz}"
    elif [ -s "$partial_npz" ]; then
        echo "[split ${split_idx}/${splits}] partial stats exist, resume: ${partial_npz}"
        resumed_partial=$((resumed_partial + 1))
    fi

    echo "[split ${split_idx}/${splits}] evaluating on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "$python_bin" -u run/eval_geoflow.py \
        --config="${config}" \
        --split_idx="${split_idx}" \
        --split_total="${splits}" \
        save_path "${exp_dir}" \
        resume "${resume}" \
        "${extra_args[@]}" \
        2>&1 | tee "${exp_dir}/geoflow-infer-split${split_idx}.log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        if [ "$status" -eq 137 ] || grep -Eqi "out of memory|OutOfMemoryError|CUDA error: out of memory|Killed" "${exp_dir}/geoflow-infer-split${split_idx}.log"; then
            reason="oom-or-killed"
        else
            reason="status-${status}"
        fi
        echo "${split_idx} ${reason}" >> "$failed_file"
        failed_now=$((failed_now + 1))
        echo "[split ${split_idx}/${splits}] failed (${reason}); skip now. Rerun the same command to resume from ${partial_npz}"
        if [ "$fail_fast" -eq 1 ]; then
            exit "$status"
        fi
        continue
    fi
    completed_now=$((completed_now + 1))
done

{
    echo "range=${start}-${end}"
    echo "splits=${splits}"
    echo "skipped_done=${skipped_done}"
    echo "resumed_partial=${resumed_partial}"
    echo "completed_now=${completed_now}"
    echo "failed_now=${failed_now}"
    echo "failed_file=${failed_file}"
} | tee "$summary_file"

if [ "$merge" -eq 1 ]; then
    "$python_bin" run/merge_eval.py --exp_dir "$exp_dir" --splits "$splits"
    "$python_bin" run/merge_ode_timing.py --exp_dir "$exp_dir" --splits "$splits"
fi

if [ "$coverage" -eq 1 ]; then
    if [ -z "$evaluation_list" ]; then
        case "$config" in
            *matterport*) evaluation_list="matterport_evaluation.txt" ;;
            *) evaluation_list="scannet_evaluation.txt" ;;
        esac
    fi
    if [ -f "$evaluation_list" ]; then
        coverage_args=(--exp_dir "$exp_dir" --splits "$splits" --evaluation-list "$evaluation_list")
        if [ "$coverage_strict" -eq 1 ]; then
            coverage_args+=(--strict)
        fi
        "$python_bin" run/check_eval_coverage.py "${coverage_args[@]}"
    else
        echo "Coverage list not found; skip coverage check: ${evaluation_list}"
    fi
fi

if [ "$failed_now" -ne 0 ]; then
    echo "Finished with ${failed_now} failed/OOM split(s). See ${failed_file}."
fi
