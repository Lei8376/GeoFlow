#!/bin/bash
# Evaluation. Runs N splits (optionally one per GPU, in parallel) and then
# merges the raw per-class stats into a single correct mIoU/mAcc.
#
# Usage:
#   sh run/val.sh --exp_dir=out/scannet --config=config/geopurify_scannet.yaml \
#                 --ckpt_name=affinity_predictor_last.pth [--gpus=0,1]

gpus=0
extra_args=()
for arg in "$@"; do
    case $arg in
        --exp_dir=*)  exp_dir="${arg#*=}" ;;
        --config=*)   config="${arg#*=}" ;;
        --ckpt_name=*) ckpt_name="${arg#*=}" ;;
        --gpus=*)     gpus="${arg#*=}" ;;
        *) extra_args+=("$arg") ;;
    esac
done

if [ -z "$exp_dir" ] || [ -z "$config" ] || [ -z "$ckpt_name" ]; then
    echo "Usage: sh run/val.sh --exp_dir=XX --config=XX --ckpt_name=XX [--gpus=0,1]"
    exit 1
fi

mkdir -p "${exp_dir}" "${exp_dir}/model"
export PYTHONPATH=".:third_party/sonata:third_party/X-Decoder:third_party/detectron2${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# Reduce glibc arena growth during long per-scene validation loops (49 GiB box).
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
PYTHON_BIN=${PYTHON_BIN:-python}
IFS=',' read -r -a gpu_arr <<< "$gpus"
split_total=${#gpu_arr[@]}
echo "Evaluating ckpt=$ckpt_name across $split_total split(s) on GPUs: $gpus"

pids=()
for split_idx in $(seq 0 $((split_total - 1))); do
    g=${gpu_arr[$split_idx]}
    CUDA_VISIBLE_DEVICES=${g} "${PYTHON_BIN}" -u run/validation.py \
        --config="${config}" \
        --split_idx="${split_idx}" \
        --split_total="${split_total}" \
        save_path "${exp_dir}" \
        resume "${exp_dir}/model/${ckpt_name}" \
        "${extra_args[@]}" \
        2>&1 | tee "${exp_dir}/infer-${ckpt_name}-split${split_idx}.log" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
if [ "$status" -ne 0 ]; then
    echo "One or more validation split processes failed; skip merge."
    exit "$status"
fi

echo "=== Merging splits ==="
"${PYTHON_BIN}" run/merge_eval.py --exp_dir "${exp_dir}" --splits "${split_total}"
