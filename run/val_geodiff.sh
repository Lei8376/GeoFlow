#!/bin/bash
# GeoDiff evaluation across N splits (one per GPU), then merge.
gpus=0
for arg in "$@"; do
    case $arg in
        --exp_dir=*)  exp_dir="${arg#*=}" ;;
        --config=*)   config="${arg#*=}" ;;
        --ckpt_name=*) ckpt_name="${arg#*=}" ;;
        --gpus=*)     gpus="${arg#*=}" ;;
    esac
done
[ -z "$exp_dir" ] || [ -z "$config" ] || [ -z "$ckpt_name" ] && { echo "Usage: sh run/val_geodiff.sh --exp_dir=XX --config=config/geodiff_scannet.yaml --ckpt_name=geodiff_last.pth [--gpus=0,1]"; exit 1; }
export PYTHONPATH=.
IFS=',' read -r -a gpu_arr <<< "$gpus"; split_total=${#gpu_arr[@]}
pids=()
for split_idx in $(seq 0 $((split_total - 1))); do
    g=${gpu_arr[$split_idx]}
    CUDA_VISIBLE_DEVICES=${g} python -u run/eval_geodiff.py \
        --config="${config}" --split_idx="${split_idx}" --split_total="${split_total}" \
        save_path "${exp_dir}" resume "${exp_dir}/model/${ckpt_name}" \
        2>&1 | tee "${exp_dir}/geodiff-infer-split${split_idx}.log" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python run/merge_eval.py --exp_dir "${exp_dir}" --splits "${split_total}"
