#!/bin/bash
# GeoFlow evaluation across N splits (one per GPU), then merge.
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
[ -z "$exp_dir" ] || [ -z "$config" ] || [ -z "$ckpt_name" ] && { echo "Usage: sh run/val_geoflow.sh --exp_dir=XX --config=config/geoflow_scannet.yaml --ckpt_name=geoflow_last.pth [--gpus=0,1]"; exit 1; }
export PYTHONPATH=".:third_party/sonata:third_party/X-Decoder:third_party/detectron2${PYTHONPATH:+:${PYTHONPATH}}"
IFS=',' read -r -a gpu_arr <<< "$gpus"; split_total=${#gpu_arr[@]}
pids=()
for split_idx in $(seq 0 $((split_total - 1))); do
    g=${gpu_arr[$split_idx]}
    CUDA_VISIBLE_DEVICES=${g} python -u run/eval_geoflow.py \
        --config="${config}" --split_idx="${split_idx}" --split_total="${split_total}" \
        save_path "${exp_dir}" resume "${exp_dir}/model/${ckpt_name}" \
        "${extra_args[@]}" \
        2>&1 | tee "${exp_dir}/geoflow-infer-split${split_idx}.log" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
if [ "$status" -ne 0 ]; then
    echo "One or more GeoFlow evaluation split processes failed; rerun the same command to resume from partial stats."
    exit "$status"
fi
python run/merge_eval.py --exp_dir "${exp_dir}" --splits "${split_total}"
