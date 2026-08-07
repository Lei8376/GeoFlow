#!/bin/sh
# Multi-GPU (DDP) training. Defaults to 2 GPUs (e.g. 2x RTX 4090).
set -x

nproc=2
gpus=0,1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --exp_dir=*) exp_dir="${1#*=}" ;;
        --config=*)  config="${1#*=}"  ;;
        --nproc=*)   nproc="${1#*=}"   ;;
        --gpus=*)    gpus="${1#*=}"    ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$exp_dir" ] || [ -z "$config" ]; then
    echo "Usage: sh run/train_ddp.sh --exp_dir=XX --config=XX [--nproc=2] [--gpus=0,1]"
    exit 1
fi

mkdir -p "${exp_dir}"
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${gpus} torchrun --standalone --nproc_per_node=${nproc} \
  run/train_ddp.py \
  --config="${config}" \
  save_path "${exp_dir}" \
  2>&1 | tee -a "${exp_dir}/train_ddp-$(date +"%Y%m%d_%H%M").log"
