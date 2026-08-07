#!/bin/sh
# GeoDiff multi-GPU (DDP) training. Defaults to 2 GPUs.
set -x
nproc=2; gpus=0,1
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
[ -z "$exp_dir" ] || [ -z "$config" ] && { echo "Usage: sh run/train_geodiff.sh --exp_dir=XX --config=config/geodiff_scannet.yaml [--nproc=2] [--gpus=0,1]"; exit 1; }
mkdir -p "${exp_dir}"; export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=${gpus} torchrun --standalone --nproc_per_node=${nproc} \
  run/train_geodiff.py --config="${config}" save_path "${exp_dir}" \
  2>&1 | tee -a "${exp_dir}/train_geodiff-$(date +%Y%m%d_%H%M).log"
