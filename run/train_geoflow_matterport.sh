#!/bin/sh
# GeoFlow Matterport training launcher (defaults to 2 GPUs, geoflow_matterport.yaml).
set -x
nproc=2; gpus=0,1; exp_dir=out/geoflow_matterport; config=config/geoflow_matterport.yaml
resume=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --exp_dir=*) exp_dir="${1#*=}" ;;
        --config=*)  config="${1#*=}"  ;;
        --resume=*)  resume="${1#*=}"  ;;
        --nproc=*)   nproc="${1#*=}"   ;;
        --gpus=*)    gpus="${1#*=}"    ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done
mkdir -p "${exp_dir}"; export PYTHONPATH=".:third_party/sonata:third_party/X-Decoder:third_party/detectron2${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -n "${resume}" ]; then
  CUDA_VISIBLE_DEVICES=${gpus} torchrun --standalone --nproc_per_node=${nproc} \
    run/train_geoflow.py --config="${config}" save_path "${exp_dir}" resume "${resume}" \
    2>&1 | tee -a "${exp_dir}/train_geoflow-$(date +%Y%m%d_%H%M).log"
else
  CUDA_VISIBLE_DEVICES=${gpus} torchrun --standalone --nproc_per_node=${nproc} \
    run/train_geoflow.py --config="${config}" save_path "${exp_dir}" \
    2>&1 | tee -a "${exp_dir}/train_geoflow-$(date +%Y%m%d_%H%M).log"
fi
