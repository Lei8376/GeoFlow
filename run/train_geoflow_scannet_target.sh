#!/usr/bin/env bash
set -euo pipefail

# Start GeoFlow ScanNet training with an explicit target-oracle choice.
#
# Usage:
#   bash run/train_geoflow_scannet_target.sh out/geoflow_scannet_k64_a40_i3 64 40 3
#   bash run/train_geoflow_scannet_target.sh out/geoflow_scannet_k96_a40_i3 96 40 3 2 0,1
#
# Args:
#   1: exp dir
#   2: target K
#   3: target alpha
#   4: target iterations
#   5: nproc, optional, default 1
#   6: gpus, optional, default 0

EXP_DIR="${1:?Usage: bash run/train_geoflow_scannet_target.sh EXP_DIR K ALPHA ITERS [NPROC] [GPUS]}"
K="${2:?Missing K}"
ALPHA="${3:?Missing ALPHA}"
ITERS="${4:?Missing ITERS}"
NPROC="${5:-1}"
GPUS="${6:-0}"
CONFIG="${CONFIG:-config/geoflow_scannet.yaml}"

mkdir -p "${EXP_DIR}"
export PYTHONPATH=".:third_party/sonata:third_party/X-Decoder:third_party/detectron2${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments* ]]; then
  echo "Unset unsupported PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  unset PYTORCH_CUDA_ALLOC_CONF
fi

echo "exp_dir=${EXP_DIR}"
echo "config=${CONFIG}"
echo "target=pool:k=${K},a=${ALPHA},it=${ITERS}"
echo "nproc=${NPROC}"
echo "gpus=${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
    run/train_geoflow.py \
    --config "${CONFIG}" \
    save_path "${EXP_DIR}" \
    geodiff_target_knn "${K}" \
    geodiff_target_alpha "${ALPHA}" \
    geodiff_target_iters "${ITERS}" \
    2>&1 | tee -a "${EXP_DIR}/train_geoflow-k${K}-a${ALPHA}-i${ITERS}-$(date +%Y%m%d_%H%M).log"
else
  CUDA_VISIBLE_DEVICES="${GPUS}" python run/train_geoflow.py \
    --config "${CONFIG}" \
    save_path "${EXP_DIR}" \
    geodiff_target_knn "${K}" \
    geodiff_target_alpha "${ALPHA}" \
    geodiff_target_iters "${ITERS}" \
    2>&1 | tee -a "${EXP_DIR}/train_geoflow-k${K}-a${ALPHA}-i${ITERS}-$(date +%Y%m%d_%H%M).log"
fi
