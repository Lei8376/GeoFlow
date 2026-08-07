#!/bin/bash
# Matterport end-to-end: Stage-0 baseline eval -> merge -> Stage-1 GeoFlow train -> Stage-2 eval.
#
# Stage 0 is usually launched separately (hours). This script waits for both
# eval_raw_split{0,1}.npz files, then runs merge + train + GeoFlow eval.
#
# Usage (after Stage-0 val.sh is running or finished):
#   source activate /home/featurize/work/habitat   # or: conda activate /home/featurize/work/habitat
#   cd /home/featurize/work/geoflow/GeoPurify-main
#   export PYTHONPATH=.
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   export MALLOC_ARENA_MAX=2
#   bash run/pipeline_matterport.sh [--skip-wait] [--gpus=0,1]

set -euo pipefail

baseline_dir=out/matterport_baseline
geoflow_dir=out/geoflow_matterport
gpus=0,1
skip_wait=0

for arg in "$@"; do
    case $arg in
        --skip-wait) skip_wait=1 ;;
        --gpus=*) gpus="${arg#*=}" ;;
        --baseline_dir=*) baseline_dir="${arg#*=}" ;;
        --geoflow_dir=*) geoflow_dir="${arg#*=}" ;;
    esac
done

wait_for_npz() {
    local d=$1 n=$2
    while true; do
        ok=1
        for i in $(seq 0 $((n - 1))); do
            if [ ! -f "${d}/eval_raw_split${i}.npz" ]; then
                ok=0
                break
            fi
        done
        if [ "$ok" -eq 1 ]; then
            return 0
        fi
        echo "[pipeline] waiting for ${d}/eval_raw_split{0..$((n-1))}.npz ..."
        sleep 120
    done
}

echo "=== Stage 0: wait for baseline eval npz (2 splits) ==="
if [ "$skip_wait" -eq 0 ]; then
    wait_for_npz "$baseline_dir" 2
fi

echo "=== Stage 0: merge baseline ==="
python run/merge_eval.py --exp_dir "$baseline_dir" --splits 2 | tee "${baseline_dir}/merged_stage0.txt"

echo "=== Stage 1: train GeoFlow ==="
bash run/train_geoflow_matterport.sh \
    --exp_dir="$geoflow_dir" \
    --config=config/geoflow_matterport.yaml \
    --nproc=2 \
    --gpus="$gpus"

echo "=== Stage 2: eval GeoFlow ==="
bash run/val_geoflow.sh \
    --exp_dir="$geoflow_dir" \
    --config=config/geoflow_matterport.yaml \
    --ckpt_name=geoflow_last.pth \
    --gpus="$gpus" | tee "${geoflow_dir}/merged_stage2.txt"

echo "=== Done. Results: ==="
echo "  Baseline: ${baseline_dir}/merged_stage0.txt"
echo "  GeoFlow:  ${geoflow_dir}/merged_stage2.txt (from merge_eval at end of val_geoflow.sh)"
