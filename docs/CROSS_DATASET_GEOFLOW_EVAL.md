# Cross-Dataset GeoFlow Evaluation

## Default ScanNet Checkpoint

The ScanNet epoch184 checkpoint used for the 312-split result
`mIoU=50.99 / mAcc=67.71 / allAcc=77.98` is:

```text
/home/sunl/work/geoflow/GeoPurify-main/out/geoflow_ft_k96_a20_i10_m025_confw085v015_min015_sigma0025_allviews_v180224_target256_loss1024_from_sigma005_ep109_to600/model/geoflow_epoch_184.pth
```

## Run Both Directions

```bash
cd /home/sunl/work/geoflow/GeoPurify-main

GPU=0 bash run_eval_cross_dataset.sh \
  --direction both \
  --out-root out/cross_dataset_geoflow
```

This runs:

```text
ScanNet-trained checkpoint     -> Matterport3D test split
Matterport-trained checkpoint  -> ScanNetV2 evaluation/val split
```

ScanNet benchmark test labels are not locally available, so the ScanNet target
direction reports metrics on `scannet_evaluation.txt`.

## Recorded Result: Matterport Epoch114 -> ScanNet

Archive checked on 2026-07-19:

```text
/home/sunl/work/geoflow/GeoPurify-main/out/matterport_epoch114_to_scannet_eval_split20.tar.gz
```

Extracted result directory:

```text
/home/sunl/work/geoflow/GeoPurify-main/out/matterport_epoch114_to_scannet_eval_split20
```

Run metadata from the archive:

```text
direction=matterport-to-scannet
source_ckpt=/home/featurize/work/checkpoints/geoflow_epoch_114_matterport.pth
target_dataset=scannet
splits=20
evaluation_list=scannet_evaluation.txt
```

Coverage:

```text
complete=True
final_splits=20
final_scene_coverage=312/312
partial_splits=0
missing_or_empty_splits=0
bad_splits=0
```

Merged ScanNet result over 20 splits:

```text
mIoU = 47.99
mAcc = 66.65
allAcc = 75.44
```

Per-class IoU on ScanNet 19 classes:

| id | class | IoU |
|---:|---|---:|
| 0 | wall | 68.1 |
| 1 | floor | 73.9 |
| 2 | cabinet | 46.1 |
| 3 | bed | 65.4 |
| 4 | chair | 56.0 |
| 5 | sofa | 58.2 |
| 6 | table | 40.2 |
| 7 | door | 50.4 |
| 8 | window | 54.1 |
| 9 | bookshelf | 57.3 |
| 10 | picture | 3.2 |
| 11 | counter | 32.5 |
| 12 | desk | 27.0 |
| 13 | curtain | 59.2 |
| 14 | refrigerator | 34.8 |
| 15 | shower curtain | 41.3 |
| 16 | toilet | 61.9 |
| 17 | sink | 41.2 |
| 18 | bathtub | 41.1 |

Merge command:

```bash
python run/merge_eval.py \
  --exp_dir out/matterport_epoch114_to_scannet_eval_split20 \
  --splits 20
```

## Specify A Custom Source Checkpoint

For one direction, use `--source-ckpt`:

```bash
GPU=0 bash run_eval_cross_dataset.sh \
  --direction scannet-to-matterport \
  --source-ckpt /absolute/path/to/scannet_or_other_source_checkpoint.pth \
  --source-tag my_source_name \
  --out-root out/cross_dataset_geoflow
```

```bash
GPU=0 bash run_eval_cross_dataset.sh \
  --direction matterport-to-scannet \
  --source-ckpt /absolute/path/to/matterport_or_other_source_checkpoint.pth \
  --source-tag my_source_name \
  --out-root out/cross_dataset_geoflow
```

For both directions at once, set the two source checkpoints separately:

```bash
GPU=0 bash run_eval_cross_dataset.sh \
  --direction both \
  --scannet-ckpt /absolute/path/to/scannet_checkpoint.pth \
  --scannet-tag scannet_custom \
  --matterport-ckpt /absolute/path/to/matterport_checkpoint.pth \
  --matterport-tag matterport_custom \
  --out-root out/cross_dataset_geoflow
```

## Partial Runs

Matterport target uses 20 splits by default:

```bash
GPU=0 bash run_eval_cross_dataset.sh \
  --direction scannet-to-matterport \
  --matterport-start 0 \
  --matterport-end 0 \
  --no-merge
```

ScanNet target uses 312 splits by default:

```bash
GPU=0 bash run_eval_cross_dataset.sh \
  --direction matterport-to-scannet \
  --scannet-start 0 \
  --scannet-end 10 \
  --no-merge
```
