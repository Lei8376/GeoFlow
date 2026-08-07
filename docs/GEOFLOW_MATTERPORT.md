# GeoFlow on Matterport3D — Technical Implementation Document

> **中文运行手册（推荐日常查阅）**：[MATTERPORT_运行与技术文档.md](MATTERPORT_运行与技术文档.md)

> End-to-end recipe for running the GeoPurify + Flow-Matching (GeoFlow)
> framework on Matterport3D using the Path B target redesign
> `x1 = StudentAffinityPool(x0; phi_S)` and the provided pretrained student
> checkpoint at `/home/featurize/weight/matterport_weight/geopurify.pth`.

---

## 1. Project background

GeoPurify (ICLR 2026, [arXiv:2510.02186](https://arxiv.org/abs/2510.02186)) is a
data-efficient, *label-free* framework for open-vocabulary 3D semantic
segmentation. It lifts 2D vision-language features from a frozen X-Decoder
backbone to 3D, then "purifies" them with geometric structure learned by a
sparse 3D student trained by contrastive distillation from a frozen 3D
self-supervised teacher (Sonata). Reported numbers on Matterport3D test split
are **40.2 mIoU / 62.4 mAcc** with `≈ 1.5%` of the training data.

GeoFlow is our proposed *flow-matching* variant: instead of GeoPurify's fixed
iterated affinity pool, it learns a geometry-conditioned velocity field
`v_theta(x_t, t, c)` and at inference solves an ODE
`dx/dt = v_theta` from the noisy projected feature to a clean,
geometry-consistent endpoint. This document is the runnable, reproducible recipe
for the Matterport3D experiment.

## 2. Original GeoPurify in one diagram

```
                  +-- frozen X-Decoder (DaViT-L) -- 512-d per-view -->-+
multi-view images |                                                   |
                  +---------------- 2D->3D lifting ---------------+   |
                                                                  v   v
3D point cloud  --+-- voxelize ---> [V, 512] noisy semantic = F_sem  |
                  +--- Sonata (frozen) ---> [V, Ds] geometry teacher  |
                  +--- RGB + normal -----> [V, 6]                     |
                                                                      |
                                                                      v
                                      student phi_S : [V, 518] -> [V, 128]
                                      AffinityPool 18 iters of F<-AF
                                      (paper Eq. 4, T=18, K=96, alpha=20)
                                                                      |
                                                                      v
                                          purified F_clean -> cosine(text) -> labels
```

The student `phi_S` (4 Minkowski ResBlocks, hidden 512, output 128) is trained
with hybrid InfoNCE (48 macro + 16 micro negatives, 4096 anchors per scene) on
Sonata-derived positives; no 3D labels are ever used.

## 3. Motivation for Flow Matching

GeoPurify's purification is mathematically a *distribution transport*: take the
noisy 2D-lifted semantic distribution and move it onto the geometry-consistent
feature manifold. Conditional Flow Matching ([Lipman et al., ICLR 2023];
[Tong et al., TMLR 2024]) regresses a velocity field along the straight-line
path between a source and a target sample:

  x_t  = (1 - t) x0 + t x1
  u_t  = x1 - x0       (constant target velocity, no schedule)
  L_fm = || v_theta(x_t, t, c) - u_t ||^2

Three properties make this a particularly natural fit for purification:

1. **No noise schedule.** A simple regression loss, far fewer knobs than
   diffusion (β-schedule, SNR weighting, variational bound).
2. **Few-step ODE inference.** A 2nd-order midpoint solver hits good accuracy
   in ≈ 8 steps; a comparable diffusion model needs ≈ 25 DDIM steps. Because
   purification runs once per scene over many voxels, this is a real wall-clock
   win.
3. **Arbitrary source distribution.** Diffusion needs a Gaussian source and
   recovers semantics through SDEdit (an extra tuned hyperparameter). Flow
   matching lets us set `x0 = noisy lifted feature` directly, so the semantic
   content is *already in the source*; the velocity field only needs to
   relocate it onto the clean manifold.

Verified by `tests/test_geoflow_numpy.py` (F1–F5; all pass on this box). F4
quantifies the few-step ODE advantage on a curved test field: 4 midpoint steps
give error 0.021 vs. 0.231 for Euler — **~11× more accurate per step**.

## 4. Current method — overall framework

```
                                                              X-Decoder (frozen)
                       multi-view --------------------------> F_sem [V, 512]
                                                                  |
2D / 3D Matterport ----- voxelize ----+-- Sonata (frozen) ------> F_son [V, Ds]
                                      |                          |
                                      +-- RGB + normal --------> geom [V, 6]
                                                                  |
                                                                  v
                                          cond = GeoCondEncoder([F_son || geom]) [V, 256]
                                                                  |
phi_S (FROZEN GeoPurify student) ------> x1 = StudentAffinityPool(x0) [V, 512]
                                                                  |
                  x0 = L2norm(F_sem) -----------+                  |
                                                v                  v
                                  CFM training:  v_theta regresses u_t = x1 - x0
                                                conditioned on cond + time t
                                                Loss = MSE + lambda * (1 - cos)
                                                                  |
                                                                  v
                                  Inference (per scene):
                                    x(0) = x0; x(1) = ODE-solve(v_theta, c, x0)
                                    8 midpoint steps, renorm onto sphere
                                    cosine(x(1), text_embeddings) -> labels
```

Only `velocity` (`GeoFlowVelocity`) and `cond_encoder` (`GeoCondEncoder`) are
optimised. X-Decoder, Sonata, and the GeoPurify student are frozen.

### 4.1 Path B target redesign (key novelty vs. previous GeoFlow code)

Previously the clean target was the Sonata-affinity teacher pool. We changed it
to the **GeoPurify student-affinity pool** so that the velocity field is
transported toward exactly the operator GeoPurify uses at inference, lifting
GeoFlow's theoretical ceiling from "teacher pool" to "student pool = paper
numbers". See `_student_manifold_target` in
[models/geoflow_module.py](../models/geoflow_module.py) — it replicates the
purification pipeline of
[models/affinity_module.py](../models/affinity_module.py)
`evaluate_scene` step-for-step: KNN(K=96) → cosine(student embed) →
softmax(α=20) → 18 iterations of `F ← AF` → first 512 dims → L2-normalize.

Unit-test for the redesign: `tests/test_geoflow_numpy.py` F6 (passes on this
box).

## 5. Dataset preparation

- 2D: `/home/featurize/data/matterport_2d/<scene>/{color,depth,intrinsic,pose}/*`
- 3D: `/home/featurize/data/matterport_3d/{train,val,test}/<region>.pth`
  (1554 / 234 / 406 .pth files respectively, one per region)
- Train scenes used: 20 regions listed in
  [matterport_train.txt](../matterport_train.txt).
- Eval scenes: 406 regions in
  [matterport_evaluation.txt](../matterport_evaluation.txt) (the standard
  Matterport3D test split).
- Class set: 21-way (`test_classes: 21`, ignore label `255`) — identical to
  the GeoPurify paper's Matterport setup.
- Class names and ID->name mapping in
  [config/geoflow_matterport.yaml](../config/geoflow_matterport.yaml) under
  `all_label` and the matching `category_split`.

The Matterport 2D/3D paths in `config/geopurify_matterport.yaml`,
`config/geoflow_matterport.yaml`, and `config/fusion_matterport.yaml` are
hardcoded to the **absolute** paths above so no symlinking or environment
variables are needed.

## 6. Environment requirements

Tested against the GeoPurify reference stack (`docs/Install.md`):

- Python 3.9 / 3.10
- CUDA toolkit 11.8 or 12.1 with `nvcc` (required to build MinkowskiEngine)
- `torch 2.2.x / 2.5.x` + CUDA build, `torch-scatter`, `MinkowskiEngine`
  (from source), `flash-attn` (optional; falls back if absent),
  `faiss-cpu` (or faiss-gpu), `open3d`, `clip`, `detectron2`,
  `tensorboardX`, `omegaconf`, `scikit-learn`, `SharedArray`, `imageio`,
  `opencv-python`, `tqdm`, plus the `sonata` and `X-Decoder` submodules in
  `third_party/`.

Important: **MinkowskiEngine must be built against the CUDA toolkit that
matches your `torch` build**. The provided GeoPurify student checkpoint at
`/home/featurize/weight/matterport_weight/geopurify.pth` was created with this
stack; reuse the same image / conda env to avoid sparse-conv kernel mismatch.

Sanity check after installation:

```bash
python -c "
import torch, MinkowskiEngine, sonata, torch_scatter, faiss
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
print('ME', MinkowskiEngine.__version__)
print('sonata', sonata.__file__)
"
```

## 7. Stage 0 — verify the provided GeoPurify student (~2–4 h on 2 GPUs)

This produces the **reference number** GeoFlow must match or beat. Eval-only;
the student weights are taken as-is.

```bash
cd /home/featurize/work/geoflow/GeoPurify-main
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Stage the provided checkpoint where val.sh expects it (file name only
#    needs to match the --ckpt_name argument below; we keep the original name).
mkdir -p out/matterport_baseline/model
cp /home/featurize/weight/matterport_weight/geopurify.pth \
   out/matterport_baseline/model/geopurify.pth

# 2. Run validation over 2 splits in parallel (one per GPU). The script
#    auto-detects Matterport from data_root and pulls scenes from
#    matterport_evaluation.txt.
bash run/val.sh \
   --exp_dir=out/matterport_baseline \
   --config=config/geopurify_matterport.yaml \
   --ckpt_name=geopurify.pth \
   --gpus=0,1
```

Acceptance criteria (printed at the end by `run/merge_eval.py`):

- mIoU within ±1 of **40.2**
- mAcc within ±1.5 of **62.4**
- allAcc and per-class IoU sanity-printed; wall/floor/ceiling > 70 IoU.

If those numbers don't match, **stop**: data paths or class mapping are wrong
and any GeoFlow number we report afterwards is meaningless.

### 7.1 Memory budget for 2-GPU parallel eval (important)

Each evaluation process is single-threaded over scenes (the
`SceneBatchSampler` yields all views of one scene back-to-back), and the
dataset holds 203 scenes × ~100 views each on this split. The historical
`ScannetLoaderFull` was materialising every `Camera` (with its ~3.7 MB
`original_image` tensor) at `__init__`, which on this 49 GB box ballooned
each process to ~30 GB and OOM-killed the second one silently when both
GPUs were used. The current implementation in
[`dataset/data_loader_matterport.py`](../dataset/data_loader_matterport.py)
stores only camera **metadata** (`scene_name`, `view_idx`, `intrinsics`) in
`self.samples` and lazy-loads each image inside `__getitem__` via the cached
`SceneDataset`. Confirmed memory footprint on this box: ~10 GB / process,
~24 GB free system-wide with both GPUs running — comfortably parallel.

### 7.2 Stage-0 result on this box

Filled in by `run/merge_eval.py` once both splits finish; see
`out/matterport_baseline/eval_results.txt`.

## 8. Stage 1 — train GeoFlow with student-affinity target (~6–10 h)

```bash
cd /home/featurize/work/geoflow/GeoPurify-main
export PYTHONPATH=.
sh run/train_geoflow_matterport.sh \
   --exp_dir=out/geoflow_matterport \
   --config=config/geoflow_matterport.yaml \
   --nproc=2 --gpus=0,1
```

What this does:

- Loads `SonataXGeoFlowTrainer` with `use_student_target=True`
  and `affinity_ckpt=/home/featurize/weight/matterport_weight/geopurify.pth`.
- Freezes X-Decoder, Sonata, and the GeoPurify student.
- For each scene: builds `[F_sem(512) || geom(6)]`, computes
  `x1 = StudentAffinityPool(x0)` once under `torch.no_grad()`, samples
  `t ~ U(0, 1)`, builds the OT interpolant `x_t`, regresses `v_theta` to
  `u_t = x1 - x0`, adds `manifold_loss_w * (1 - cos(x_t + (1-t) v_theta, x1))`.
- DDP across 2 GPUs via `DistributedSceneBatchSampler` (whole scenes
  partitioned disjointly across ranks; one scene == one optimizer step).
- AMP enabled (`use_amp: True`) to fit a 10 GiB RTX 3080.

Monitoring: `tensorboard --logdir out/geoflow_matterport`. Scalars to watch:

- `loss_fm_iter` — should drop from ~1e0 to ~1e-2 within the first few epochs
  on Matterport; mid-training it sits around 1e-2 – 1e-3.
- `loss_manifold_iter` — should fall below 0.1 by epoch ~5 and below 0.02 by
  epoch ~20.
- `cos_hat_x1_iter` — should rise above 0.95 by epoch 10 and above 0.99 by
  epoch 30.
- `lr` — warmup → cosine, with `lr_3d = 2e-4` peak (see
  `config/geoflow_matterport.yaml`).

Checkpoints:

- `out/geoflow_matterport/model/geoflow_last.pth` (updated every epoch)
- `out/geoflow_matterport/model/geoflow_epoch_{N}.pth` (every 5 epochs)

## 9. Stage 2 — evaluate GeoFlow on Matterport test (~30–60 min)

```bash
cd /home/featurize/work/geoflow/GeoPurify-main
export PYTHONPATH=.
sh run/val_geoflow.sh \
   --exp_dir=out/geoflow_matterport \
   --config=config/geoflow_matterport.yaml \
   --ckpt_name=geoflow_last.pth \
   --gpus=0,1
```

This splits 406 evaluation scenes across 2 GPUs (each rank writes
`eval_raw_split{0,1}.npz`), and `run/merge_eval.py` merges them into the final
per-class IoU / mIoU / mAcc / allAcc table.

## 10. Metric definitions (paper-compatible)

- `intersection`, `union`, `target` are accumulated per scene via
  [util/util.py](../util/util.py) `intersectionAndUnionGPU(pred, label, K, ignore_indexs)`.
- Final scalars by `merge_eval.py`:
  - `IoU_c = intersection_c / (union_c + 1e-10)`
  - `Acc_c = intersection_c / (target_c + 1e-10)`
  - `mIoU = mean_c(IoU_c)`, `mAcc = mean_c(Acc_c)`,
    `allAcc = sum(intersection) / sum(target)`.
- `K = test_classes = 21` for Matterport; `ignore_indexs = [255]`.
- This is the **only correct way** to aggregate across eval splits — averaging
  per-split mIoUs is biased. The implementation is the same one GeoPurify uses
  in `run/validation.py`.

## 11. Troubleshooting tree

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage-0 mIoU < 35 | Wrong data_root or class map | Confirm `data_root: /home/featurize/data/matterport_3d`; check `category_split.all_category` matches yaml |
| `0 file is loaded in the point loader` | Path wrong / wrong split | `ls /home/featurize/data/matterport_3d/{train,val,test} | wc -l` — should give 1554/234/406 |
| `KeyError: model_state_dict` when loading student | Newer checkpoint format | The fallback in `_build_and_freeze_student` already handles both `dict-with-model_state_dict` and bare state-dict formats |
| `loss_fm` flat near initial value | `t` sampling degenerate / cond_encoder dead | Increase `manifold_loss_w` to 0.5; verify `cond_feats.std()` > 0 |
| `cos_hat_x1` stuck below 0.5 by epoch 10 | Student target mismatch | Run `tests/test_geoflow_numpy.py` F6; verify checkpoint loads with **no** missing keys |
| OOM on 10 GiB | `geodiff_hidden=512` too big | Reduce to 384 or 256; the Matterport config already uses 384 |
| Stage-2 mIoU < stage-0 by > 1 | Train/test mismatch in ODE drift | Try `geoflow_ode_steps=16`, `geoflow_renorm_each_step: True` |
| Stage-2 mIoU near stage-0 but mAcc lower | Class confusion at sphere boundary | Add `geoflow_sigma_min: 0.02` (OT-CFM noise) |
| `KDTree`/faiss segfault | mismatched libstdc++ | reinstall `faiss-cpu` from the env's pip |

## 12. How to tell whether the model is really learning

Within the first epoch you should see:

1. `loss_fm` strictly decreasing every 5 print steps.
2. `cos_hat_x1` strictly increasing.
3. Gradient norms of `velocity.input` and `velocity.out` non-zero (TB log).
4. Loss curve smooth — flow matching has no noise schedule so the loss should
   not "tooth" the way diffusion loss does.

By epoch 5 a quick eval on 10 held-out scenes (run `eval_geoflow.py` with a
short subset) should give mIoU > 35 on Matterport; otherwise investigate per
section 11.

## 13. Automated pipeline (Stage 0 → 1 → 2)

After Stage-0 `val.sh` is running (or finished), launch the waiter that merges
baseline numbers, trains GeoFlow, and evaluates it:

```bash
conda activate /home/featurize/work/habitat
cd /home/featurize/work/geoflow/GeoPurify-main
export PYTHONPATH=.
bash run/pipeline_matterport.sh --gpus=0,1
```

This blocks until `out/matterport_baseline/eval_raw_split{0,1}.npz` exist, then
runs `merge_eval.py`, `train_geoflow_matterport.sh`, and `val_geoflow.sh`.
Logs: `out/pipeline_matterport.log`.

`run/val.sh` now exports `MALLOC_ARENA_MAX=2` and uses
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so two parallel validation
processes stay under ~49 GiB system RAM on the target box.

## 14. Reproducibility recipe

1. Verify the env (section 6).
2. Confirm the data paths return the expected counts (section 11 row 2).
3. Run the math test: `python tests/test_geoflow_numpy.py` → all 6 must PASS.
4. Run stage-0; record mIoU/mAcc/allAcc and per-class IoU.
5. Run stage-1 with the seeds in `config/geoflow_matterport.yaml`
   (`manual_seed: 5557`).
6. Run stage-2; produce per-class IoU and the merged scalars.
7. Verify (math test) is still PASS after any code edits.

## 15. Fair-comparison protocol against the GeoPurify paper

| Method | Backbone | Purifier | Inference steps | Reported / observed Matterport |
|---|---|---|---|---|
| GeoPurify (paper) | X-Decoder (frozen) + Sonata | learned student affinity pool | 18 (F ← AF) | **40.2 mIoU / 62.4 mAcc** |
| GeoPurify (this repo, stage-0) | identical | identical | identical | filled in by section 7 |
| GeoFlow (this work) | identical | learned velocity field, ODE | 8 midpoint | filled in by section 9 |

Comparison rules:

- Same evaluation scene list (`matterport_evaluation.txt`).
- Same class set (21-way, ignore=255).
- Same 2D feature extractor (frozen X-Decoder DaViT-L).
- Same KDTree unseen-point fill (already shared across both eval scripts).
- Both purifiers consume the same lifted `F_sem` (identical 2D->3D pipeline).
- Only the **purification operator** differs (student pool vs. velocity field
  ODE).

This isolates the contribution of flow matching from any 2D / lifting changes.

## 16. Performance roadmap to match / exceed 40.2 / 62.4

In order of expected payoff:

1. **Confirm stage-0 = paper.** If the provided ckpt does not match, no
   GeoFlow number is comparable.
2. **Stage-1 baseline run** with `use_student_target: True`,
   `geoflow_ode_steps: 8`, `geoflow_manifold_loss_w: 0.1`,
   `geoflow_sigma_min: 0.0` — expected ≤ ±0.5 mIoU of stage-0.
3. If stage-2 mIoU < stage-0:
   - Bump `geoflow_ode_steps` from 8 to 16 (no retrain).
   - Bump `geoflow_manifold_loss_w` from 0.1 to 0.3 and retrain.
4. If stage-2 mIoU ≈ stage-0 but mAcc lower: set `geoflow_sigma_min: 0.02`
   (OT-CFM regularity) and retrain.
5. To push past stage-0: per-voxel time sampling
   (replace `t = torch.rand(1)` with `t = torch.rand(V, 1)` in
   `models/geoflow_module.py:forward`) and/or 1-step consistency distillation
   as a short fine-tune of the trained model.

## 17. File map

- New / modified by this implementation:
  - [config/geoflow_matterport.yaml](../config/geoflow_matterport.yaml) **new**
  - [config/geopurify_matterport.yaml](../config/geopurify_matterport.yaml) — paths fixed, caption/entity optional
  - [config/fusion_matterport.yaml](../config/fusion_matterport.yaml) — paths fixed
  - [models/geoflow_module.py](../models/geoflow_module.py) — Path B target + per-loss logging + frozen student
  - [models/affinity_module.py](../models/affinity_module.py) — chunked feature lifting + per-voxel affinity to avoid OOM on long views / 200k-point scenes
  - [models/utils/dataset_utils.py](../models/utils/dataset_utils.py) — `fetchPth` accepts the Matterport 3-tuple `(coords, colors, labels)` and estimates normals on-the-fly with Open3D
  - [dataset/data_loader_matterport.py](../dataset/data_loader_matterport.py) — captions optional + 3-tuple `.pth` support + **lazy view loading** (stores camera metadata only, not loaded images)
  - [run/train_geoflow.py](../run/train_geoflow.py) — dataset-aware + per-loss TB + AMP
  - [run/eval_geoflow.py](../run/eval_geoflow.py) — robust ckpt load + per-class print
  - [run/train_geoflow_matterport.sh](../run/train_geoflow_matterport.sh) **new**
  - [run/validation.py](../run/validation.py) — guard against empty `val_loader` (initialise `mIoU_2d_*` before the loop)
  - [tests/test_geoflow_numpy.py](../tests/test_geoflow_numpy.py) — adds F6 (student target unit test)
- Untouched (the operator we reuse / compare against):
  - [models/geodiff_module.py](../models/geodiff_module.py) (parent of geoflow)
  - [run/train.py](../run/train.py)
  - [run/merge_eval.py](../run/merge_eval.py)
