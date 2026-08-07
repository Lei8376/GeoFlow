"""GeoFlow: Geometry-Conditioned Flow Matching for 3D Feature Purification.

Sibling of GeoDiff, both extending GeoPurify. Purification is recast as a
distribution transport: from the source distribution of noisy 2D->3D projected
semantics to a geometry-consistent feature manifold. A conditional flow
matching (CFM) velocity field v_theta(x, t, c) is regressed to the straight-line
target velocity (x1 - x0) conditioned on geometry
c = Enc(Sonata (+) RGB (+) normal); at inference we solve dx/dt = v_theta from
x0 (the noisy lifted feature) to t = 1 with a few-step ODE solver.

Conditional Flow Matching (Lipman et al. 2023; Tong et al. 2023):
    x_t   = (1 - t) x0 + t x1            (OT / straight-line interpolant)
    u_t   = x1 - x0                      (constant target velocity)
    L_fm  = || v_theta(x_t, t, c) - u_t ||^2
    L_man = 1 - cos(x_t + (1 - t) v_theta, x1)
Inference: ODE Euler/midpoint from t=0 to t=1, renorm on the unit sphere each
step; the cosine head downstream consumes a normalized feature.

Target choice (key design choice):
    use_student_target=True:
        x1 = StudentAffinityPool(x0; phi_S)
        phi_S is the FROZEN GeoPurify affinity student. This is *exactly* the
        operator that produces GeoPurify's published 40.2 mIoU / 62.4 mAcc on
        Matterport3D, so a well-fit velocity field reproduces GeoPurify; an
        imperfect one can still help via the manifold cosine loss and the
        smoothing effect of straight-line CFM transport.
    use_student_target=False (or no affinity_ckpt):
        x1 = SonataAffinityPool(x0)   (the original GeoDiff teacher target)

Heavy backbones (X-Decoder, Sonata) and feature lifting are reused from
SonataXGeoDiffTrainer (which subclasses SonataXAffinityTrainer); only the
velocity field, condition encoder, and (optional) frozen student are added.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import faiss

from models.geodiff_module import (
    SonataXGeoDiffTrainer, GeoCondEncoder, SinusoidalTimeEmbedding, _rebuild)
from models.affinity_module import MinkowskiResBlock, AffinityPredictor


# --------------------------------------------------------------------------- #
#  Velocity field network (geometry- and time-conditioned)
# --------------------------------------------------------------------------- #
class GeoFlowVelocity(nn.Module):
    """Sparse 3D velocity field v(x_t, t, c) -> R^{feat_dim}.

    Same backbone shape as the GeoDiff denoiser, but predicts a VELOCITY
    (x1 - x0 direction) rather than noise. Time t is a continuous scalar in
    [0, 1] (not a discrete diffusion index).
    """

    def __init__(self, feat_dim=512, cond_dim=256, hidden=512, time_dim=256, n_blocks=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )# 256 dim
        self.input = nn.Sequential(
            ME.MinkowskiConvolution(feat_dim + cond_dim, hidden, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(hidden),
            ME.MinkowskiReLU(),
        )
        self.blocks = nn.ModuleList([MinkowskiResBlock(hidden) for _ in range(n_blocks)])
        self.films = nn.ModuleList([nn.Linear(time_dim, hidden * 2) for _ in range(n_blocks)])
        self.out = ME.MinkowskiConvolution(hidden, feat_dim, kernel_size=1, dimension=3)

    def forward(self, x_t_sparse, cond_feats, t_scalar):
        # Embed t * 1000 for numerical range parity with the diffusion embedding.
        temb = self.time_mlp(t_scalar * 1000.0)        # [1, time_dim]
        h_in = torch.cat([x_t_sparse.F, cond_feats], dim=1)
        h = self.input(_rebuild(x_t_sparse, h_in))
        del h_in
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            gamma, beta = film(temb).chunk(2, dim=-1)
            h_feats = h.F * (1 + gamma)
            h_feats.add_(beta)
            h = _rebuild(h, h_feats)
            del h_feats, gamma, beta
        return self.out(h).F                           # [V, feat_dim]


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class SonataXGeoFlowTrainer(SonataXGeoDiffTrainer):
    """Geometry-conditioned flow matching with a frozen-student target.

    Reuses the frozen backbones, lifting, and (optionally) the GeoDiff teacher
    target from SonataXGeoDiffTrainer. Adds:
      * a 512-d velocity field with FiLM time-conditioning
      * an ODE integrator (Euler / midpoint) for inference
      * an *optional* frozen GeoPurify affinity student whose embeddings define
        the clean target x1 via the same pool operator GeoPurify uses at
        inference time.
    """

    def __init__(self, cfg, xdecoder_cfg, scene_config, device="cuda", use_lseg=False):
        super().__init__(cfg, xdecoder_cfg, scene_config, device, use_lseg)
        # The diffusion-specific pieces are unused here.
        if hasattr(self, "denoiser"):
            del self.denoiser
        if hasattr(self, "diffusion"):
            del self.diffusion

        # Flow-matching hyper-parameters.
        self.ode_steps = int(getattr(cfg, "geoflow_ode_steps", 8))
        self.ode_solver = str(getattr(cfg, "geoflow_ode_solver", "midpoint"))
        self.sigma_min = float(getattr(cfg, "geoflow_sigma_min", 0.0))
        self.manifold_loss_w = float(getattr(cfg, "geoflow_manifold_loss_w", 0.1))
        self.renorm_each_step = bool(getattr(cfg, "geoflow_renorm_each_step", True))

        # Student-target hyper-parameters (Path B in the plan).
        self.use_student_target = bool(getattr(cfg, "use_student_target", False))
        self.affinity_ckpt = getattr(cfg, "affinity_ckpt", None)
        self.student_pool_iters = int(getattr(cfg, "pooling_iterations", 18))
        self.student_alpha = float(getattr(cfg, "affinity_sharpen_factor", 20.0))
        self.student_knn = int(getattr(cfg, "pooling_knn", 96))
        self.student_affinity_chunk_v = int(getattr(cfg, "student_affinity_chunk_v", 8192))
        self.loss_chunk_v = int(getattr(cfg, "geoflow_loss_chunk_v", 32768))
        self.train_max_voxels = int(getattr(cfg, "geoflow_train_max_voxels", 65536))
        self.eval_max_voxels = int(getattr(cfg, "geoflow_eval_max_voxels", 0) or 0)
        self.confidence_weighting = bool(getattr(cfg, "geoflow_confidence_weighting", False))
        self.confidence_min = float(getattr(cfg, "geoflow_confidence_min", 0.05))
        self.confidence_power = float(getattr(cfg, "geoflow_confidence_power", 1.0))

        self.velocity = GeoFlowVelocity(
            feat_dim=self.feat_dim, cond_dim=self.cond_dim,
            hidden=self.hidden, n_blocks=self.n_blocks,
        ).to(device)
        print("GeoFlow velocity field created (flow matching, ODE inference).")

        # Lazily attach the frozen affinity student if requested.
        self._student_loaded = False
        self.affinity_student = None
        # The parent SonataXAffinityTrainer already builds a self.affinity_student
        # so we replace it with a freshly-built one and load weights.
        if self.use_student_target and self.affinity_ckpt:
            self._build_and_freeze_student()

    # -------------------------------------------------------------------- #
    #  Frozen student helpers (Path B target operator)
    # -------------------------------------------------------------------- #
    def _build_and_freeze_student(self):
        """Build the GeoPurify AffinityPredictor and load a frozen checkpoint.

        Same hyper-parameters as SonataXAffinityTrainer (input_dim=518=512+6,
        hidden=512, embed=128). The checkpoint format matches what
        run/train.py saves: dict with 'model_state_dict'.
        """
        student = AffinityPredictor(input_dim=512 + 6, embed_dim=128, hidden_dim=512).to(self.device)
        ck_path = self.affinity_ckpt
        if not os.path.isfile(ck_path):
            print(f"[GeoFlow] WARNING affinity_ckpt='{ck_path}' not found; "
                  f"falling back to Sonata teacher target.")
            self.use_student_target = False
            self.affinity_student = None
            return
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        missing, unexpected = student.load_state_dict(state, strict=False)
        if missing:
            print(f"[GeoFlow] student missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"[GeoFlow] student unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
        student.eval()
        for p in student.parameters():
            p.requires_grad = False
        self.affinity_student = student
        self._student_loaded = True
        print(f"[GeoFlow] frozen GeoPurify student loaded from {ck_path}")

    @torch.no_grad()
    def _student_manifold_target(self, F_sem_vox, geom_vox, coords_vox):
        """Label-free clean target via the GeoPurify student-affinity pool.

        Mirrors SonataXAffinityTrainer.evaluate_scene's purification pipeline so
        that ``x1`` is exactly the per-voxel feature the trained GeoPurify model
        would emit on this scene. Returned tensor is L2-normalized along dim=1.

            student input  : [F_sem_vox(512) || geom_vox(6)] sparse, dim=518
            student output : 128-d voxel embedding
            affinity       : cosine over K spatial KNN, softmax(alpha=20)
            pool           : F <- A F  for T=18 iterations on the 518-d input
            target         : F[:, :512] (L2-normalized)
        """
        assert self.affinity_student is not None, "student target requested but no student loaded"
        alpha = self.student_alpha
        T = self.student_pool_iters

        device = self.device
        device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
        voxel_features_input = torch.cat([F_sem_vox, geom_vox], dim=1).to(
            device=device, dtype=torch.float32)                                      # [V, 518]
        coords_b = ME.utils.batched_coordinates([coords_vox])

        # Student embedding (Minkowski stride-1 preserves row order).
        with torch.amp.autocast(device_type=device_type, enabled=False):
            s_in = ME.SparseTensor(features=voxel_features_input, coordinates=coords_b, device=device)
            s_out = self.affinity_student(s_in)
        F_embed = F.normalize(s_out.F.float(), p=2, dim=1)                          # [V, 128]
        del s_out

        # Spatial KNN graph (excluding self).
        cnp = s_in.C[:, 1:].contiguous().float().cpu().numpy()
        del s_in
        V = cnp.shape[0]
        if V <= 1:
            return F.normalize(F_sem_vox.float(), p=2, dim=1)
        K = min(self.student_knn, max(V - 1, 1))
        index = faiss.IndexFlatL2(cnp.shape[1])
        index.add(cnp)
        _, nbr = index.search(cnp, K + 1)
        del index
        nbr = torch.from_numpy(nbr[:, 1:]).to(device)                               # [V, K]

        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Build the same row-wise cosine affinity as the dense expression
            # ``(F_embed[:, None] * F_embed[nbr]).sum(-1)``, but stream rows so
            # Matterport scenes do not materialise [V, K, 128] at once.
            w = torch.empty(V, K, device=device, dtype=torch.float32)                # [V, K]
            chunk_v = max(self.student_affinity_chunk_v, 1)
            for s in range(0, V, chunk_v):
                e = min(s + chunk_v, V)
                neigh = F_embed[nbr[s:e]]                                           # [c, K, 128]
                aff = (F_embed[s:e].unsqueeze(1) * neigh).sum(-1)                   # [c, K]
                w[s:e] = torch.softmax(aff * alpha, dim=1)
                del neigh, aff
            del F_embed

            # Sparse row-stochastic affinity for fast T-step propagation.
            row_idx = torch.arange(V, device=device).repeat_interleave(K)
            col_idx = nbr.flatten()
            A = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]),
                                        w.flatten(), size=(V, V))
            del row_idx, col_idx, w, nbr

            F_pooled = voxel_features_input                                         # [V, 518]
            del voxel_features_input
            for _ in range(T):
                F_pooled = torch.sparse.mm(A, F_pooled)
            del A

            target = F_pooled[:, :512]                                              # [V, 512]
            return F.normalize(target, p=2, dim=1)

    def _chunked_training_losses(self, x0_eff, x1, x_t, v_pred, t, weights=None):
        """Compute dense per-voxel losses without materialising extra [V,D] tensors."""
        V, D = v_pred.shape
        chunk_v = max(1, min(self.loss_chunk_v, V))
        loss_fm_sum = v_pred.new_zeros((), dtype=torch.float32)
        cos_sum = v_pred.new_zeros((), dtype=torch.float32)
        weight_sum = v_pred.new_zeros((), dtype=torch.float32)
        one_minus_t = (1.0 - t.float()).reshape(())

        for s in range(0, V, chunk_v):
            e = min(s + chunk_v, V)
            v_c = v_pred[s:e].float()
            x1_c = x1[s:e].float()
            target_v = x1_c - x0_eff[s:e].float()

            x1_hat_c = x_t[s:e].float() + one_minus_t * v_c
            cos_c = F.cosine_similarity(x1_hat_c, x1_c, dim=1, eps=1e-8)
            if weights is None:
                loss_fm_sum = loss_fm_sum + F.mse_loss(v_c, target_v, reduction="sum")
                cos_sum = cos_sum + cos_c.sum()
                weight_sum = weight_sum + float(e - s)
            else:
                w_c = weights[s:e].float().clamp_min(0.0)
                loss_fm_sum = loss_fm_sum + (
                    ((v_c - target_v) ** 2).sum(dim=1) * w_c
                ).sum()
                cos_sum = cos_sum + (cos_c * w_c).sum()
                weight_sum = weight_sum + w_c.sum()
                del w_c
            del target_v, x1_hat_c, cos_c

        denom_voxels = weight_sum.clamp_min(1.0)
        loss_fm = loss_fm_sum / (denom_voxels * D)
        cos = cos_sum / denom_voxels
        loss_manifold = 1.0 - cos
        return loss_fm, loss_manifold, cos

    def _subsample_training_voxels(self, x0, x1, F_son_vox, geom_vox, coords_vox, weights=None):
        max_voxels = self.train_max_voxels
        V = x0.shape[0]
        if max_voxels <= 0 or V <= max_voxels:
            return x0, x1, F_son_vox, geom_vox, coords_vox, weights

        idx = torch.randperm(V, device=x0.device)[:max_voxels]
        idx, _ = torch.sort(idx)
        return (
            x0.index_select(0, idx),
            x1.index_select(0, idx),
            F_son_vox.index_select(0, idx),
            geom_vox.index_select(0, idx),
            coords_vox.index_select(0, idx),
            None if weights is None else weights.index_select(0, idx),
        )

    # -------------------------------------------------------------------- #
    #  Optimisable parameters (only velocity + cond_encoder are trained)
    # -------------------------------------------------------------------- #
    def trainable_parameters(self):
        params = list(self.velocity.parameters())
        if self.cond_encoder is not None:
            params += list(self.cond_encoder.parameters())
        return params

    # -------------------------------------------------------------------- #
    #  Training (conditional flow matching)
    # -------------------------------------------------------------------- #
    def forward(self, batch_data):
        with torch.no_grad():
            (F_sem_vox, F_son_vox, geom_vox, coords_vox, inds,
             _txt, _ls) = self._voxel_inputs(batch_data)
            # Source x0 = noisy lifted semantics (already L2-normalized).
            x0 = F_sem_vox                                            # [V, 512]
            # Target x1: student-affinity pool of x0 (Path B) or Sonata pool fallback.
            if self.use_student_target and self.affinity_student is not None:
                x1 = self._student_manifold_target(F_sem_vox, geom_vox, coords_vox)
            else:
                x1 = self._teacher_manifold_target(F_sem_vox, F_son_vox, coords_vox)
            conf_vox = None
            if self.confidence_weighting:
                conf_vox = getattr(self, "last_voxel_confidence", None)
                if conf_vox is not None:
                    conf_vox = conf_vox.to(device=x0.device, dtype=torch.float32).clamp(0.0, 1.0)
                    if self.confidence_power != 1.0:
                        conf_vox = conf_vox.pow(self.confidence_power)
                    conf_vox = self.confidence_min + (1.0 - self.confidence_min) * conf_vox
            x0, x1, F_son_vox, geom_vox, coords_vox, conf_vox = self._subsample_training_voxels(
                x0, x1, F_son_vox, geom_vox, coords_vox, conf_vox)
            del F_sem_vox
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cond_in = torch.cat([F_son_vox, geom_vox], dim=1)
        self._maybe_build_cond_encoder(cond_in.shape[1])
        coords_b = ME.utils.batched_coordinates([coords_vox])
        cond_sparse = ME.SparseTensor(features=cond_in, coordinates=coords_b, device=self.device)
        cond_feats = self.cond_encoder(cond_sparse).F                # [V, cond_dim]
        del cond_sparse, cond_in, F_son_vox, geom_vox, coords_vox, inds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t = torch.rand(1, device=self.device)                        # one t per scene

        # OT interpolant (optionally with small Gaussian for OT-CFM regularity).
        if self.sigma_min > 0:
            # x0 is no longer needed separately below. Add noise in small
            # chunks so enabling sigma does not allocate two full [V, 512]
            # tensors (noise plus x0_eff) at peak memory.
            x0_eff = x0
            noise_chunk_v = max(1, min(self.loss_chunk_v, 1024))
            for s in range(0, x0_eff.shape[0], noise_chunk_v):
                e = min(s + noise_chunk_v, x0_eff.shape[0])
                x0_chunk = x0_eff[s:e]
                x0_chunk.add_(torch.randn_like(x0_chunk), alpha=self.sigma_min)
        else:
            x0_eff = x0
        x_t = torch.lerp(x0_eff, x1, t)                              # [V, 512]

        x_t_sparse = ME.SparseTensor(features=x_t, coordinates=coords_b, device=self.device)
        v_pred = self.velocity(x_t_sparse, cond_feats, t)            # [V, 512]
        del x_t_sparse

        loss_fm, loss_manifold, cos = self._chunked_training_losses(
            x0_eff, x1, x_t, v_pred, t, conf_vox)
        loss = loss_fm

        if self.manifold_loss_w > 0:
            loss = loss + self.manifold_loss_w * loss_manifold

        del v_pred, x_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out = {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_manifold": loss_manifold.detach(),
            "cos_hat_x1": cos.detach(),
        }
        if conf_vox is not None:
            out["conf_mean"] = conf_vox.detach().mean()
        return out

    # -------------------------------------------------------------------- #
    #  Inference (solve dx/dt = v)
    # -------------------------------------------------------------------- #
    @torch.no_grad()
    def _flow_purify(self, x_init, cond_feats, coords_b):
        n = max(self.ode_steps, 1)
        dt = 1.0 / n
        x = x_init.clone()
        for i in range(n):
            t0 = torch.tensor([i * dt], device=self.device)
            x_sparse = ME.SparseTensor(features=x, coordinates=coords_b, device=self.device)
            v0 = self.velocity(x_sparse, cond_feats, t0)
            del x_sparse
            if self.ode_solver == "euler":
                x = x + dt * v0
                del v0
            else:
                x_mid = x + 0.5 * dt * v0
                del v0
                t_mid = torch.tensor([(i + 0.5) * dt], device=self.device)
                xm_sparse = ME.SparseTensor(features=x_mid, coordinates=coords_b, device=self.device)
                del x_mid
                v_mid = self.velocity(xm_sparse, cond_feats, t_mid)
                del xm_sparse
                x = x + dt * v_mid
                del v_mid
            if self.renorm_each_step:
                x = F.normalize(x, dim=1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return F.normalize(x, dim=1)

    @torch.no_grad()
    def evaluate_scene(self, batch_data, vis_prefix="scene"):
        (F_sem_vox, F_son_vox, geom_vox, coords_vox, inds,
         text_features, logit_scale) = self._voxel_inputs(batch_data)

        selected_idx = None
        if self.eval_max_voxels > 0 and coords_vox.shape[0] > self.eval_max_voxels:
            V = coords_vox.shape[0]
            valid = torch.where(F_sem_vox.abs().sum(1) > 0)[0]
            if valid.numel() == 0:
                valid = torch.arange(V, device=coords_vox.device)
            if valid.numel() > self.eval_max_voxels:
                pick = torch.linspace(
                    0,
                    valid.numel() - 1,
                    steps=self.eval_max_voxels,
                    device=coords_vox.device,
                ).long()
                selected_idx = valid[pick]
            else:
                remaining = torch.ones(V, dtype=torch.bool, device=coords_vox.device)
                remaining[valid] = False
                extra = torch.where(remaining)[0][
                    : max(0, self.eval_max_voxels - valid.numel())
                ]
                selected_idx = torch.cat([valid, extra], dim=0)
            selected_idx = selected_idx.sort().values
            print(
                f"[GeoFlow] eval voxel cap: {V} -> {selected_idx.numel()} "
                f"(geoflow_eval_max_voxels={self.eval_max_voxels})"
            )
            F_sem_eval = F_sem_vox[selected_idx]
            F_son_eval = F_son_vox[selected_idx]
            geom_eval = geom_vox[selected_idx]
            coords_eval = coords_vox[selected_idx]
        else:
            F_sem_eval, F_son_eval, geom_eval, coords_eval = (
                F_sem_vox, F_son_vox, geom_vox, coords_vox
            )

        cond_in = torch.cat([F_son_eval, geom_eval], dim=1)
        del F_son_vox, geom_vox, F_son_eval, geom_eval
        torch.cuda.empty_cache()
        self._maybe_build_cond_encoder(cond_in.shape[1])
        coords_b = ME.utils.batched_coordinates([coords_eval])
        cond_sparse = ME.SparseTensor(features=cond_in, coordinates=coords_b, device=self.device)
        del cond_in
        cond_feats = self.cond_encoder(cond_sparse).F
        del cond_sparse
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        ode_start = time.perf_counter()
        x_clean = self._flow_purify(F_sem_eval, cond_feats, coords_b)   # [V_eval, 512]
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        ode_time_sec = time.perf_counter() - ode_start

        if selected_idx is None:
            scene_features = x_clean[inds]                              # [N, 512]
        else:
            selected_mask = torch.zeros(
                coords_vox.shape[0], dtype=torch.bool, device=coords_vox.device
            )
            selected_mask[selected_idx] = True
            subset_inverse = torch.full(
                (coords_vox.shape[0],), -1, dtype=torch.long, device=coords_vox.device
            )
            subset_inverse[selected_idx] = torch.arange(
                selected_idx.numel(), device=coords_vox.device
            )
            scene_features = torch.zeros(
                inds.shape[0], x_clean.shape[1], device=x_clean.device, dtype=x_clean.dtype
            )
            point_mask = selected_mask[inds]
            if point_mask.any():
                scene_features[point_mask] = x_clean[subset_inverse[inds[point_mask]]]
        return {
            "scene_features": scene_features.to(text_features.device),
            "text_features": text_features,
            "logit_scale": logit_scale,
            "ode_time_sec": ode_time_sec,
            "ode_voxels": int(F_sem_eval.shape[0]),
            "ode_steps": int(self.ode_steps),
            "ode_solver": str(self.ode_solver),
        }
