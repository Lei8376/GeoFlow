"""GeoDiff: Geometry-Conditioned Diffusion for 3D Feature Purification.

New idea (extends GeoPurify): instead of denoising the noisy 2D->3D projected
semantic features with a fixed affinity-pooling step, we recast purification as a
*conditional generation* problem and learn a geometry-conditioned diffusion model
(GeoDiff) that reconstructs clean, geometry-consistent semantic features.

    input  : noisy 2D->3D semantic feature  x  (per voxel, 512-d)
             + geometric condition  c  = Encoder(Sonata feat (+) RGB (+) normal)
    output : clean semantic feature  x0  lying on the geometry-consistent manifold
    head   : dot(x0, text_embeddings) -> open-vocabulary labels

Why diffusion (vs. the linear affinity pooling in GeoPurify)?
  * The denoiser is a *nonlinear, iterative, learned* operator conditioned on
    geometry, so it can model the density of the clean-feature manifold rather
    than only averaging neighbours.
  * Training is still label-free: the "clean" target is defined purely by the
    frozen geometry teacher (Sonata) via a teacher-affinity geometry pooling of
    the noisy semantics -- no 3D semantic annotations are used.

Label-free clean target (the manifold sample x0):
    x0 = GeometryPool_Sonata(x_noisy)   # Sonata-affinity KNN soft-pooling
This is the same geometric-coherence prior GeoPurify relies on, but used here as
a *generation target* instead of an inference-time operator.

Inference is SDEdit-style: we add a controlled amount of noise to the (already
informative) noisy lifted feature at t_start and run DDIM reverse conditioned on
geometry. Starting from the lifted feature (not pure Gaussian noise) is essential
-- geometry alone cannot decide chair-vs-table semantics, so we must retain the
semantic content while letting geometry purify its structure.

All heavy frozen backbones (X-Decoder, Sonata) and the feature-lifting code are
reused from SonataXAffinityTrainer via subclassing.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch_scatter
import faiss

from models.affinity_module import SonataXAffinityTrainer, MinkowskiResBlock


# --------------------------------------------------------------------------- #
#  Diffusion schedule utilities
# --------------------------------------------------------------------------- #
def cosine_beta_schedule(T, s=0.008):
    """Nichol & Dhariwal cosine schedule."""
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    acp = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    acp = acp / acp[0]
    betas = 1 - acp[1:] / acp[:-1]
    return betas.clamp(1e-5, 0.999)


#将时间步 t编码成 dim维的正弦/余弦时间向量
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] (long/float) -> [B, dim]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# --------------------------------------------------------------------------- #
#  Sparse network building blocks
# --------------------------------------------------------------------------- #
def _rebuild(x_ref, new_feats):
    """Wrap new per-voxel features into a SparseTensor sharing x_ref's coords."""
    return ME.SparseTensor(
        features=new_feats,
        coordinate_map_key=x_ref.coordinate_map_key,
        coordinate_manager=x_ref.coordinate_manager,
    )


class GeoCondEncoder(nn.Module):
    """Sparse 3D CNN that encodes geometry [Sonata (+) RGB (+) normal] -> cond."""

    def __init__(self, in_dim, cond_dim=256, hidden=256, n_blocks=2):
        super().__init__()
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_dim, hidden, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(hidden),
            ME.MinkowskiReLU(),
        )
        self.blocks = nn.Sequential(*[MinkowskiResBlock(hidden) for _ in range(n_blocks)])
        self.head = ME.MinkowskiConvolution(hidden, cond_dim, kernel_size=1, dimension=3)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class GeoDiffDenoiser(nn.Module):
    """epsilon-prediction sparse denoiser, FiLM time-conditioning + concat cond.

    Forward signature:
        forward(x_t_sparse, cond_feats, t_index) -> predicted noise [V, feat_dim]
    where x_t_sparse is a SparseTensor [V, feat_dim] (carries coords), cond_feats
    is the aligned [V, cond_dim] geometry condition, and t_index is a 1-D long
    tensor with the (single, per-scene) diffusion step.
    """

    def __init__(self, feat_dim=512, cond_dim=256, hidden=512, time_dim=256, n_blocks=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Sequential(
            ME.MinkowskiConvolution(feat_dim + cond_dim, hidden, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(hidden),
            ME.MinkowskiReLU(),
        )
        self.blocks = nn.ModuleList([MinkowskiResBlock(hidden) for _ in range(n_blocks)])
        # FiLM: per-step (gamma, beta) broadcast across all voxels.
        self.films = nn.ModuleList([nn.Linear(time_dim, hidden * 2) for _ in range(n_blocks)])
        self.out = ME.MinkowskiConvolution(hidden, feat_dim, kernel_size=1, dimension=3)

    def forward(self, x_t_sparse, cond_feats, t_index):
        temb = self.time_mlp(t_index)            # [1, time_dim]
        h_in = torch.cat([x_t_sparse.F, cond_feats], dim=1)
        h = self.input(_rebuild(x_t_sparse, h_in))
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            gamma, beta = film(temb).chunk(2, dim=-1)   # [1, hidden] each
            h = _rebuild(h, h.F * (1 + gamma) + beta)
        return self.out(h).F                      # [V, feat_dim]


# --------------------------------------------------------------------------- #
#  Diffusion wrapper (schedule buffers + q_sample + DDIM)
# --------------------------------------------------------------------------- #
class GeoDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.T = num_timesteps
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    def q_sample(self, x0, t, noise):
        """x_t = sqrt(acp_t) x0 + sqrt(1-acp_t) noise."""
        a = self.sqrt_acp[t].view(-1, 1)
        b = self.sqrt_one_minus_acp[t].view(-1, 1)
        return a * x0 + b * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        a = self.sqrt_acp[t].view(-1, 1)
        b = self.sqrt_one_minus_acp[t].view(-1, 1)
        return (x_t - b * eps) / a.clamp(min=1e-8)


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class SonataXGeoDiffTrainer(SonataXAffinityTrainer):
    """Reuses X-Decoder + Sonata + feature lifting from the parent; replaces the
    affinity-student purification with a geometry-conditioned diffusion model."""

    def __init__(self, cfg, xdecoder_cfg, scene_config, device="cuda", use_lseg=False):
        super().__init__(cfg, xdecoder_cfg, scene_config, device, use_lseg)
        # The old affinity student is unused here.
        if hasattr(self, "affinity_student"):
            del self.affinity_student

        self.feat_dim = 512
        self.cond_dim = int(getattr(cfg, "geodiff_cond_dim", 256))
        self.hidden = int(getattr(cfg, "geodiff_hidden", 512))
        self.n_blocks = int(getattr(cfg, "geodiff_blocks", 4))
        self.num_timesteps = int(getattr(cfg, "geodiff_timesteps", 1000))

        # Clean-target (teacher manifold) pooling hyper-params.
        self.target_knn = int(getattr(cfg, "geodiff_target_knn", 96))
        self.target_alpha = float(getattr(cfg, "geodiff_target_alpha", 20.0))
        self.target_iters = int(getattr(cfg, "geodiff_target_iters", 6))
        self.target_chunk_v = int(getattr(cfg, "geodiff_target_chunk_v", 8192))

        # Inference (SDEdit + DDIM) hyper-params.
        self.ddim_steps = int(getattr(cfg, "geodiff_ddim_steps", 25))
        self.sdedit_strength = float(getattr(cfg, "geodiff_sdedit_strength", 0.4))
        self.manifold_loss_w = float(getattr(cfg, "geodiff_manifold_loss_w", 0.1))

        self.diffusion = GeoDiffusion(self.num_timesteps)
        self.denoiser = GeoDiffDenoiser(
            feat_dim=self.feat_dim, cond_dim=self.cond_dim,
            hidden=self.hidden, n_blocks=self.n_blocks,
        ).to(device)
        # Build the geometry-condition encoder eagerly so that ALL trainable
        # parameters exist before DDP wraps the model (lazy build would leave the
        # encoder out of the DDP reducer). We probe Sonata's output dim once; if
        # that fails (e.g. CPU-only smoke test), fall back to a lazy build, which
        # is fine for single-GPU runs.
        self.cond_encoder = None
        self._cond_in_dim = None
        try:
            ds = self._probe_sonata_dim()
            self._maybe_build_cond_encoder(ds + 6)
        except Exception as e:  # pragma: no cover
            print(f"[GeoDiff] Sonata dim probe failed ({e}); cond encoder will "
                  f"build lazily (single-GPU only).")
        print("GeoDiff modules created (denoiser + diffusion + cond encoder).")

    @torch.no_grad()
    def _probe_sonata_dim(self, n=2048):
        import sonata
        coords = (np.random.rand(n, 3).astype(np.float32)) * 2.0
        point = {
            "coord": coords,
            "color": np.random.rand(n, 3).astype(np.float32),
            "normal": np.random.randn(n, 3).astype(np.float32),
        }
        point = sonata.transform.default()(point)
        for k in list(point.keys()):
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].cuda()
        point = self.sonata_teacher(point)
        for _ in range(2):
            if "pooling_parent" not in point.keys():
                break
            parent = point.pop("pooling_parent")
            inv = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inv = point.pop("pooling_inverse")
            parent.feat = point.feat[inv]
            point = parent
        return int(point.feat.shape[1])

    # ---- helpers ---------------------------------------------------------- #
    def _maybe_build_cond_encoder(self, in_dim):
        if self.cond_encoder is None:
            self._cond_in_dim = in_dim
            self.cond_encoder = GeoCondEncoder(
                in_dim=in_dim, cond_dim=self.cond_dim
            ).to(self.device)
            print(f"GeoCondEncoder built with in_dim={in_dim}")

    def trainable_parameters(self):
        params = list(self.denoiser.parameters())
        if self.cond_encoder is not None:
            params += list(self.cond_encoder.parameters())
        return params

    def _voxel_inputs(self, batch_data):
        """Return per-voxel (semantic[V,512], sonata[V,Ds], geom[V,6], coords[V,3],
        scene_inds_reconstruct) using the parent's frozen backbones."""
        F_sem_pts, text_features, logit_scale = self.lift_xdecoder_features(batch_data)
        F_sonata_pts = self.get_sonata_features(batch_data)
        (
            scene_coords, scene_coords_3d, scene_inds_reconstruct, *_rest,
            scene_gauss_features
        ) = batch_data
        dev = self.device
        inds = scene_inds_reconstruct.to(dev)
        V = scene_coords_3d.shape[0]

        F_sem_vox = torch_scatter.scatter_mean(F_sem_pts.to(dev), inds, dim=0)        # [V,512]
        F_son_vox = torch_scatter.scatter_mean(F_sonata_pts.to(dev), inds, dim=0)     # [V,Ds]
        geom_vox = torch_scatter.scatter_mean(
            scene_gauss_features[:, :6].float().to(dev), inds, dim=0)                 # [V,6]
        conf_pts = getattr(self, "last_lift_confidence", None)
        self.last_voxel_confidence = None
        if conf_pts is not None and conf_pts.shape[0] == inds.shape[0]:
            conf_src = conf_pts.float().to(dev).view(-1, 1)
            self.last_voxel_confidence = torch_scatter.scatter_mean(
                conf_src, inds, dim=0, dim_size=V).view(-1)
        coords_vox = scene_coords_3d.to(dev)
        return (F.normalize(F_sem_vox, dim=1), F_son_vox, geom_vox, coords_vox,
                inds, text_features, logit_scale)

    @torch.no_grad()
    def _teacher_manifold_target(self, F_sem_vox, F_son_vox, coords_vox):
        """Label-free clean target: Sonata-affinity geometry pooling of semantics."""
        V = coords_vox.shape[0]
        if V <= 1:
            return F.normalize(F_sem_vox.float(), dim=1)
        device = coords_vox.device
        device_type = device.type
        K = min(self.target_knn, max(V - 1, 1))
        cnp = coords_vox.float().cpu().numpy()
        index = faiss.IndexFlatL2(cnp.shape[1])
        index.add(cnp)
        _, nbr = index.search(cnp, K + 1)
        nbr = torch.from_numpy(nbr[:, 1:]).to(device)                    # [V,K]

        with torch.amp.autocast(device_type=device_type, enabled=False):
            son = F.normalize(F_son_vox.to(device=device, dtype=torch.float32), dim=1)
            w = torch.empty(V, K, device=device, dtype=torch.float32)
            chunk_v = max(1, self.target_chunk_v)
            for s in range(0, V, chunk_v):
                e = min(s + chunk_v, V)
                neigh = son[nbr[s:e]]                                    # [c,K,Ds]
                aff = (son[s:e].unsqueeze(1) * neigh).sum(-1)            # [c,K]
                w[s:e] = torch.softmax(aff * self.target_alpha, dim=1)
                del neigh, aff
            del son

            x = F_sem_vox.to(device=device, dtype=torch.float32)
            for _ in range(self.target_iters):
                x_next = torch.empty_like(x)
                for s in range(0, V, chunk_v):
                    e = min(s + chunk_v, V)
                    neigh_x = x[nbr[s:e]]                                # [c,K,512]
                    x_next[s:e] = (w[s:e].unsqueeze(-1) * neigh_x).sum(1)
                    del neigh_x
                x = x_next                                               # weighted neighbour mean
            del w, nbr
            return F.normalize(x, dim=1)

    # ---- training --------------------------------------------------------- #
    def forward(self, batch_data):
        with torch.no_grad():
            (F_sem_vox, F_son_vox, geom_vox, coords_vox, inds,
             _txt, _ls) = self._voxel_inputs(batch_data)
            x0 = self._teacher_manifold_target(F_sem_vox, F_son_vox, coords_vox)  # [V,512]

        cond_in = torch.cat([F_son_vox, geom_vox], dim=1)
        self._maybe_build_cond_encoder(cond_in.shape[1])
        coords_b = ME.utils.batched_coordinates([coords_vox])
        cond_sparse = ME.SparseTensor(features=cond_in, coordinates=coords_b, device=self.device)
        cond_feats = self.cond_encoder(cond_sparse).F                    # [V,cond_dim]

        V = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device)
        t_full = t.expand(V)
        noise = torch.randn_like(x0)
        x_t = self.diffusion.q_sample(x0, t_full, noise)

        x_t_sparse = ME.SparseTensor(features=x_t, coordinates=coords_b, device=self.device)
        eps_pred = self.denoiser(x_t_sparse, cond_feats, t)              # [V,512]

        loss_eps = F.mse_loss(eps_pred, noise)
        loss = loss_eps
        if self.manifold_loss_w > 0:
            x0_hat = self.diffusion.predict_x0_from_eps(x_t, t_full, eps_pred)
            x0_hat = F.normalize(x0_hat, dim=1)
            loss_manifold = (1.0 - (x0_hat * x0).sum(1)).mean()         # 1 - cosine
            loss = loss + self.manifold_loss_w * loss_manifold

        del cond_sparse, x_t_sparse, eps_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    # ---- inference -------------------------------------------------------- #
    @torch.no_grad()
    def _ddim_purify(self, x_init, cond_feats, coords_b):
        """SDEdit init at t_start, then DDIM reverse -> clean x0."""
        diff = self.diffusion
        T = self.num_timesteps
        t_start = int(self.sdedit_strength * (T - 1))
        t_start = max(t_start, 1)

        V = x_init.shape[0]
        noise = torch.randn_like(x_init)
        x_t = diff.q_sample(x_init, torch.full((V,), t_start, device=self.device, dtype=torch.long), noise)

        # DDIM timestep grid from t_start down to 0.
        steps = torch.linspace(t_start, 0, self.ddim_steps + 1, device=self.device).long()
        for i in range(self.ddim_steps):
            t_cur, t_next = int(steps[i].item()), int(steps[i + 1].item())
            t_idx = torch.tensor([t_cur], device=self.device, dtype=torch.long)
            x_sparse = ME.SparseTensor(features=x_t, coordinates=coords_b, device=self.device)
            eps = self.denoiser(x_sparse, cond_feats, t_idx)
            x0_hat = diff.predict_x0_from_eps(
                x_t, torch.full((V,), t_cur, device=self.device, dtype=torch.long), eps)
            x0_hat = F.normalize(x0_hat, dim=1)
            if t_next == 0:
                x_t = x0_hat                          # final DDIM step: emit x0
                break
            a_next = diff.sqrt_acp[t_next]
            b_next = diff.sqrt_one_minus_acp[t_next]
            x_t = a_next * x0_hat + b_next * eps      # deterministic DDIM (eta=0)
        return F.normalize(x_t, dim=1)

    @torch.no_grad()
    def evaluate_scene(self, batch_data, vis_prefix="scene"):
        (F_sem_vox, F_son_vox, geom_vox, coords_vox, inds,
         text_features, logit_scale) = self._voxel_inputs(batch_data)

        cond_in = torch.cat([F_son_vox, geom_vox], dim=1)
        self._maybe_build_cond_encoder(cond_in.shape[1])
        coords_b = ME.utils.batched_coordinates([coords_vox])
        cond_sparse = ME.SparseTensor(features=cond_in, coordinates=coords_b, device=self.device)
        cond_feats = self.cond_encoder(cond_sparse).F

        x0_clean = self._ddim_purify(F_sem_vox, cond_feats, coords_b)     # [V,512]
        # voxel -> point
        scene_features = x0_clean[inds]                                   # [N,512]
        return {
            "scene_features": scene_features.to(text_features.device),
            "text_features": text_features,
            "logit_scale": logit_scale,
        }
