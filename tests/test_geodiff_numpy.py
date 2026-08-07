"""Pure-NumPy validation of GeoDiff's core math (no torch / ME / GPU needed).

Checks the identities the implementation relies on:

  G1  Cosine schedule: alpha_bar in (0,1], alpha_bar[0]~1, monotone decreasing.
  G2  q_sample variance identity: E[x_t^2] ~ ab*E[x0^2] + (1-ab).
  G3  x0 reconstruction from (x_t, true eps) is exact.
  G4  SDEdit round-trip: noise x0 to t_start, then DDIM reverse with an ORACLE
      denoiser recovers x0 (i.e. a perfect denoiser inverts the forward process).
  G5  Label-free clean target: Sonata-affinity geometry pooling of the noisy
      semantics is closer to the true (segment) manifold than the noisy input.

Run:  python tests/test_geodiff_numpy.py
"""
import math
import numpy as np

rng = np.random.default_rng(0)


def cosine_alpha_bar(T, s=0.008):
    steps = T + 1
    t = np.linspace(0, T, steps) / T
    ab = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 1e-5, 0.999)
    alphas = 1 - betas
    return np.cumprod(alphas)            # alpha_bar over t=1..T


def l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def run():
    res = []
    T = 1000
    ab = cosine_alpha_bar(T)
    sqrt_ab = np.sqrt(ab)
    sqrt_om = np.sqrt(1 - ab)

    # G1
    ok1 = (ab[0] > 0.99 and ab[-1] < 0.05 and np.all(np.diff(ab) <= 1e-9)
           and ab.min() > 0 and ab.max() <= 1.0 + 1e-6)
    res.append(("G1 cosine alpha_bar valid & monotone", ok1,
                f"ab[0]={ab[0]:.3f}, ab[-1]={ab[-1]:.4f}"))

    # G2 variance identity
    D = 64
    x0 = rng.standard_normal((20000, D))
    t = 400
    noise = rng.standard_normal((20000, D))
    x_t = sqrt_ab[t] * x0 + sqrt_om[t] * noise
    lhs = (x_t ** 2).mean()
    rhs = ab[t] * (x0 ** 2).mean() + (1 - ab[t])
    ok2 = abs(lhs - rhs) < 0.05
    res.append(("G2 q_sample variance identity", ok2, f"E[x_t^2]={lhs:.3f} vs {rhs:.3f}"))

    # G3 exact x0 from true eps
    x0_hat = (x_t - sqrt_om[t] * noise) / sqrt_ab[t]
    ok3 = np.abs(x0_hat - x0).max() < 1e-6
    res.append(("G3 predict_x0_from_eps exact (true eps)", ok3,
                f"max|Δ|={np.abs(x0_hat - x0).max():.2e}"))

    # G4 SDEdit + DDIM round trip with oracle denoiser
    V = 500
    x0v = rng.standard_normal((V, D))
    t_start = int(0.4 * (T - 1))
    x_t = sqrt_ab[t_start] * x0v + sqrt_om[t_start] * rng.standard_normal((V, D))
    grid = np.linspace(t_start, 0, 26).astype(int)
    for i in range(len(grid) - 1):
        tc, tn = grid[i], grid[i + 1]
        eps = (x_t - sqrt_ab[tc] * x0v) / max(sqrt_om[tc], 1e-8)   # ORACLE eps
        x0_pred = (x_t - sqrt_om[tc] * eps) / sqrt_ab[tc]
        if tn == 0:
            x_t = x0_pred
            break
        x_t = sqrt_ab[tn] * x0_pred + sqrt_om[tn] * eps            # DDIM eta=0
    ok4 = np.abs(x_t - x0v).max() < 1e-3
    res.append(("G4 SDEdit+DDIM round trip recovers x0 (oracle)", ok4,
                f"max|Δ|={np.abs(x_t - x0v).max():.2e}"))

    # G5 teacher-affinity pooling improves toward the manifold
    S, per = 8, 100
    Np = S * per
    seg = np.repeat(np.arange(S), per)
    seg_mean = rng.standard_normal((S, 16))            # true clean semantic per segment
    clean = seg_mean[seg]                              # [Np,16]
    noisy = clean + 0.8 * rng.standard_normal((Np, 16))
    # geometry: coords cluster by segment; sonata feat ~ segment one-hot
    coords = seg[:, None] * 5.0 + 0.3 * rng.standard_normal((Np, 3))
    sonata = np.eye(S)[seg] + 0.05 * rng.standard_normal((Np, S))
    # spatial KNN
    K = 24
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :K]
    son = l2(sonata)
    aff = (son[:, None, :] * son[nbr]).sum(2)
    w = softmax(aff * 20.0, axis=1)
    pooled = l2(noisy.copy())
    base = l2(noisy)
    for _ in range(6):
        pooled = (w[..., None] * pooled[nbr]).sum(1)
        pooled = l2(pooled)
    clean_n = l2(clean)
    err_noisy = ((base - clean_n) ** 2).mean()
    err_pooled = ((pooled - clean_n) ** 2).mean()
    ok5 = err_pooled < err_noisy
    res.append(("G5 teacher-affinity target closer to clean manifold", ok5,
                f"MSE noisy={err_noisy:.3f} -> pooled={err_pooled:.3f}"))

    print("\n" + "=" * 76)
    print("GeoDiff core-math CPU validation (NumPy)")
    print("=" * 76)
    all_ok = True
    for name, ok, detail in res:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
    print("=" * 76)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
