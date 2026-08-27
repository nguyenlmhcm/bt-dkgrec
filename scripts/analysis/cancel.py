"""Does behaviour-time weighting change WHICH items dominate a user's profile,
or only the overall SCALE of that user's row?

Per user, correlate the bt_dkgrec row of A_hat with the static row.
  corr ~ 1.0  -> same mix, only the scale changed
  corr <  1.0 -> the mix of neighbours really changed

WARNING, added 2026-08-27. This file used to conclude that a row which is only
rescaled is "INERT by construction" for ranking. That conclusion is wrong and
was falsified by measurement (bt_dkgrec and static_kg_gcn differ on warm_deg1).
Three reasons a pure row rescale is not inert:

  1. Symmetric normalisation does not cancel W. The denominator is sqrt(d_u),
     not d_u, so a degree-1 user keeps a coefficient of sqrt(W)/sqrt(d_i).
  2. The final embedding is the mean over layers and includes h^0, which is
     never multiplied by A_hat. Scaling A_hat therefore does not scale z_u; it
     rotates it.
  3. A_hat is symmetric, so rescaling user u's row also rescales column u,
     which moves item degrees and item embeddings for everyone.

What this script still measures correctly is amplitude: how often the weighting
changes the mix of a user's neighbours rather than only the scale.
"""
import sys; sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
import numpy as np, scipy.sparse as sp
from pathlib import Path
from src.graph.normalize import symmetric_normalize

A = {}
for m in ("bt_dkgrec","static_kg_gcn"):
    A[m] = symmetric_normalize(
        sp.load_npz(f"/root/ThS_HUIT/bt-dkgrec/data/processed/original/{m}/adjacency.npz")
    ).tocsr()
d, s = A["bt_dkgrec"], A["static_kg_gcn"]

rng = np.random.default_rng(2020)
n_users = 1_027_985
rows = rng.choice(n_users, 4000, replace=False)

corrs, scales = [], []
for r in rows:
    dv = d.getrow(r).toarray().ravel(); sv = s.getrow(r).toarray().ravel()
    nz = sv != 0
    if nz.sum() < 3: continue
    x, y = dv[nz], sv[nz]
    if x.std() == 0 or y.std() == 0: continue
    corrs.append(np.corrcoef(x, y)[0,1])
    scales.append(x.sum()/y.sum())

corrs = np.array(corrs); scales = np.array(scales)
print(f"{len(corrs):,} user co >= 3 canh")
print(f"\ntuong quan hang (mix co doi khong):")
for q in (5, 25, 50, 75, 95):
    print(f"   p{q:<3} {np.percentile(corrs, q):.4f}")
print(f"   trung binh {corrs.mean():.4f}")
print(f"\nty le phan tram user co corr > 0,99 (chi doi SCALE, VO HIEU cho xep hang): "
      f"{(corrs > 0.99).mean():.1%}")
print(f"ty le user co corr < 0,95 (mix doi that): {(corrs < 0.95).mean():.1%}")
print(f"\nhe so ty le hang (scale): trung vi {np.median(scales):.4f}, "
      f"p5 {np.percentile(scales,5):.4f}, p95 {np.percentile(scales,95):.4f}")
