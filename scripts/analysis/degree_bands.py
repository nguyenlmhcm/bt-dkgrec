"""So sanh dong/tinh tren tung phan doan bac (D34).

Tien doan phat bieu TRUOC khi do:
  warm_deg1     -> hieu so DUNG BANG 0 (trong so bi chuan hoa triet tieu)
  warm_deg3plus -> neu co che co that thi phai lon hon muc gop
"""
import sys; sys.path.insert(0, '/root/ThS_HUIT/bt-dkgrec')
from pathlib import Path
import numpy as np
from scipy import stats
from src.evaluation.reporting import load_runs

RUNS = Path("/root/ThS_HUIT/bt-dkgrec/experiments/runs")
BANDS = ("warm", "warm_deg1", "warm_deg2", "warm_deg3plus")
PAIRS = [("bt_dkgrec", "static_kg_gcn"),
         ("bt_dkgrec_l05", "static_kg_gcn"),
         ("bt_dkgrec", "lightgcn")]

for metric in ("recall@20", "ndcg@20"):
    print(f"\n{'='*74}\n  active / test / {metric}\n{'='*74}")
    for band in BANDS:
        f = load_runs(RUNS, split="test", segment=band)
        f = f[f.cohort == "active"]
        if f.empty or metric not in f.columns:
            print(f"\n  [{band}] khong do duoc")
            continue
        n_users = int(f["n_users"].iloc[0])
        print(f"\n  [{band}]  n_user = {n_users}")
        p = f.pivot(index="seed", columns="model", values=metric)
        for a, b in PAIRS:
            if a not in p.columns or b not in p.columns:
                continue
            d = (p[a] - p[b]).dropna()
            if len(d) < 2:
                continue
            sd = d.std(ddof=1)
            if sd > 0:
                t = d.mean() / (sd / np.sqrt(len(d)))
                p_pair = 2 * (1 - stats.t.cdf(abs(t), len(d) - 1))
            else:
                p_pair = float("nan")
            _, p_w = stats.ttest_ind(p[a].dropna(), p[b].dropna(), equal_var=False)
            rel = d.mean() / p[b].mean() if p[b].mean() else float("nan")
            print(f"     {a:>14} vs {b:<14} {d.mean():+.6f} ({rel:+7.2%})"
                  f"  thang {int((d>0).sum())}/{len(d)}"
                  f"  Welch p={p_w:.4f}  ghep cap p={p_pair:.4f}")
