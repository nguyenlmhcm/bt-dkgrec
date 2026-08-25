"""bt_dkgrec_l05 (lambda=0.05) doi chieu voi cac mo hinh con lai.

Bao cao ca Welch (khong ghep cap, dung CLAUDE.md) va ghep cap theo seed
(seed la yeu to khoi chung) de quyet dinh chua chot van con mo.
"""
import sys; sys.path.insert(0, '/root/ThS_HUIT/bt-dkgrec')
from pathlib import Path
import numpy as np
from scipy import stats
from src.evaluation.reporting import load_runs

RUNS = Path("/root/ThS_HUIT/bt-dkgrec/experiments/runs")
GRAPH = ["lightgcn", "static_kg_gcn", "bt_dkgrec", "bt_dkgrec_l05"]
PAIRS = [("bt_dkgrec_l05", "static_kg_gcn"),
         ("bt_dkgrec_l05", "bt_dkgrec"),
         ("bt_dkgrec_l05", "lightgcn"),
         ("bt_dkgrec",     "static_kg_gcn")]

for cohort in ("original", "active"):
    for split in ("test", "valid"):
        for metric in ("recall@20", "ndcg@20"):
            f = load_runs(RUNS, split=split, segment="warm")
            f = f[(f["cohort"] == cohort) & f["model"].isin(GRAPH)]
            if f.empty:
                continue
            wide = f.pivot(index="seed", columns="model", values=metric)
            have = [m for m in GRAPH if m in wide.columns]
            wide = wide[have].dropna()
            print(f"\n=== {cohort} / {split} / {metric} ===")
            for m in have:
                print(f"  {m:>16}: " + "  ".join(f"{v:.6f}" for v in wide[m])
                      + f"   TB {wide[m].mean():.6f} +/- {wide[m].std(ddof=1):.6f}")
            for a, b in PAIRS:
                if a not in wide.columns or b not in wide.columns:
                    continue
                d = wide[a] - wide[b]
                n = len(d)
                sd = d.std(ddof=1)
                if sd > 0:
                    t = d.mean() / (sd / np.sqrt(n))
                    p_pair = 2 * (1 - stats.t.cdf(abs(t), n - 1))
                else:
                    p_pair = float("nan")
                _, p_welch = stats.ttest_ind(wide[a], wide[b], equal_var=False)
                win = sum(d > 0)
                print(f"    {a:>14} vs {b:<14} chenh {d.mean():+.6f} "
                      f"({d.mean()/wide[b].mean():+6.2%})  thang {win}/{n}  "
                      f"Welch p={p_welch:.4f}   ghep cap p={p_pair:.4f}")
