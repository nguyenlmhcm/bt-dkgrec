"""How much of the seed-to-seed spread is COMMON to every model?

If seed is a shared blocking factor, an unpaired test throws that away and
looks far weaker than the data actually is.
"""
import sys; sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from src.evaluation.reporting import load_runs

RUNS = Path("/root/ThS_HUIT/bt-dkgrec/experiments/runs")
GRAPH = ["lightgcn","static_kg_gcn","bt_dkgrec"]

for split in ("test","valid"):
    for metric in ("recall@20","ndcg@20"):
        f = load_runs(RUNS, split=split, segment="warm")
        f = f[(f["cohort"]=="original") & f["model"].isin(GRAPH)]
        wide = f.pivot(index="seed", columns="model", values=metric)[GRAPH]

        # variance decomposition
        grand = wide.values.mean()
        seed_effect = wide.mean(axis=1) - grand          # per-seed shift
        model_effect = wide.mean(axis=0) - grand
        resid = wide.values - grand - seed_effect.values[:,None] - model_effect.values[None,:]
        ss_seed  = len(GRAPH) * (seed_effect**2).sum()
        ss_model = len(wide)  * (model_effect**2).sum()
        ss_res   = (resid**2).sum()
        tot = ss_seed + ss_model + ss_res

        print(f"\n=== original / {split} / {metric} ===")
        print(f"  phuong sai do SEED  : {ss_seed/tot:6.1%}   <- chung cho moi mo hinh, tru duoc")
        print(f"  phuong sai do MO HINH: {ss_model/tot:6.1%}")
        print(f"  phan du (that su ngau nhien): {ss_res/tot:6.1%}")

        for a,b in (("bt_dkgrec","static_kg_gcn"), ("bt_dkgrec","lightgcn"),
                    ("static_kg_gcn","lightgcn")):
            d = wide[a] - wide[b]
            t_pair = d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else np.inf
            p_pair = 2*(1-stats.t.cdf(abs(t_pair), len(d)-1))
            _, p_welch = stats.ttest_ind(wide[a], wide[b], equal_var=False)
            print(f"    {a:>14} vs {b:<14} chenh {d.mean():+.6f}  "
                  f"Welch p={p_welch:.4f}   GHEP CAP p={p_pair:.4f}")
