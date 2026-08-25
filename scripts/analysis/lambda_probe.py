"""Does lambda have a strong optimum? Answer on the CPU before spending GPU hours."""
import json, sys
sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps

ALPHA = {"view":1.0, "addtocart":2.0, "transaction":3.0}
DAY = 86_400_000
LAMBDAS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

def auc(score, label):
    r = sps.rankdata(score); n1 = label.sum(); n0 = len(label)-n1
    return (r[label].sum() - n1*(n1+1)/2)/(n1*n0) if n1 and n0 else float("nan")

for cohort in ("original","active"):
    interim = Path(f"/root/ThS_HUIT/bt-dkgrec/data/interim/{cohort}")
    ev = pd.read_parquet(interim/"events.parquet")
    t_train = int(json.loads((interim/"split.json").read_text())["t_train"])
    tr = ev[ev["split"]=="train"].copy()
    tr["a"] = tr["behavior"].astype(str).map(ALPHA).astype("float64")
    tr["dt"] = np.maximum(0.0, (t_train - tr["timestamp"])/DAY)

    tgt = ev[(ev["split"]=="valid") & ev["behavior"].astype(str).isin(["addtocart","transaction"])]
    keys = set(zip(tgt["visitorid"], tgt["itemid"]))

    print(f"\n=== {cohort} ===")
    print(f"{'lambda':>8}{'w = alpha*decay':>18}{'chi decay (alpha=1)':>22}")
    for lam in LAMBDAS:
        tr["d"] = np.exp(-lam*tr["dt"])
        tr["w"] = tr["a"]*tr["d"]
        agg = tr.groupby(["visitorid","itemid"]).agg(w=("w","sum"), d=("d","sum")).reset_index()
        lab = np.array([(u,i) in keys for u,i in zip(agg["visitorid"], agg["itemid"])])
        print(f"{lam:>8.3f}{auc(agg['w'].to_numpy(), lab):>18.4f}{auc(agg['d'].to_numpy(), lab):>22.4f}")
