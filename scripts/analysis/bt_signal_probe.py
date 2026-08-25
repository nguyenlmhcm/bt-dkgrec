"""Does the behavior-time weight carry information about FUTURE targets?

For every train edge (u,i) we ask: did u perform a target behaviour on i during
the validation window? Then we test how well each candidate edge score separates
the edges that did from the edges that did not:

    w(u,i)          the behaviour-time weight  -> what bt_dkgrec propagates
    count(u,i)      number of interactions     -> what static_kg_gcn effectively has
    recency only    exp(-lambda dt) of last touch
    behaviour only  max alpha_b, no decay

AUC = probability that a randomly chosen "led to a target" edge scores above a
randomly chosen one that did not. 0.5 means the score is pure noise.
"""
import json, sys
sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps

ALPHA = {"view":1.0, "addtocart":2.0, "transaction":3.0}
DAY, LAM = 86_400_000, 0.01

def auc(score, label):
    r = sps.rankdata(score)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    return (r[label].sum() - n1*(n1+1)/2) / (n1*n0)

for cohort in ("original","active"):
    interim = Path(f"/root/ThS_HUIT/bt-dkgrec/data/interim/{cohort}")
    ev = pd.read_parquet(interim/"events.parquet")
    t_train = int(json.loads((interim/"split.json").read_text())["t_train"])

    tr = ev[ev["split"]=="train"]
    g = tr.groupby(["visitorid","itemid"])
    feat = g.agg(count=("behavior","size"), last_ts=("timestamp","max")).reset_index()
    tr2 = tr.copy()
    tr2["a"] = tr2["behavior"].astype(str).map(ALPHA).astype("float64")
    tr2["dt"] = np.maximum(0.0, (t_train - tr2["timestamp"]) / DAY)
    tr2["w"] = tr2["a"] * np.exp(-LAM*tr2["dt"])
    agg = tr2.groupby(["visitorid","itemid"]).agg(w=("w","sum"), amax=("a","max")).reset_index()
    feat = feat.merge(agg, on=["visitorid","itemid"])
    feat["recency"] = np.exp(-LAM * np.maximum(0.0,(t_train-feat["last_ts"])/DAY))

    tgt = ev[(ev["split"]=="valid") & ev["behavior"].isin(["addtocart","transaction"])]
    keys = set(zip(tgt["visitorid"], tgt["itemid"]))
    feat["hit"] = [ (u,i) in keys for u,i in zip(feat["visitorid"], feat["itemid"]) ]
    lab = feat["hit"].to_numpy()

    print(f"\n=== {cohort}: {len(feat):,} canh train, {lab.sum():,} canh dan toi target o valid "
          f"({lab.mean():.3%}) ===")
    for name, col in (("w  (behavior-time, day du)","w"),
                      ("count (so luot tuong tac)","count"),
                      ("recency (chi suy giam t.gian)","recency"),
                      ("behaviour (chi alpha, k suy giam)","amax")):
        print(f"   AUC {name:<34} {auc(feat[col].to_numpy(), lab):.4f}")
