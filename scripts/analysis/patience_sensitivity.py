"""Do nhay cua ket luan theo ngan sach dung som (khong can train lai).

curves.csv luu validation moi lan eval, nen co the phat lai luat dung som
voi bat ky patience nao NHO HON HOAC BANG muc da chay that.
"""
import sys; sys.path.insert(0, '/root/ThS_HUIT/bt-dkgrec')
import glob, os
import pandas as pd

EVAL_EVERY = 5
COL = "valid_ndcg@20"

def replay(curve: pd.DataFrame, patience: int):
    e = curve[curve["evaluated"] == True].sort_values("epoch")
    best_v, best_e, bad = -1.0, None, 0
    for _, row in e.iterrows():
        v = row.get(COL)
        if v is None or pd.isna(v):
            continue
        if v > best_v:
            best_v, best_e, bad = float(v), int(row["epoch"]), 0
        else:
            bad += 1
            if bad >= patience:
                return best_e, best_v, int(row["epoch"])
    return best_e, best_v, int(e["epoch"].max())

rows = []
for d in sorted(glob.glob("experiments/runs/*")):
    p = os.path.join(d, "curves.csv")
    if not os.path.exists(p):
        continue
    c = pd.read_csv(p)
    if len(c) < 5 or COL not in c.columns:
        continue
    n = os.path.basename(d).split("_")
    cohort, seed, model = n[0], int(n[-2]), "_".join(n[1:-2])
    rec = dict(cohort=cohort, model=model, seed=seed, chay_den=int(c["epoch"].max()))
    for pat in (5, 10, 20):
        be, bv, stop = replay(c, pat)
        rec[f"p{pat}_best"] = be
        rec[f"p{pat}_valid"] = round(bv, 6)
    rows.append(rec)

t = pd.DataFrame(rows).sort_values(["cohort", "model", "seed"])
pd.set_option("display.width", 250)
print(t.to_string(index=False))
print("\n=== So run bi doi lua chon neu siet patience ===")
for pat in (5, 10):
    ch = t[t[f"p{pat}_best"] != t["p20_best"]]
    print(f"  patience={pat:>2} (={pat*EVAL_EVERY} epoch): {len(ch)}/{len(t)} run doi best_epoch")
