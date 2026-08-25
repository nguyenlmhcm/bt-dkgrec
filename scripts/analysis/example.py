"""One real visitor, worked through formula (3.16)-(3.18) by hand."""
import json, sys
sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
import numpy as np, pandas as pd
from pathlib import Path

interim = Path("/root/ThS_HUIT/bt-dkgrec/data/interim/original")
ev = pd.read_parquet(interim/"events.parquet")
T = int(json.loads((interim/"split.json").read_text())["t_train"])
DAY = 86_400_000
tr = ev[ev["split"]=="train"].copy()
tr["behavior"] = tr["behavior"].astype(str)

# a visitor with several behaviour types spread over time
g = tr.groupby("visitorid").agg(n=("itemid","size"), kinds=("behavior","nunique"),
                                span=("timestamp", lambda s: (s.max()-s.min())/DAY))
cand = g[(g["n"].between(4,7)) & (g["kinds"]>=2) & (g["span"]>20)]
vid = int(cand.index[0])

rows = tr[tr["visitorid"]==vid].sort_values("timestamp")
print(f"visitor {vid} — {len(rows)} tuong tac trong train")
print(f"T_train (moc quan sat tau) = {T}  ->  {pd.to_datetime(T, unit='ms').date()}\n")

ALPHA = {"view":1.0, "addtocart":2.0, "transaction":3.0}
print(f"{'item':>8}{'hanh vi':>12}{'ngay':>13}{'dt (ngay)':>11}"
      f"{'alpha':>7}{'exp(-0,01dt)':>14}{'w':>9}")
print("-"*76)
for _, r in rows.iterrows():
    dt = max(0.0, (T - r["timestamp"])/DAY)
    a = ALPHA[r["behavior"]]
    d = np.exp(-0.01*dt)
    print(f"{int(r['itemid']):>8}{r['behavior']:>12}"
          f"{str(pd.to_datetime(r['timestamp'], unit='ms').date()):>13}"
          f"{dt:>11.1f}{a:>7.1f}{d:>14.4f}{a*d:>9.4f}")

rows = rows.assign(dt=np.maximum(0.0,(T-rows["timestamp"])/DAY))
rows["w"] = rows["behavior"].map(ALPHA)*np.exp(-0.01*rows["dt"])
print("\nW(u,i) = tong w theo tung cap (cong thuc 3.18):")
for item, w in rows.groupby("itemid")["w"].sum().items():
    n = (rows["itemid"]==item).sum()
    print(f"   canh visitor {vid} -- item {int(item)}:  W = {w:.4f}   (gop tu {n} tuong tac)")

print("\n" + "="*76)
print("DOI MOC QUAN SAT tau -> CUNG DU LIEU, DO THI KHAC")
print("="*76)
print(f"{'tau':>26}", end="")
for item in sorted(rows['itemid'].unique()):
    print(f"{'item '+str(int(item)):>14}", end="")
print()
for shift in (0, 30, 60):
    tau = T - shift*DAY
    sub = rows[rows["timestamp"] <= tau].copy()
    label = f"T_train - {shift} ngay" if shift else "T_train (dang dung)"
    print(f"{label:>26}", end="")
    if sub.empty:
        print("  (khong con canh nao)"); continue
    sub["dt"] = np.maximum(0.0,(tau-sub["timestamp"])/DAY)
    sub["w"] = sub["behavior"].map(ALPHA)*np.exp(-0.01*sub["dt"])
    agg = sub.groupby("itemid")["w"].sum()
    for item in sorted(rows['itemid'].unique()):
        v = agg.get(item)
        print(f"{('—' if v is None else f'{v:.4f}'):>14}", end="")
    print()
