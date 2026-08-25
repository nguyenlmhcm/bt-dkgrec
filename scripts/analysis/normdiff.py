"""After D^-1/2 A D^-1/2, how different are the dynamic and static graphs really?"""
import sys; sys.path.insert(0,'/root/ThS_HUIT/bt-dkgrec')
import numpy as np, scipy.sparse as sp
from pathlib import Path
from src.graph.normalize import symmetric_normalize

for cohort in ("original",):
    A = {}
    for model in ("bt_dkgrec","static_kg_gcn"):
        p = Path(f"/root/ThS_HUIT/bt-dkgrec/data/processed/{cohort}/{model}/adjacency.npz")
        A[model] = symmetric_normalize(sp.load_npz(p)).tocsr()
    d, s = A["bt_dkgrec"], A["static_kg_gcn"]
    print(f"=== {cohort} ===")
    print(f"  cung so phan tu khac 0? {d.nnz == s.nnz}  ({d.nnz:,} vs {s.nnz:,})")
    dv, svv = d.data, s.data
    rel = np.abs(dv - svv) / np.maximum(np.abs(svv), 1e-12)
    print(f"  lech tuong doi giua hai ma tran chuan hoa:")
    for q in (50, 75, 90, 99):
        print(f"     p{q:<3} {np.percentile(rel, q):.4f}")
    print(f"     trung binh {rel.mean():.4f}   toi da {rel.max():.4f}")
    print(f"  ty le phan tu lech > 10%: {(rel > 0.10).mean():.1%}")
    print(f"  ty le phan tu lech <  1%: {(rel < 0.01).mean():.1%}")
