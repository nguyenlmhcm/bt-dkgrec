#!/usr/bin/env python
"""Buoc 9 — sinh toan bo bang ket qua tu `experiments/runs/`.

Vi sao buoc nay ton tai
-----------------------
Truoc khi co no, moi bang trong de an deu duoc chep tay tu man hinh terminal.
Chep tay thi khong ai kiem lai duoc, va mot con so go nham khong de lai dau vet.
Buoc 9 bien `experiments/runs/` thanh nguon su that duy nhat: bang nao trong de
an cung phai sinh ra tu day.

Bang duoc sinh
--------------
* Bang chinh    -- mean +/- std moi (cohort, mo hinh) tren `test`/`warm`.
* Bang phan tang -- cung nhu tren nhung tach theo bac cua nguoi dung (D34).
* Bang kiem dinh -- Welch cho cac cap quan trong, kem bien the ghep cap.

Ve phep kiem ghep cap
---------------------
`compare_models()` dung Welch (mau doc lap) theo dung quy tac trong CLAUDE.md.
Script nay in THEM bien the ghep cap theo seed, nhung danh dau ro la CHUA DUOC
CHOT -- vi phan ra phuong sai cho thay 73-86% bien thien la hieu ung seed chung
(docs/PHAN_TICH_KET_QUA.md muc 1). Bao cao ca hai, khong am tham doi phep kiem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluator import DEGREE_BANDS  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    assert_no_duplicate_cells,
    MODEL_LABELS,
    compare_models,
    format_table,
    load_runs,
    summarize,
)
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

#: Cap mo hinh dang duoc bao cao, theo thu tu quan trong doi voi de an.
#: Cap dau tien la cap chung minh chu "dong" trong ten de tai.
COMPARISONS = (
    ("bt_dkgrec", "static_kg_gcn"),        # dong vs tinh  <- cap cot loi
    ("bt_dkgrec_l05", "static_kg_gcn"),
    ("bt_dkgrec", "lightgcn"),             # do thi tri thuc co ich khong
    ("bt_dkgrec_l05", "lightgcn"),
    ("static_kg_gcn", "lightgcn"),
)


def paired_test(frame: pd.DataFrame, a: str, b: str, cohort: str, metric: str) -> dict:
    """Bien the ghep cap theo seed. CHUA DUOC CHOT -- xem docstring dau file."""
    sub = frame[(frame["cohort"] == cohort) & frame["model"].isin([a, b])]
    if sub.empty or metric not in sub.columns:
        return {"p_value": None, "n": 0}
    wide = sub.pivot(index="seed", columns="model", values=metric)
    if a not in wide.columns or b not in wide.columns:
        return {"p_value": None, "n": 0}
    diff = (wide[a] - wide[b]).dropna()
    if len(diff) < 2:
        return {"p_value": None, "n": len(diff)}
    sd = diff.std(ddof=1)
    if np.isclose(sd, 0):
        return {"p_value": None, "n": len(diff), "note": "hieu so khong doi giua cac seed"}
    t = diff.mean() / (sd / np.sqrt(len(diff)))
    return {
        "p_value": float(2 * (1 - stats.t.cdf(abs(t), len(diff) - 1))),
        "n": int(len(diff)),
        "wins": int((diff > 0).sum()),
    }


def comparison_table(frame: pd.DataFrame, cohort: str, metric: str) -> str:
    """Bang kiem dinh cho mot cohort va mot metric."""
    header = ["So sanh", "Chenh lech", "Tuong doi", "Thang", "Welch p", "Ghep cap p*"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    present = set(frame["model"]) if not frame.empty else set()

    for a, b in COMPARISONS:
        if not {a, b} <= present:
            continue
        welch = compare_models(frame, a, b, cohort, metric)
        if welch["difference"] is None:
            continue
        pair = paired_test(frame, a, b, cohort, metric)
        relative = (
            f"{welch['difference'] / welch['mean_b']:+.2%}"
            if welch["mean_b"] else "—"
        )
        lines.append("| " + " | ".join([
            f"{MODEL_LABELS.get(a, a)} vs {MODEL_LABELS.get(b, b)}",
            f"{welch['difference']:+.6f}",
            relative,
            f"{pair.get('wins', '—')}/{pair.get('n', '—')}",
            f"{welch['p_value']:.4f}" if welch["p_value"] is not None else "—",
            f"{pair['p_value']:.4f}" if pair["p_value"] is not None else "—",
        ]) + " |")

    lines.append("")
    lines.append(
        "*\\* Cot ghep cap CHUA duoc chot lam phep kiem chinh thuc. Quy tac hien hanh "
        "trong CLAUDE.md la Welch. Bao cao ca hai vi phan ra phuong sai cho thay "
        "73-86% bien thien la hieu ung seed chung — xem docs/PHAN_TICH_KET_QUA.md muc 1. "
        "Can y kien nguoi huong dan truoc khi chon mot phep kiem de bao cao.*"
    )
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    log.info("da ghi %s (%s dong)", path, f"{len(text.splitlines()):,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Buoc 9 — sinh bang ket qua")
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tables"))
    parser.add_argument("--split", default="test", choices=("test", "valid"))
    parser.add_argument("--k", type=int, default=20, choices=(10, 20))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    warm = load_runs(args.runs_dir, split=args.split, segment="warm")
    assert_no_duplicate_cells(warm)
    if warm.empty:
        raise SystemExit(f"LOI: khong doc duoc run nao tu {args.runs_dir}")

    cohorts = list(dict.fromkeys(warm["cohort"]))

    # ── Bang chinh ──────────────────────────────────────────────────────
    main_table = format_table(summarize(warm, k=args.k), k=args.k)
    write(args.out_dir / f"bang_chinh_{args.split}_warm_k{args.k}.md",
          f"# Ket qua chinh — {args.split} / warm / @{args.k}\n\n{main_table}")

    # ── Bang phan tang theo bac (D34) ───────────────────────────────────
    blocks = []
    for band in DEGREE_BANDS:
        frame = load_runs(args.runs_dir, split=args.split, segment=band)
        assert_no_duplicate_cells(frame)
        if frame.empty:
            blocks.append(f"## {band}\n\n(khong run nao do duoc phan doan nay)")
            continue
        n_users = frame.groupby("cohort")["n_users"].first().to_dict()
        note = ", ".join(f"{c}: {int(n):,} user" for c, n in n_users.items())
        blocks.append(f"## {band}  ({note})\n\n"
                      f"{format_table(summarize(frame, k=args.k), k=args.k)}")
    write(args.out_dir / f"bang_phan_tang_{args.split}_k{args.k}.md",
          "# Phan tang theo bac cua nguoi dung (D34)\n\n"
          "Chuan hoa doi xung dua trong so vao lan truyen theo CAN BAC HAI: voi nguoi dung\n"
          "mot canh he so la sqrt(W)/sqrt(d_i), nen ti le 3:1 giua transaction va view\n"
          "con khoang 1,73:1. Bac nguoi dung la bien dong hanh manh nhat cua luong tin\n"
          "hieu ca nhan hoa, nen tach ra de thay hieu qua den tu nhom nao.\n\n"
          + "\n\n".join(blocks))

    # ── Bang kiem dinh ──────────────────────────────────────────────────
    blocks = []
    for cohort in cohorts:
        for metric in (f"recall@{args.k}", f"ndcg@{args.k}"):
            if metric not in warm.columns:
                continue
            blocks.append(f"## {cohort} / {metric}\n\n"
                          f"{comparison_table(warm, cohort, metric)}")
    write(args.out_dir / f"bang_kiem_dinh_{args.split}_k{args.k}.md",
          f"# Kiem dinh thong ke — {args.split} / warm / @{args.k}\n\n"
          + "\n\n".join(blocks))

    print(f"\nDa sinh bang trong {args.out_dir}/")
    for path in sorted(args.out_dir.glob("*.md")):
        print(f"  {path.name}")
    print(f"\n{main_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
