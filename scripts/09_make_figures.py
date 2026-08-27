#!/usr/bin/env python
"""Sinh hinh Chuong 4 tu `experiments/runs/`.

Script nay chi la LOP DIEU KHIEN. Toan bo viec ve nam o
`src/evaluation/figures.py` -- module do da chon mau qua bo kiem mu mau, co
hatch cho ban in trang den, va nhan gia tri tren tung cot. Khong ve lai o day:
hinh va bang phai cung mot nguon (`load_runs`) va cung mot bang mau, neu khong
Chuong 4 se co hai he mau cho cung mot mo hinh.

Bam theo `De_an_thac_si_v11.docx`, Chuong 4 co dung bon hinh:

    Hinh 4.1  So sanh cac chi so test tren nhom nguoi dung co lich su ban dau
    Hinh 4.2  Duong loss huan luyen cua cac mo hinh tren original split
    Hinh 4.3  So sanh cac chi so test tren nhom nguoi dung tich cuc
    Hinh 4.4  Duong loss huan luyen cua cac mo hinh tren active split

Tieu de trong hinh bi TAT (`show_title=False`): file Word da co caption
"Hinh 4.1. ..." dat ngay duoi anh, ve them tieu de la lap hai lan.

Chay:
    python scripts/09_make_figures.py
    python scripts/09_make_figures.py --runs-dir experiments/runs --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import figures  # noqa: E402
from src.evaluation.reporting import assert_no_duplicate_cells, load_runs  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

PRIMARY_K = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    # Word nhung PNG; PDF chi can khi xuat LaTeX nen khong sinh cho nhe thu muc.
    figures.OPTIONS.formats = ("png",)
    # Caption "Hinh 4.x" nam duoi anh trong file Word roi.
    figures.OPTIONS.show_title = False

    frame = load_runs(args.runs_dir, split="test", segment="warm")
    assert_no_duplicate_cells(frame)
    cold = load_runs(args.runs_dir, split="test", segment="cold")

    written = []
    for cohort in ("original", "active"):
        if cohort not in set(frame["cohort"]):
            log.warning("bo qua cohort %s: chua co run nao", cohort)
            continue
        written.append(figures.plot_model_comparison(
            frame, cohort, args.out_dir, k=PRIMARY_K))
        curve = figures.plot_training_curves(args.runs_dir, cohort, args.out_dir)
        if curve is None:
            log.warning("cohort %s chua co run nao co duong loss", cohort)
        else:
            written.append(curve)

        # Warm va cold bao canh nhau, khong bao gio lay trung binh chung. Ham tra
        # None khi cohort khong co user cold do duoc — mot hinh rong se ngu y
        # rang segment da duoc do va bang khong.
        pair = figures.plot_warm_cold(frame, cold, cohort, args.out_dir, k=PRIMARY_K)
        if pair is None:
            log.info("cohort %s khong co user cold — bo qua hinh warm/cold", cohort)
        else:
            written.append(pair)

    # So sanh cot loi cua de tai: KG tinh so voi KG dong. Ham nay ve ca hai cohort
    # canh nhau nen goi MOT lan, ngoai vong lap.
    if {"static_kg_gcn", "bt_dkgrec"} <= set(frame["model"]):
        written.append(figures.plot_ablation_pair(frame, args.out_dir, k=PRIMARY_K))
    else:
        log.warning("chua du cap static_kg_gcn/bt_dkgrec — bo qua hinh ablation")

    print(f"Da ghi {len(written)} hinh vao {args.out_dir}:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
