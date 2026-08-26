#!/usr/bin/env python
"""Sinh cac hinh cho Chuong 4 tu du lieu trong `experiments/runs/`.

Vi sao la script chu khong phai anh chup man hinh
--------------------------------------------------
Hinh trong bao cao phai noi cung mot con so voi bang o ngay tren no. Cach duy
nhat bao dam duoc dieu do la ca hai cung doc mot nguon: `load_runs()` cho bang
diem va `curves.csv` cho duong loss, va cung di qua `assert_no_duplicate_cells`.
Khi cac run original chay xong, chay lai script la hinh tu cap nhat.

Bam theo `De_an_thac_si_v11.docx`, Chuong 4 co dung bon hinh:

    Hinh 4.1  So sanh cac chi so test tren nhom nguoi dung co lich su ban dau
    Hinh 4.2  Duong loss huan luyen cua cac mo hinh tren original split
    Hinh 4.3  So sanh cac chi so test tren nhom nguoi dung tich cuc
    Hinh 4.4  Duong loss huan luyen cua cac mo hinh tren active split

Kich thuoc anh giu dung ty le cua v11: 5.71 x 3.99 inch cho bieu do cot va
5.71 x 3.23 inch cho duong loss.

Mot vai quyet dinh ve cach doc hinh, ghi ra de nguoi doc khong phai doan:

* Mau gan CO DINH theo mo hinh (bang `MODEL_COLORS`), khong cap phat theo thu
  tu xuat hien. Nho vay Hinh 4.1 va Hinh 4.3 dung mot mau cho cung mot mo hinh,
  va khi mot mo hinh chua chay xong thi cac mo hinh con lai khong bi doi mau.
* Bang mau la Okabe-Ito, bang dinh tinh chuan cho nguoi mu mau. Ngoai mau, moi
  duong loss con co kieu net rieng nen danh tinh khong bao gio chi nam o mau.
* Truc loss dung thang log. Loss BPR roi tu ~0.6 xuong ~0.006; tren thang tuyen
  tinh moi duong deu dinh vao 0 sau epoch 50 va khong con doc duoc gi.
* Duong loss la trung binh theo cac seed hien co, dai bang mau xam la khoang
  min-max giua cac seed. Cac seed dung som o epoch khac nhau, nen phan duoi cua
  duong dai nhat co the chi con mot seed; do la ly do dai min-max hep dan ve
  cuoi chu khong phai vi phuong sai giam.

Chay:
    python scripts/09_make_figures.py
    python scripts/09_make_figures.py --runs-dir experiments/runs --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reporting import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    assert_no_duplicate_cells,
    load_runs,
)
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

#: Okabe-Ito, gan co dinh theo mo hinh. Hai heuristic dung xam vi chung khong hoc.
MODEL_COLORS = {
    "popularity": "#999999",
    "recent_popularity": "#56B4E9",
    "lightgcn": "#E69F00",
    "static_kg_gcn": "#009E73",
    "bt_dkgrec": "#0072B2",
    "bt_dkgrec_l05": "#CC79A7",
}
#: Danh tinh khong chi nam o mau: moi mo hinh co them mot kieu net rieng.
MODEL_LINESTYLES = {
    "lightgcn": "-",
    "static_kg_gcn": "--",
    "bt_dkgrec": "-.",
    "bt_dkgrec_l05": (0, (3, 1, 1, 1, 1, 1)),
}
#: Ba chi so dung trong Bang 4.5 va 4.6. Coverage bi bo ra vi nho hon hai bac
#: do lon, ve chung mot truc se lam ba chi so kia bep xuong thanh mot vach.
FIGURE_METRICS = ("recall@20", "ndcg@20", "hit_rate@20")
METRIC_TICKS = {"recall@20": "Recall@20", "ndcg@20": "NDCG@20",
                "hit_rate@20": "HitRate@20"}
COHORT_TITLE = {"original": "nhóm người dùng có lịch sử ban đầu",
                "active": "nhóm người dùng tích cực"}
BAR_SIZE = (5.71, 3.99)
LOSS_SIZE = (5.71, 3.23)
DPI = 200

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "figure.dpi": DPI,
})


def strip_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# ── Hinh 4.1 / 4.3: so sanh chi so test ─────────────────────────────────


def bar_values(subset: pd.DataFrame, model: str) -> tuple[list[float], list[float]]:
    """Chieu cao cot va thanh sai so cua mot mo hinh, theo `FIGURE_METRICS`.

    Tach rieng khoi phan ve de test doc duoc dung con so se ve len giay: hinh
    va bang o ngay tren no khong bao gio duoc noi hai so khac nhau.
    """
    rows = subset[subset["model"] == model]
    means = [float(rows[m].mean()) for m in FIGURE_METRICS]
    stds = [float(rows[m].std(ddof=1)) if len(rows) > 1 else 0.0
            for m in FIGURE_METRICS]
    return means, stds


def figure_metrics(frame: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Bieu do cot ba chi so test tai K = 20, mot cot moi mo hinh.

    Thanh sai so la do lech chuan giua cac seed; hai heuristic tat dinh nen
    khong co thanh sai so, dung nhu cot `±0.000000` trong bang.
    """
    subset = frame[frame["cohort"] == cohort]
    models = [m for m in MODEL_ORDER if m in set(subset["model"])]
    if not models:
        raise SystemExit(f"LOI: khong co run nao cho cohort {cohort!r}.")

    fig, ax = plt.subplots(figsize=BAR_SIZE)
    width = 0.8 / len(models)
    for slot, model in enumerate(models):
        means, stds = bar_values(subset, model)
        offset = (slot - (len(models) - 1) / 2) * width
        ax.bar([i + offset for i in range(len(FIGURE_METRICS))], means,
               width=width * 0.9, label=label(model),
               color=MODEL_COLORS[model], edgecolor="white", linewidth=0.8)
        ax.errorbar([i + offset for i in range(len(FIGURE_METRICS))], means,
                    yerr=stds, fmt="none", ecolor="#444444",
                    elinewidth=0.8, capsize=2.5)

    ax.set_xticks(range(len(FIGURE_METRICS)))
    ax.set_xticklabels([METRIC_TICKS[m] for m in FIGURE_METRICS])
    ax.set_ylabel("Giá trị chỉ số")
    ax.set_xlabel(f"Chỉ số kiểm thử tại K = 20 — {COHORT_TITLE[cohort]}")
    ax.grid(axis="x", visible=False)
    strip_spines(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("da ve %s (%d mo hinh)", path, len(models))
    return path


# ── Hinh 4.2 / 4.4: duong loss huan luyen ───────────────────────────────


def load_curves(runs_dir: Path, cohort: str) -> pd.DataFrame:
    """Gop `curves.csv` cua moi run trong cohort, chi giu cac epoch co loss.

    Hai heuristic ghi mot dong duy nhat voi `loss` rong va ghi chu "mo hinh
    khong hoc"; chung bi loai o day chu khong bi quen.
    """
    frames = []
    for path in sorted(runs_dir.glob(f"{cohort}_*/curves.csv")):
        curve = pd.read_csv(path)
        curve["run"] = path.parent.name
        frames.append(curve)
    if not frames:
        raise SystemExit(f"LOI: khong tim thay curves.csv nao cho cohort {cohort!r}.")
    curves = pd.concat(frames, ignore_index=True)
    return curves[curves["loss"].notna() & (curves["cohort"] == cohort)]


def figure_loss(curves: pd.DataFrame, cohort: str, path: Path) -> Path:
    """Duong loss theo epoch: trung binh giua cac seed, dai min-max lam nen."""
    models = [m for m in MODEL_ORDER if m in set(curves["model"])]
    if not models:
        raise SystemExit(f"LOI: khong co mo hinh nao co duong loss o cohort {cohort!r}.")

    fig, ax = plt.subplots(figsize=LOSS_SIZE)
    for model in models:
        rows = curves[curves["model"] == model]
        grouped = rows.groupby("epoch")["loss"]
        mean, low, high = grouped.mean(), grouped.min(), grouped.max()
        if len(rows["seed"].unique()) > 1:
            ax.fill_between(mean.index, low, high, color=MODEL_COLORS[model],
                            alpha=0.15, linewidth=0)
        ax.plot(mean.index, mean.values, label=label(model),
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES.get(model, "-"), linewidth=1.6)

    ax.set_yscale("log")
    ax.set_xlabel(f"Epoch — {COHORT_TITLE[cohort]}")
    ax.set_ylabel("Loss huấn luyện (thang log)")
    strip_spines(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("da ve %s (%d mo hinh)", path, len(models))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_runs(args.runs_dir, split="test", segment="warm")
    assert_no_duplicate_cells(frame)

    written = []
    for cohort in ("original", "active"):
        if cohort not in set(frame["cohort"]):
            log.warning("bo qua cohort %s: chua co run nao", cohort)
            continue
        written.append(figure_metrics(frame, cohort, args.out_dir / f"{cohort}_metrics.png"))
        written.append(figure_loss(load_curves(args.runs_dir, cohort), cohort,
                                   args.out_dir / f"{cohort}_loss.png"))

    print(f"Da ghi {len(written)} hinh vao {args.out_dir}:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
