"""Ve lai sau hinh Chuong 4 cho quyen in: dung kho 16 cm, font Times, chu nam ngang.

Chi doi cach trinh bay. Thanh sai so chi ve phia ngoai cot de khong de len cot. So doc tu experiments/runs/ qua load_runs(), cung nguon voi
bang ket qua. Mau: mo hinh de xuat do, con lai xam, in den trang van doc duoc vi
moi cot co ten va so ngay ben canh.

Chay:
    python scripts/22_bieu_do_chuong4.py     # ghi docs/figures/fig_v20_4_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.reporting import assert_no_duplicate_cells, load_runs  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
OUT = ROOT / "docs" / "figures"
K = 20
WIDTH = 6.3  # 16 cm
INK, MUTED, GRID = "#1F2328", "#5B6068", "#DDDFE3"
METRICS = [("recall", "Recall"), ("ndcg", "NDCG"), ("hit_rate", "HitRate"), ("coverage", "Coverage")]
#: Thu tu tren hinh: tu tren xuong, mo hinh de xuat cuoi cung.
MODELS = ("popularity", "recent_popularity", "lightgcn", "static_kg_gcn", "bt_dkgrec", "bt_dkgrec_l05")
LABEL = {
    "popularity": "Popularity",
    "recent_popularity": "Recent Popularity",
    "lightgcn": "LightGCN",
    "static_kg_gcn": "Static KG-GCN",
    "bt_dkgrec": "BT-DKGRec-GCN (λ=0.01)",
    "bt_dkgrec_l05": "BT-DKGRec-GCN (λ=0.05)",
}
COLOR = {
    "popularity": "#C9CCD1",
    "recent_popularity": "#C9CCD1",
    "lightgcn": "#9CA0A8",
    "static_kg_gcn": "#6C717A",
    "bt_dkgrec": "#E2A69E",
    "bt_dkgrec_l05": "#B3261E",
}
LINE = {"lightgcn": ":", "static_kg_gcn": "--", "bt_dkgrec": "-.", "bt_dkgrec_l05": "-"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED,
        "ytick.color": INK, "legend.frameon": False, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.facecolor": "white",
    })


def so(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def so_le(values) -> int:
    """So le chung cho mot o: du de gia tri nho nhat khac 0 con hai chu so co nghia."""
    nho = min((v for v in values if v > 0), default=1)
    return 5 if nho < 0.001 else 4


def save(fig, name: str) -> Path:
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def so_sanh(frame: pd.DataFrame, cohort: str, name: str) -> Path:
    """Sau mo hinh, bon chi so: cot ngang, ten mo hinh ghi mot lan ben trai."""
    sub = frame[frame["cohort"] == cohort]
    models = [m for m in MODELS if m in set(sub["model"])]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.35), sharey=True,
                             gridspec_kw={"wspace": 0.28})
    y = np.arange(len(models))[::-1]
    for ax, (metric, title) in zip(axes, METRICS):
        col = f"{metric}@{K}"
        mean = [sub[sub["model"] == m][col].mean() for m in models]
        std = [sub[sub["model"] == m][col].std(ddof=0) for m in models]
        ax.barh(y, mean, height=0.68, color=[COLOR[m] for m in models])
        ax.errorbar(mean, y, xerr=[np.zeros(len(std)), std], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=1.8, capthick=0.8)
        top = max(a + b for a, b in zip(mean, std))
        d = so_le(mean)
        for yi, a, b, m in zip(y, mean, std, models):
            ax.text(a + b + top * 0.04, yi, so(a, d), va="center", fontsize=8.2, color=INK,
                    fontweight="bold" if m == "bt_dkgrec_l05" else "normal")
        ax.set_xlim(0, top * 1.62)
        ax.set_title(f"{title}@{K}", pad=5)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [LABEL[m] for m in models])
    return save(fig, name)


def duong_cong(runs: Path, cohort: str, name: str) -> Path:
    """Loss va NDCG@20 tren tap xac thuc theo epoch, seed 2020 cua bon mo hinh do thi."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"wspace": 0.2})
    for model, ls in LINE.items():
        files = sorted(runs.glob(f"{cohort}_{model}_2020_*/curves.csv"))
        if len(files) != 1:
            raise SystemExit(f"LOI: {cohort}/{model} seed 2020 co {len(files)} run")
        data = pd.read_csv(files[0])
        kw = {"color": COLOR[model], "linestyle": ls, "linewidth": 1.8 if model == "bt_dkgrec_l05" else 1.3,
              "label": LABEL[model]}
        if model == "bt_dkgrec":
            kw["color"] = "#C97F74"
        axes[0].plot(data["epoch"], data["loss"], **kw)
        do = data[data["valid_ndcg@20"].notna()]
        axes[1].plot(do["epoch"], do["valid_ndcg@20"], **kw)
    axes[0].set_title("Loss huấn luyện (seed 2020)", pad=5)
    axes[1].set_title("NDCG@20 trên tập xác thực (seed 2020)", pad=5)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8.5)
        ax.set_xlim(left=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.19),
               handlelength=2.4, columnspacing=1.1)
    return save(fig, name)


def tinh_dong(frame: pd.DataFrame, name: str, pair=("static_kg_gcn", "bt_dkgrec")) -> Path:
    """Cap ablation: moi nhom nguoi dung mot o, bon chi so, hai cot ngang."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), sharey=True, gridspec_kw={"wspace": 0.1})
    y = np.arange(4)[::-1].astype(float)
    for ax, (cohort, title) in zip(axes, (("original", "Nhóm Original"), ("active", "Nhóm Active"))):
        sub = frame[frame["cohort"] == cohort]
        top = max((sub[sub["model"].isin(pair)][f"{m}@{K}"].mean() for m, _ in METRICS))
        for off, model in zip((0.2, -0.2), pair):
            mean = [sub[sub["model"] == model][f"{m}@{K}"].mean() for m, _ in METRICS]
            std = [sub[sub["model"] == model][f"{m}@{K}"].std(ddof=0) for m, _ in METRICS]
            ax.barh(y + off, mean, height=0.37, color=COLOR[model], label=LABEL[model])
            ax.errorbar(mean, y + off, xerr=[np.zeros(len(std)), std], fmt="none", ecolor=MUTED, elinewidth=0.8, capsize=1.8, capthick=0.8)
            for yi, a, b in zip(y + off, mean, std):
                ax.text(a + b + top * 0.025, yi, so(a, 4), va="center", fontsize=8.2, color=INK)
        ax.set_xlim(0, top * 1.5)
        ax.set_title(title, pad=5)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [f"{t}@{K}" for _, t in METRICS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    return save(fig, name)


def co_chua_lich_su(warm: pd.DataFrame, cold: pd.DataFrame, cohort: str, name: str) -> Path:
    """Hai mo hinh dua tren do pho bien, hai nhom nguoi dung, bao canh nhau."""
    w, c = warm[warm["cohort"] == cohort], cold[cold["cohort"] == cohort]
    models = [m for m in ("popularity", "recent_popularity") if m in set(c["model"])]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.75), sharey=True, gridspec_kw={"wspace": 0.28})
    y = np.arange(len(models))[::-1].astype(float)
    for ax, (metric, title) in zip(axes, METRICS):
        col = f"{metric}@{K}"
        a = [w[w["model"] == m][col].mean() for m in models]
        b = [c[c["model"] == m][col].mean() for m in models]
        ax.barh(y + 0.19, a, height=0.36, color="#6C717A", label="Có lịch sử")
        ax.barh(y - 0.19, b, height=0.36, color="#C9CCD1", label="Chưa có lịch sử")
        top, d = max(a + b), so_le(a + b)
        for yi, v in [*zip(y + 0.19, a), *zip(y - 0.19, b)]:
            ax.text(v + top * 0.04, yi, so(v, d), va="center", fontsize=8.2, color=INK)
        ax.set_xlim(0, top * 1.68)
        ax.set_title(f"{title}@{K}", pad=5)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [LABEL[m] for m in models])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.55, -0.13))
    return save(fig, name)


def main(runs: Path = RUNS) -> dict[str, Path]:
    style()
    warm = load_runs(runs, split="test", segment="warm")
    assert_no_duplicate_cells(warm)
    cold = load_runs(runs, split="test", segment="cold")
    return {
        "Hình 4.1.": so_sanh(warm, "original", "fig_v20_4_1_so_sanh_original"),
        "Hình 4.2.": duong_cong(runs, "original", "fig_v20_4_2_duong_cong_original"),
        "Hình 4.3.": so_sanh(warm, "active", "fig_v20_4_3_so_sanh_active"),
        "Hình 4.4.": duong_cong(runs, "active", "fig_v20_4_4_duong_cong_active"),
        "Hình 4.5.": tinh_dong(warm, "fig_v20_4_5_tinh_dong"),
        "Hình 4.6.": co_chua_lich_su(warm, cold, "original", "fig_v20_4_6_lich_su_original"),
    }


if __name__ == "__main__":
    for cap, path in main().items():
        print(cap, path.name)
