"""Aggregate run artifacts into the tables the thesis reports.

Reads **every** run in ``experiments/runs/`` -- there is deliberately no seed
filter, so a run cannot be quietly excluded because its numbers are inconvenient
(CLAUDE.md, fairness rule 11 and integrity rule 12).

Results are reported as ``mean +/- std`` over the seed set. Models are compared
with Welch's t-test: the two runs are independent samples, not paired, because
sparse CUDA kernels are not bit-exact even at a fixed seed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

log = get_logger(__name__)

METRICS = ("recall", "ndcg", "hit_rate", "coverage")
METRIC_LABELS = {
    "recall": "Recall",
    "ndcg": "NDCG",
    "hit_rate": "HitRate",
    "coverage": "Coverage",
}
#: Canonical model order for every table and figure: baselines first, ablation,
#: then the proposed model. Fixed so readers can compare figures side by side.
MODEL_ORDER = (
    "popularity", "recent_popularity", "lightgcn", "static_kg_gcn",
    "bt_dkgrec", "bt_dkgrec_l05",
)
MODEL_LABELS = {
    "popularity": "Popularity",
    "recent_popularity": "Recent Popularity",
    "lightgcn": "LightGCN",
    "static_kg_gcn": "Static KG-GCN",
    "bt_dkgrec": "BT-DKGRec-GCN (λ=0,01)",
    "bt_dkgrec_l05": "BT-DKGRec-GCN (λ=0,05)",
}
#: Models that cannot be run without training; used to flag deterministic std=0.
DETERMINISTIC_MODELS = ("popularity", "recent_popularity")


def load_runs(runs_dir: Path, split: str = "test", segment: str = "warm") -> pd.DataFrame:
    """Load one row per run artifact.

    Args:
        runs_dir: ``experiments/runs/``.
        split: ``"valid"`` or ``"test"``.
        segment: ``"warm"``, one of the degree bands ``"warm_deg1"`` /
            ``"warm_deg2"`` / ``"warm_deg3plus"``, ``"cold"`` or ``"all"``.

    Returns:
        Long-format frame with ``cohort``, ``model``, ``seed``, ``n_users`` and
        one column per ``metric@k``. Runs whose segment is unmeasured (a
        personalised model on cold users) are skipped rather than counted as 0.

    Raises:
        FileNotFoundError: If ``runs_dir`` does not exist.
    """
    if not runs_dir.exists():
        raise FileNotFoundError(f"chua co thu muc run: {runs_dir}")

    rows: list[dict[str, object]] = []
    skipped = 0
    for path in sorted(runs_dir.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = payload.get(split, {})
        values = block.get(segment)
        if values is None:
            skipped += 1
            continue
        rows.append(
            {
                "run_id": payload["run_id"],
                "cohort": payload["cohort"],
                "model": payload["model"],
                "seed": payload["seed"],
                "n_users": block["n_users"][segment],
                **values,
            }
        )

    frame = pd.DataFrame(rows)
    log.info(
        "doc %s run tu %s (split=%s, segment=%s); bo qua %s run khong do duoc phan doan nay",
        f"{len(frame):,}", runs_dir, split, segment, f"{skipped:,}",
    )
    return frame


def assert_no_duplicate_cells(frame: pd.DataFrame) -> None:
    """Moi o (cohort, model, seed) chi duoc co DUNG MOT run.

    Day khong phai lo xa. `load_runs()` doc moi thu muc trong `experiments/runs/`
    va khong loc trung: neu mot run cu (vi du ngan sach dung som khac) con nam
    canh run moi, `summarize()` se lay trung binh hai cau hinh khac nhau vao cung
    mot o va bang van in ra binh thuong. Do la hong am tham -- dung loai loi ma
    ca du an nay dung guard de chan. Hinh ve dung chung guard nay voi bang, de
    khong bao gio co chuyen hinh va bang trong cung mot chuong noi hai so khac
    nhau.
    """
    if frame.empty:
        return
    counts = Counter(zip(frame["cohort"], frame["model"], frame["seed"]))
    duplicates = {cell: n for cell, n in counts.items() if n > 1}
    if not duplicates:
        return
    lines = [f"  {c}/{m}/{s}: {n} run" for (c, m, s), n in sorted(duplicates.items())]
    raise SystemExit(
        "LOI: co o bi trung run — bang se tron nhieu cau hinh vao mot so.\n"
        + "\n".join(lines)
        + "\n\nXoa run cu di roi chay lai. Du lieu khong mat: moi run deu con "
        "trong lich su git."
    )


def summarize(frame: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Mean and standard deviation per ``(cohort, model)`` at cut-off ``k``.

    ``std`` is 0 for deterministic baselines. That is a property of the model,
    not a computation error, and must be footnoted wherever the table appears
    (docs/DECISIONS.md muc D8).
    """
    if frame.empty:
        return pd.DataFrame()

    columns = [f"{m}@{k}" for m in METRICS if f"{m}@{k}" in frame.columns]
    grouped = frame.groupby(["cohort", "model"], sort=False)

    summary = grouped[columns].agg(["mean", "std"]).fillna(0.0)
    summary[("n_seeds", "")] = grouped["seed"].nunique()
    summary[("n_users", "")] = grouped["n_users"].first()

    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    summary = summary.reset_index()
    sort_key = summary["model"].map(lambda m: order.get(m, len(order)))
    summary = summary.assign(_order=sort_key).sort_values(["cohort", "_order"])
    summary = summary.drop(columns=["_order"], level=0)
    return summary.set_index(["cohort", "model"])


def format_table(summary: pd.DataFrame, k: int = 20, decimals: int = 6) -> str:
    """Render the summary as a Markdown table ready to paste into the thesis."""
    if summary.empty:
        return "(chua co run nao)"

    header = ["Cohort", "Mo hinh", "Users", "Seeds"] + [
        f"{METRIC_LABELS[m]}@{k}" for m in METRICS if (f"{m}@{k}", "mean") in summary.columns
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for (cohort, model), row in summary.iterrows():
        cells = [cohort, MODEL_LABELS.get(model, model),
                 f"{int(row[('n_users', '')]):,}", f"{int(row[('n_seeds', '')])}"]
        for metric in METRICS:
            key = f"{metric}@{k}"
            if (key, "mean") not in summary.columns:
                continue
            mean, std = row[(key, "mean")], row[(key, "std")]
            cells.append(f"{mean:.{decimals}f} ± {std:.{decimals}f}")
        lines.append("| " + " | ".join(cells) + " |")

    if any(m in summary.index.get_level_values("model") for m in DETERMINISTIC_MODELS):
        lines.append("")
        lines.append(
            "*Ghi chu: std = 0 o Popularity va Recent Popularity la do hai mo hinh nay "
            "TAT DINH — khong phu thuoc seed. Day la tinh chat cua mo hinh, khong phai "
            "loi tinh toan.*"
        )
    return "\n".join(lines)


def compare_models(
    frame: pd.DataFrame, model_a: str, model_b: str, cohort: str, metric: str = "ndcg@20"
) -> dict[str, object]:
    """Welch's t-test between two models on one cohort.

    Welch (unequal variance, **independent** samples) rather than a paired test:
    two runs at the same seed are not matched observations, because sparse CUDA
    operations are not bit-exact (CLAUDE.md muc "Non-determinism").

    Returns:
        Means, difference, t statistic, p-value and sample sizes. ``p_value`` is
        ``None`` when either side has fewer than two runs or zero variance on
        both sides -- reporting a p-value there would be meaningless.
    """
    a = frame[(frame["cohort"] == cohort) & (frame["model"] == model_a)][metric].to_numpy()
    b = frame[(frame["cohort"] == cohort) & (frame["model"] == model_b)][metric].to_numpy()

    result: dict[str, object] = {
        "cohort": cohort, "metric": metric,
        "model_a": model_a, "model_b": model_b,
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()) if len(a) else None,
        "mean_b": float(b.mean()) if len(b) else None,
        "difference": float(a.mean() - b.mean()) if len(a) and len(b) else None,
        "t_statistic": None, "p_value": None,
    }
    if len(a) < 2 or len(b) < 2:
        result["note"] = "can it nhat 2 run moi ben moi kiem dinh duoc"
        return result
    if np.isclose(a.var(), 0) and np.isclose(b.var(), 0):
        result["note"] = "ca hai deu tat dinh (phuong sai 0) — khong co gi de kiem dinh"
        return result

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    result["t_statistic"] = float(t_stat)
    result["p_value"] = float(p_value)
    return result
