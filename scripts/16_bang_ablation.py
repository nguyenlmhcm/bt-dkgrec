"""Bang 2x2 tach alpha khoi lambda (D35), sinh thang tu experiments/runs/.

Bon o cua bang:

    static_kg_gcn          alpha 1/1/1   lambda 0      (khong co ca hai)
    bt_dkgrec_alpha_only   alpha 1/2/3   lambda 0
    bt_dkgrec_time_only    alpha 1/1/1   lambda 0,05
    bt_dkgrec_l05          alpha 1/2/3   lambda 0,05   (mo hinh de xuat)

Ket luan cung do code tinh, theo nguong da ghi TRUOC khi do trong
docs/DECISIONS.md muc D35 -- khong ai chon cach doc sau khi nhin so:

    share(time_only) >= 2/3 va share(alpha_only) <= 1/3  -> chu yeu tu thoi gian
    share(alpha_only) >= 2/3                              -> chu yeu tu alpha (bac D33)
    con lai                                               -> hai thanh phan bo sung

share(x) = (x - static) / (l05 - static), tren trung binh 3 seed.

Chay:
    python scripts/16_bang_ablation.py      # ghi docs/BANG_ABLATION_D35.md
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2020, 2021, 2022]
METRICS = ["recall@20", "ndcg@20"]

#: (model, nhan, alpha, lambda) theo thu tu trinh bay.
CELLS = [
    ("static_kg_gcn", "Static KG-GCN (không có cả hai)", "1/1/1", "0"),
    ("bt_dkgrec_alpha_only", "Chỉ loại hành vi", "1/2/3", "0"),
    ("bt_dkgrec_time_only", "Chỉ thời gian", "1/1/1", "0.05"),
    ("bt_dkgrec_l05", "BT-DKGRec-GCN (cả hai)", "1/2/3", "0.05"),
]
COHORTS = [("original", "Original"), ("active", "Active")]


def doc_ket_qua(runs: Path) -> dict[tuple[str, str], dict[int, dict]]:
    """(cohort, model) -> {seed: test.warm}, chi cho bon mo hinh cua bang."""
    can = {model for model, *_ in CELLS}
    out: dict[tuple[str, str], dict[int, dict]] = {}
    for path in sorted(glob.glob(str(runs / "*" / "metrics.json"))):
        m = json.loads(Path(path).read_text())
        if m["model"] in can:
            out.setdefault((m["cohort"], m["model"]), {})[m["seed"]] = m["test"]["warm"]
    thieu = [(c, mo) for c, _ in COHORTS for mo, *_ in CELLS
             if sorted(out.get((c, mo), {})) != SEEDS]
    if thieu:
        raise SystemExit(f"LOI: chua du 3 seed o {thieu}")
    return out


def trung_binh(kq: dict, cohort: str, model: str, metric: str) -> float:
    return st.mean(v[metric] for v in kq[(cohort, model)].values())


def ty_le(kq: dict, cohort: str, model: str, metric: str) -> float:
    """Phan tang cua `model` so voi muc tang day du cua mo hinh de xuat."""
    base = trung_binh(kq, cohort, "static_kg_gcn", metric)
    full = trung_binh(kq, cohort, "bt_dkgrec_l05", metric)
    return (trung_binh(kq, cohort, model, metric) - base) / (full - base)


def ket_luan(kq: dict, cohort: str, metric: str = "recall@20") -> str:
    """Nguong D35, ap may moc."""
    a = ty_le(kq, cohort, "bt_dkgrec_alpha_only", metric)
    t = ty_le(kq, cohort, "bt_dkgrec_time_only", metric)
    if t >= 2 / 3 and a <= 1 / 3:
        return "chủ yếu từ thời gian"
    if a >= 2 / 3:
        return "chủ yếu từ loại hành vi"
    return "hai thành phần bổ sung nhau"


def thang_theo_seed(kq: dict, cohort: str, model: str, metric: str) -> int:
    base = kq[(cohort, "static_kg_gcn")]
    return sum(kq[(cohort, model)][s][metric] > base[s][metric] for s in SEEDS)


def pct(x: float) -> str:
    """Phan tram theo quy uoc de an: dau phay thap phan."""
    return f"{x:+.0%}".replace(".", ",")


def bang_markdown(kq: dict) -> str:
    dong = [
        "# Bảng phụ lục — tách loại hành vi (α) khỏi suy giảm thời gian (λ)",
        "",
        "Sinh bởi `scripts/16_bang_ablation.py` từ `experiments/runs/` — không sửa tay.",
        "Tập test, nhóm warm, 3 seed (2020, 2021, 2022). Tỉ lệ = (mô hình − Static KG-GCN)",
        "/ (BT-DKGRec-GCN − Static KG-GCN), tính trên trung bình 3 seed.",
        "Cột \"thắng Static\" đếm số seed mô hình cao hơn Static KG-GCN ở cùng seed.",
        "",
    ]
    for cohort, nhan_cohort in COHORTS:
        for metric in METRICS:
            dong += [
                f"## {nhan_cohort} — {metric.replace('recall', 'Recall').replace('ndcg', 'NDCG')}",
                "",
                "| Cấu hình | α | λ | Seed 2020 | Seed 2021 | Seed 2022 | Trung bình | Tỉ lệ | Thắng Static |",
                "|---|---|---|---:|---:|---:|---:|---:|:-:|",
            ]
            for model, nhan, alpha, lam in CELLS:
                seeds = [f"{kq[(cohort, model)][s][metric]:.6f}" for s in SEEDS]
                mean = trung_binh(kq, cohort, model, metric)
                share = "—" if model == "static_kg_gcn" else pct(ty_le(kq, cohort, model, metric))
                wins = "—" if model == "static_kg_gcn" else f"{thang_theo_seed(kq, cohort, model, metric)}/3"
                dong.append(f"| {nhan} | {alpha} | {lam} | {' | '.join(seeds)} | {mean:.6f} | {share} | {wins} |")
            dong.append("")
        dong += [f"**Kết luận theo ngưỡng D35 ({nhan_cohort}, Recall@20): {ket_luan(kq, cohort)}.**", ""]
    return "\n".join(dong)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "experiments" / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "BANG_ABLATION_D35.md")
    args = parser.parse_args()
    kq = doc_ket_qua(args.runs)
    args.out.write_text(bang_markdown(kq) + "\n", encoding="utf-8")
    for cohort, nhan in COHORTS:
        print(f"{nhan:<9} {ket_luan(kq, cohort)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
