"""Bang 2x2 D35: ket luan phai di theo nguong da chot TRUOC khi do.

Nguong nam trong code (scripts/16_bang_ablation.py::ket_luan). Neu ai sua nguong
sau khi nhin so, cac test duoi day do -- dung muc dich cua viec chot truoc.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bang_ablation", ROOT / "scripts" / "16_bang_ablation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bang_ablation"] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def _runs(tmp_path: Path, recall: dict[str, float], seeds=(2020, 2021, 2022)) -> Path:
    """Tao metrics.json gia: moi model co cung mot recall o moi seed, ca hai cohort."""
    for cohort in ("original", "active"):
        for model, value in recall.items():
            for seed in seeds:
                run = tmp_path / f"{cohort}_{model}_{seed}"
                run.mkdir(parents=True)
                (run / "metrics.json").write_text(json.dumps({
                    "cohort": cohort, "model": model, "seed": seed,
                    "test": {"warm": {"recall@20": value, "ndcg@20": value / 2}},
                }))
    return tmp_path


def _verdict(tmp_path: Path, alpha_only: float, time_only: float) -> str:
    """static = 0.020, de xuat = 0.030 -> ty le = (x - 0.020) / 0.010."""
    kq = mod.doc_ket_qua(_runs(tmp_path, {
        "static_kg_gcn": 0.020, "bt_dkgrec_alpha_only": alpha_only,
        "bt_dkgrec_time_only": time_only, "bt_dkgrec_l05": 0.030,
    }))
    return mod.ket_luan(kq, "original")


@pytest.mark.parametrize(("alpha_only", "time_only", "expected"), [
    (0.021, 0.029, "chủ yếu từ thời gian"),          # alpha 10%, time 90%
    (0.023, 0.0267, "chủ yếu từ thời gian"),         # ngay tren hai bien 1/3 va 2/3
    (0.029, 0.021, "chủ yếu từ loại hành vi"),       # alpha 90%
    (0.025, 0.025, "hai thành phần bổ sung nhau"),   # 50% / 50%
    (0.021, 0.017, "hai thành phần bổ sung nhau"),   # nhu Original that: 10% / -30%
    (0.024, 0.029, "hai thành phần bổ sung nhau"),   # time >= 2/3 nhung alpha > 1/3
    (0.021, 0.026, "hai thành phần bổ sung nhau"),   # time 60%: duoi 2/3 -> chua phai "chu yeu"
    (0.021, 0.0265, "hai thành phần bổ sung nhau"),  # time 65%: sat bien 2/3 tu phia duoi
])
def test_the_verdict_follows_the_pre_registered_thresholds(tmp_path, alpha_only, time_only, expected):
    assert _verdict(tmp_path, alpha_only, time_only) == expected


def test_a_missing_seed_stops_the_table(tmp_path) -> None:
    """Thieu seed thi khong sinh bang -- tranh trung binh tren 2 seed ma khong ai biet."""
    runs = _runs(tmp_path / "a", {"static_kg_gcn": 0.02, "bt_dkgrec_alpha_only": 0.02,
                                  "bt_dkgrec_time_only": 0.02})
    _runs(runs, {"bt_dkgrec_l05": 0.03}, seeds=(2020, 2021))
    with pytest.raises(SystemExit, match="chua du 3 seed"):
        mod.doc_ket_qua(runs)


def test_other_models_in_runs_are_ignored(tmp_path) -> None:
    """lightgcn, popularity... nam chung thu muc nhung khong vao bang 2x2."""
    runs = _runs(tmp_path, {"static_kg_gcn": 0.02, "bt_dkgrec_alpha_only": 0.021,
                            "bt_dkgrec_time_only": 0.029, "bt_dkgrec_l05": 0.03,
                            "lightgcn": 0.5})
    kq = mod.doc_ket_qua(runs)
    assert {model for _, model in kq} == {m for m, *_ in mod.CELLS}
    assert "lightgcn" not in mod.bang_markdown(kq).lower()
