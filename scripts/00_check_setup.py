"""Step 1 completion check: environment, raw data, and config tree are sound.

Run via ``make setup``. Prints a report and exits non-zero on any failure so a
broken environment can never be mistaken for a working one.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import COHORTS, MODELS, load_config, load_seeds  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

REQUIRED_PACKAGES = ["numpy", "pandas", "scipy", "pyarrow", "yaml", "pydantic", "pytest"]
RAW_FILES = [
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
]


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_packages() -> list[str]:
    """Verify every core dependency imports."""
    _section("1. Thu vien loi")
    problems: list[str] = []
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  OK      {name:10s} {getattr(module, '__version__', '?')}")
        except ImportError:
            print(f"  THIEU   {name}")
            problems.append(f"missing package: {name}")
    return problems


def check_raw_data() -> list[str]:
    """Verify the raw dataset is reachable through ``data/raw``."""
    _section("2. Du lieu raw (data/raw)")
    problems: list[str] = []
    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return [f"missing raw data directory: {raw_dir}"]
    for name in RAW_FILES:
        path = raw_dir / name
        if path.exists():
            print(f"  OK      {name:28s} {path.stat().st_size / 1e6:9.1f} MB")
        else:
            print(f"  THIEU   {name}")
            problems.append(f"missing raw file: {name}")
    return problems


def check_configs() -> list[str]:
    """Validate every model x cohort x seed combination of the experiment matrix."""
    _section("3. Ma tran cau hinh (model x cohort x seed)")
    problems: list[str] = []
    seeds = load_seeds()
    print(f"  seeds = {seeds}")
    count = 0
    for cohort in COHORTS:
        for model in MODELS:
            for seed in seeds:
                try:
                    load_config(model=model, cohort=cohort, seed=seed)
                    count += 1
                except Exception as exc:  # noqa: BLE001 - report, do not mask
                    print(f"  LOI     {cohort}/{model}/{seed}: {exc}")
                    problems.append(f"config invalid: {cohort}/{model}/{seed}")
    print(f"  OK      {count}/{len(COHORTS) * len(MODELS) * len(seeds)} cau hinh validate thanh cong")
    return problems


def show_sample_config() -> None:
    """Print one fully resolved config as evidence that merging works."""
    _section("4. Vi du config da hop nhat (bt_dkgrec / original / seed 2020)")
    cfg = load_config(model="bt_dkgrec", cohort="original", seed=2020)
    print(cfg.to_yaml())
    print(f"  run_id mau: {cfg.run_id}")


def main() -> int:
    setup_logging()
    print("=" * 70)
    print("BT-DKGRec-GCN — kiem tra Buoc 1")
    print("=" * 70)

    problems = check_packages() + check_raw_data() + check_configs()
    if not problems:
        show_sample_config()

    _section("KET QUA")
    if problems:
        for item in problems:
            print(f"  FAIL  {item}")
        print(f"\n  {len(problems)} van de — setup CHUA xanh.")
        return 1
    print("  Tat ca kiem tra PASS — Buoc 1 xanh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
