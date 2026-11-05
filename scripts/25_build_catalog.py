"""Build the demo shop's illustrative catalogue (data/app/catalog.parquet)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shop.catalog import build_catalog  # noqa: E402

catalog = build_catalog(ROOT / "data" / "interim" / "active", ROOT / "data" / "raw")
out = ROOT / "data" / "app" / "catalog.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
catalog.to_parquet(out, index=False)
print(len(catalog), "san pham;", int(catalog.price_is_real.sum()), "gia that;", int((~catalog.available).sum()), "het hang")
print(catalog.department.value_counts().to_string())
print(catalog.sample(8, random_state=1)[["item_id", "name", "price", "price_is_real", "available"]].to_string())
print("gia", catalog.price.describe()[["50%", "max"]].to_dict())
