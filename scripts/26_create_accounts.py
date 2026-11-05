"""Create the four demo accounts. Passwords go to a root-only file, never to the screen."""

import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shop.auth import Accounts  # noqa: E402
from src.shop.catalog import SHOWCASE  # noqa: E402
from src.shop.db import ShopDB  # noqa: E402

SECRETS = Path(os.environ.get("BT_SHOP_ENV", "/etc/bt-shop.env"))
USERS = [("khach1", "customer", SHOWCASE[962057], 962057), ("khach2", "customer", SHOWCASE[481437], 481437),
         ("cuahang", "store", "Quản lý Chợ Hiên", None), ("nhacungcap", "provider", "BT-DKGRec", None)]

accounts = Accounts(ShopDB(ROOT / "data" / "app" / "shop.sqlite"))
text = SECRETS.read_text(encoding="utf-8") if SECRETS.exists() else ""
text = re.sub(r"(?m)^(SELLER_USER|SELLER_PASSWORD|DEMO_PW_\w+)=.*\n?", "", text)
for username, role, display, visitor in USERS:
    password = secrets.token_urlsafe(9)
    accounts.upsert(username, password, role, display, visitor)
    text += f"DEMO_PW_{username}={password}\n"
    print("da tao", username, role)
SECRETS.write_text(text, encoding="utf-8")
SECRETS.chmod(0o600)
