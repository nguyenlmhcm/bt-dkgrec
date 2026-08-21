"""Ghi lai moi truong thuc te da chay mot run.

Tren Colab ta co y KHONG ghim phien ban thu vien (xem ``requirements-colab.txt``
va DECISIONS.md D25), nen tinh truy nguyen khong the dua vao file requirements.
No dua vao day: moi run artifact mang theo ``env.json`` liet ke ban da thuc su
duoc nap vao tien trinh sinh ra con so do.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

# Nhung goi anh huong truc tiep den CON SO, khong phai toan bo moi truong.
TRACKED = ("numpy", "pandas", "scipy", "pyarrow", "pydantic", "torch")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _torch_facts() -> dict[str, object]:
    """Chi tiet GPU — quyet dinh vi sao mot run khong bit-exact voi run khac."""
    if "torch" not in sys.modules and _package_version("torch") is None:
        return {}
    try:
        import torch
    except ImportError:
        return {}

    facts: dict[str, object] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.version.cuda:
        facts["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        facts["gpu"] = torch.cuda.get_device_name(0)
    return facts


def environment_record() -> dict[str, object]:
    """Anh chup moi truong, an toan de goi ca khi torch chua duoc cai."""
    record: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in TRACKED},
    }
    torch_facts = _torch_facts()
    if torch_facts:
        record["torch"] = torch_facts
    return record
