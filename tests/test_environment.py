"""Moi run phai mang theo moi truong thuc te da sinh ra con so cua no.

Tren Colab ta co y khong ghim phien ban (DECISIONS.md D25), nen ban ghi nay la
thu duy nhat cho phep tai dung mot ket qua ve sau.
"""

from src.utils.environment import TRACKED, environment_record


def test_record_names_the_packages_that_change_the_numbers():
    record = environment_record()
    assert set(record["packages"]) == set(TRACKED)
    for name in ("numpy", "pandas", "scipy"):
        assert record["packages"][name], f"{name} phai co phien ban that"


def test_record_survives_torch_being_absent():
    """VPS khong cai torch — ham van phai chay, chi bo trong o do."""
    record = environment_record()
    assert record["python"]
    assert record["platform"]
    # torch vang mat duoc ghi la None chu khong lam vo ban ghi
    assert "torch" in record["packages"]


def test_record_is_json_serialisable():
    import json

    json.dumps(environment_record())
