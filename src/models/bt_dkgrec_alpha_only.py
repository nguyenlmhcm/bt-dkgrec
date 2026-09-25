"""BT-DKGRec with the time decay switched off: behavior weights only (D35).

One half of the factorial that separates the two signals in formula (3.17).
``lambda_decay = 0`` makes ``exp(-lambda * delta)`` equal 1 for every edge, so
the only thing left in the edge weight is ``alpha_b``. Compared with
``bt_dkgrec_l05`` it differs in that one parameter, and compared with
``static_kg_gcn`` it differs in the behavior weights alone.

Nothing else differs, which ``tests/test_ablation.py`` asserts.
"""

from __future__ import annotations

from src.models.bt_dkgrec import BTDKGRec


class BTDKGRecAlphaOnly(BTDKGRec):
    """Behavior weights without decay. Same code, one parameter apart."""

    name = "bt_dkgrec_alpha_only"
