"""BT-DKGRec with uniform behavior weights: time decay only (D35).

The other half of the factorial that separates the two signals in formula
(3.17). ``alpha = 1`` for every behavior leaves ``exp(-lambda * delta)`` as the
whole edge weight. Compared with ``bt_dkgrec_l05`` it differs only in ``alpha``,
and compared with ``static_kg_gcn`` it differs in the decay alone.

Nothing else differs, which ``tests/test_ablation.py`` asserts.
"""

from __future__ import annotations

from src.models.bt_dkgrec import BTDKGRec


class BTDKGRecTimeOnly(BTDKGRec):
    """Time decay without behavior weights. Same code, one parameter apart."""

    name = "bt_dkgrec_time_only"
