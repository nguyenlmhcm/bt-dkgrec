"""BT-DKGRec with the decay rate tuned on validation (Buoc 8b).

``bt_dkgrec`` inherited ``lambda_decay = 0.01`` from MBGCN [Jin et al., SIGIR
2020] and KHGT [Xia et al., AAAI 2021] without ever tuning it -- the limitation
recorded in docs/DECISIONS.md muc D30. Measured on the training window, that
value decays an edge only from 1.00 down to 0.38 over 97 days, which is a very
mild recency preference.

A sweep on **validation-period** data (never test) put the optimum at 0.02-0.05
on both cohorts; 0.05 is the joint peak. This class exists so both settings can
appear side by side in the results table instead of one silently replacing the
other: the reader sees the tuning, not just its outcome.

Nothing else differs. The weighting formula, the graph, the propagation, the
loss and the evaluation protocol are the ones ``bt_dkgrec`` already uses --
``lambda_decay`` is a **parameter** of formula (3.17), not a change to it. The
only line that differs between the two configs is the parameter itself, which
``tests/test_ablation.py`` asserts.
"""

from __future__ import annotations

from src.models.bt_dkgrec import BTDKGRec


class BTDKGRecL05(BTDKGRec):
    """The proposed model at the tuned decay rate. Same code, one parameter apart."""

    name = "bt_dkgrec_l05"
