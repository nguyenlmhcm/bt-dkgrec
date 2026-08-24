"""Negative sampling for BPR, and the place leakage rule 6 becomes enforceable.

BPR needs a triple ``(u, i, j)``: an observed item ``i`` and an unobserved item
``j``. Two properties decide whether the resulting comparison is honest:

**Rule 6 -- negatives come from ``I_train`` only.** Sampling ``j`` from the full
item catalogue would let the model learn about items that do not exist yet at
``T_train``, which is a future leak dressed up as a negative example. The
sampler draws matrix indices in ``[0, n_items)``, and that space *is* ``I_train``
by construction of the mapping (rule 2) -- but "by construction" is exactly the
kind of claim that stops being true after a refactor, so
:func:`~src.guards.leakage.assert_negatives_in_train` is called on real sampled
ids, translated back through the mapping.

**A negative must be genuinely unobserved.** Drawing an item the visitor already
interacted with in train would push the model to rank a true positive *below*
another true positive. Collisions are rejected and redrawn.

Cost of the guard
-----------------
``assert_negatives_in_train`` does a ``setdiff1d`` over the whole item universe.
Running it on all ~24 batches of all ~300 epochs would cost minutes of GPU time
to re-verify an invariant that cannot change within a run. It therefore runs on
the **first batch of every epoch**: a defect in the index space shows up on the
very first check, and every epoch still gets verified. The guard is never
switched off, and the number of checks performed is recorded in the run
artifact so the claim is auditable (docs/DECISIONS.md muc D26).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.guards.leakage import assert_negatives_in_train
from src.utils.logging import get_logger

log = get_logger(__name__)

#: How many times a colliding negative is redrawn before it is accepted as-is.
#: On RetailRocket the interaction matrix has a density around 7e-6, so a
#: collision is already rare; eight rounds make a surviving collision
#: astronomically unlikely without an unbounded loop.
MAX_RESAMPLE_ROUNDS = 8


class NegativeSampler:
    """Draws unobserved items uniformly from ``I_train``.

    Uniform sampling is deliberate. Popularity-biased negative sampling would
    make the loss depend on the same popularity signal the ``popularity``
    baseline already represents, blurring the comparison this thesis is built
    on.

    Attributes:
        n_items: Size of ``I_train``.
        num_negatives: Negatives drawn per positive pair.
        n_collisions: Sampled negatives that turned out to be observed and were
            redrawn -- reported so the rejection loop is not a silent step.
        n_unresolved: Collisions still unresolved after
            :data:`MAX_RESAMPLE_ROUNDS` rounds.
        n_guard_checks: How many times rule 6 was asserted.
    """

    def __init__(
        self,
        seen: sp.csr_matrix,
        item_ids: np.ndarray,
        num_negatives: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            seen: ``visitor x item`` boolean matrix of train interactions.
            item_ids: Raw item ids ordered by matrix index, from the mapping.
                Used to state rule 6 in terms of real identifiers.
            num_negatives: Negatives per positive.
            rng: Seeded generator; the run seed reaches sampling through here.

        Raises:
            ValueError: If ``seen`` and ``item_ids`` disagree on the item count.
        """
        if seen.shape[1] != len(item_ids):
            raise ValueError(
                f"seen co {seen.shape[1]:,} cot nhung mapping co {len(item_ids):,} item"
            )
        self.seen = seen.tocsr()
        self.item_ids = np.asarray(item_ids)
        self.n_items = int(seen.shape[1])
        self.num_negatives = int(num_negatives)
        self.rng = rng

        self._sorted_item_ids = np.sort(self.item_ids)
        self.n_collisions = 0
        self.n_unresolved = 0
        self.n_guard_checks = 0

    def sample(self, visitors: np.ndarray, verify: bool = False) -> np.ndarray:
        """Draw negatives for a batch of visitors.

        Args:
            visitors: Visitor matrix indices, shape ``(n_pairs,)``.
            verify: Assert leakage rule 6 on the drawn negatives. Set on the
                first batch of each epoch by :class:`~src.training.trainer.Trainer`.

        Returns:
            Item matrix indices, shape ``(n_pairs, num_negatives)``.

        Raises:
            LeakageError: If ``verify`` is set and any negative falls outside
                ``I_train``.
        """
        n_pairs = len(visitors)
        negatives = self.rng.integers(0, self.n_items, size=(n_pairs, self.num_negatives))

        repeated = np.repeat(visitors, self.num_negatives)
        for _ in range(MAX_RESAMPLE_ROUNDS):
            flat = negatives.ravel()
            observed = np.asarray(self.seen[repeated, flat]).ravel() > 0
            n_bad = int(observed.sum())
            if n_bad == 0:
                break
            self.n_collisions += n_bad
            flat[observed] = self.rng.integers(0, self.n_items, size=n_bad)
            negatives = flat.reshape(negatives.shape)
        else:
            still_observed = np.asarray(self.seen[repeated, negatives.ravel()]).ravel() > 0
            self.n_unresolved += int(still_observed.sum())

        if verify:
            assert_negatives_in_train(self.item_ids[negatives.ravel()], self._sorted_item_ids)
            self.n_guard_checks += 1
        return negatives

    def describe(self) -> dict[str, object]:
        """Serialisable sampling record for the run artifact."""
        return {
            "strategy": "uniform_over_train_items",
            "num_negatives": self.num_negatives,
            "n_collisions_resampled": int(self.n_collisions),
            "n_collisions_unresolved": int(self.n_unresolved),
            "rule_6_checks": int(self.n_guard_checks),
        }
