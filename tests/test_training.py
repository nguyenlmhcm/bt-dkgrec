"""Tests for the training layer (Buoc 6): losses and negative sampling.

The load-bearing test here is
:func:`test_sampler_never_returns_an_item_the_visitor_already_saw`. A negative
that is really a positive teaches the model to rank a true target *below*
another true target, and the damage is invisible: the loss still falls, the
curves still look healthy, and every reported number is quietly wrong.
"""

from __future__ import annotations

import math

import pytest

# Torch khong nam trong requirements.txt: VPS chi sua code va chay test, khong
# train (D28). Nhung bai test duoi day chay duong huan luyen that, nen chung
# chay tren Colab — noi co torch ban CUDA — va tu bo qua o may khong co torch.
# Bo qua KHAC voi bo sot: o 15 cua notebook la noi chung bat buoc phai xanh.
pytest.importorskip("torch", reason="can torch — chay tren Colab (notebook o 15)")

import numpy as np
import scipy.sparse as sp
import torch

from src.guards.leakage import LeakageError, assert_negatives_in_train
from src.training.loss import (
    LOSS_BY_NAME,
    bpr_loss,
    l2_regularization,
    weighted_bpr_loss,
)
from src.training.sampler import MAX_RESAMPLE_ROUNDS, NegativeSampler


# ══ Formula (3.30) — BPR ═════════════════════════════════════════════════


def test_bpr_is_ln2_when_the_model_cannot_tell_the_pair_apart() -> None:
    """Equal scores mean a coin flip: -ln sigma(0) = ln 2."""
    scores = torch.zeros(4)
    assert bpr_loss(scores, scores).item() == pytest.approx(math.log(2), abs=1e-6)


def test_bpr_falls_as_the_positive_pulls_ahead() -> None:
    negative = torch.zeros(3)
    losses = [bpr_loss(torch.full((3,), gap), negative).item() for gap in (0.0, 1.0, 5.0)]
    assert losses[0] > losses[1] > losses[2] > 0


def test_bpr_stays_finite_on_a_score_gap_that_would_overflow_float32() -> None:
    """The naive -ln(sigmoid(x)) form overflows past a gap of about 88."""
    loss = bpr_loss(torch.tensor([-500.0]), torch.tensor([500.0]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1000.0, rel=1e-4)


def test_bpr_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape lech nhau"):
        bpr_loss(torch.zeros(4), torch.zeros(3))


# ══ Formula (3.31) — weighted BPR, ablation only ═════════════════════════


def test_weighted_bpr_equals_plain_bpr_when_every_weight_is_equal() -> None:
    pos, neg = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 0.5, 0.5])
    weighted = weighted_bpr_loss(pos, neg, torch.ones(3))
    assert weighted.item() == pytest.approx(bpr_loss(pos, neg).item(), abs=1e-6)


def test_weighted_bpr_follows_the_heavier_edge() -> None:
    """Raising W(u,i) on the badly-ranked pair must raise the loss."""
    pos, neg = torch.tensor([5.0, -5.0]), torch.tensor([0.0, 0.0])
    light = weighted_bpr_loss(pos, neg, torch.tensor([10.0, 1.0]))
    heavy = weighted_bpr_loss(pos, neg, torch.tensor([1.0, 10.0]))
    assert heavy.item() > light.item()


def test_weighted_bpr_refuses_a_batch_whose_weights_sum_to_zero() -> None:
    with pytest.raises(ValueError, match="tong trong so"):
        weighted_bpr_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2))


def test_both_losses_are_reachable_from_the_config_name() -> None:
    assert set(LOSS_BY_NAME) == {"bpr", "weighted_bpr"}


# ══ L2 regularisation ════════════════════════════════════════════════════


def test_l2_penalises_only_the_embeddings_in_the_batch() -> None:
    batch = torch.ones(4, 3)          # 4 rows, ||e||^2 = 3 each
    assert l2_regularization(batch).item() == pytest.approx(12.0 / 8.0)


def test_l2_sums_across_the_three_roles_of_a_triple() -> None:
    single = l2_regularization(torch.ones(2, 3))
    triple = l2_regularization(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3))
    assert triple.item() == pytest.approx(3 * single.item())


# ══ Negative sampling and leakage rule 6 ═════════════════════════════════


@pytest.fixture
def seen() -> sp.csr_matrix:
    """5 visitors x 8 items; visitor v has interacted with items [v, v+1]."""
    rows = np.repeat(np.arange(5), 2)
    cols = np.concatenate([[v, v + 1] for v in range(5)])
    return sp.csr_matrix((np.ones(len(rows), dtype="bool"), (rows, cols)), shape=(5, 8))


def _sampler(seen: sp.csr_matrix, num_negatives: int = 1, seed: int = 0) -> NegativeSampler:
    return NegativeSampler(
        seen=seen,
        item_ids=np.arange(1000, 1000 + seen.shape[1]),
        num_negatives=num_negatives,
        rng=np.random.default_rng(seed),
    )


def test_sampler_never_returns_an_item_the_visitor_already_saw(seen) -> None:
    """★ A false negative is a silent correctness bug — assert it cannot happen."""
    sampler = _sampler(seen)
    visitors = np.repeat(np.arange(5), 200)
    negatives = sampler.sample(visitors)
    observed = np.asarray(seen[visitors, negatives.ravel()]).ravel()
    assert not observed.any()
    assert sampler.n_unresolved == 0


def test_sampler_returns_one_column_per_requested_negative(seen) -> None:
    negatives = _sampler(seen, num_negatives=3).sample(np.arange(5))
    assert negatives.shape == (5, 3)


def test_sampler_draws_only_matrix_indices_inside_i_train(seen) -> None:
    negatives = _sampler(seen).sample(np.repeat(np.arange(5), 100))
    assert negatives.min() >= 0
    assert negatives.max() < seen.shape[1]


def test_sampler_records_the_collisions_it_had_to_redraw(seen) -> None:
    sampler = _sampler(seen)
    sampler.sample(np.repeat(np.arange(5), 400))
    # 2 of 8 items collide per visitor, so a quarter of the first draw is redrawn.
    assert sampler.n_collisions > 0
    assert sampler.describe()["n_collisions_resampled"] == sampler.n_collisions


def test_sampler_gives_up_visibly_when_every_item_is_already_seen() -> None:
    """A visitor with no unseen item cannot get an honest negative.

    The sampler must record that rather than loop forever or pretend it found
    one -- ``n_collisions_unresolved`` is what makes the situation auditable.
    """
    full = sp.csr_matrix(np.ones((1, 4), dtype="bool"))
    sampler = _sampler(full)
    sampler.sample(np.zeros(10, dtype="int64"))
    assert sampler.n_unresolved == 10
    assert sampler.n_collisions == 10 * MAX_RESAMPLE_ROUNDS


def test_rule_6_is_asserted_only_when_asked(seen) -> None:
    sampler = _sampler(seen)
    sampler.sample(np.arange(5), verify=False)
    assert sampler.n_guard_checks == 0
    sampler.sample(np.arange(5), verify=True)
    assert sampler.n_guard_checks == 1
    assert sampler.describe()["rule_6_checks"] == 1


def test_rule_6_catches_a_negative_from_outside_i_train() -> None:
    """The guard itself, on the defect it exists to catch."""
    with pytest.raises(LeakageError) as caught:
        assert_negatives_in_train(np.array([10, 11, 999]), np.array([10, 11, 12]))
    assert caught.value.rule == "6"
    assert caught.value.kind == "identity"


def test_sampler_refuses_a_mapping_that_disagrees_with_the_seen_matrix(seen) -> None:
    with pytest.raises(ValueError, match="mapping co"):
        NegativeSampler(
            seen=seen,
            item_ids=np.arange(3),
            num_negatives=1,
            rng=np.random.default_rng(0),
        )


def test_sampling_is_reproducible_from_the_seed(seen) -> None:
    first = _sampler(seen, seed=2020).sample(np.arange(5))
    second = _sampler(seen, seed=2020).sample(np.arange(5))
    assert np.array_equal(first, second)
