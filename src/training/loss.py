"""Pairwise ranking losses -- formulas (3.29)-(3.31).

Bayesian Personalised Ranking [Rendle et al., UAI 2009] is the loss of the whole
experiment matrix. It suits implicit feedback: RetailRocket records what a
visitor *did*, never what they rejected, so there is no ground truth for
"disliked" -- only the assumption that an observed item is preferred over an
unobserved one::

    (3.29)  s(u,i) = z_u . z_i                     # score, see models
    (3.30)  L_BPR  = -mean ln sigma(s(u,i) - s(u,j))
    (3.31)  L_wBPR = weighted mean by W(u,i)       # ABLATION ONLY

Why (3.31) is an ablation and not the default
---------------------------------------------
Putting ``W(u,i)`` into the loss as well as into the graph would apply the same
behavior-time signal twice, and any gain could no longer be attributed to the
graph. The whole 5-model x 3-seed x 2-cohort matrix therefore runs standard BPR;
weighted BPR is reported as a single extra row of ``bt_dkgrec``. The rule is
enforced in :class:`~src.utils.config.Config` (``_weighted_bpr_is_ablation_only``),
not left to discipline.

Numerical note: ``-ln sigma(x)`` is computed as ``softplus(-x)``. The naive form
overflows to ``inf`` once the score gap exceeds about 88 in float32, which is
reachable early in training when embeddings are still unconstrained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Formula (3.30): standard BPR over one batch of triples.

    Args:
        pos_scores: ``s(u,i)`` for the observed item, shape ``(n_triples,)``.
        neg_scores: ``s(u,j)`` for the sampled item, same shape.

    Returns:
        Scalar mean loss.

    Raises:
        ValueError: If the two score tensors have different shapes, which would
            silently broadcast into a loss over the wrong pairs.
    """
    _assert_same_shape(pos_scores, neg_scores)
    return F.softplus(neg_scores - pos_scores).mean()


def weighted_bpr_loss(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Formula (3.31): BPR whose terms are weighted by ``W(u,i)``. Ablation only.

    Normalised by ``sum(weights)`` rather than by the batch size, so the result
    stays on the same scale as :func:`bpr_loss` and the two learning rates
    remain comparable.

    Args:
        pos_scores: ``s(u,i)``, shape ``(n_triples,)``.
        neg_scores: ``s(u,j)``, same shape.
        weights: Aggregated edge weight ``W(u,i)`` of the positive pair.

    Returns:
        Scalar weighted mean loss.

    Raises:
        ValueError: On a shape mismatch, or if the weights sum to zero.
    """
    _assert_same_shape(pos_scores, neg_scores)
    _assert_same_shape(pos_scores, weights)
    total = weights.sum()
    if float(total) <= 0:
        raise ValueError("tong trong so cua batch bang 0 — khong the chuan hoa loss")
    return (weights * F.softplus(neg_scores - pos_scores)).sum() / total


def l2_regularization(*embeddings: torch.Tensor) -> torch.Tensor:
    """Mean squared L2 norm of the layer-0 embeddings touched by a batch.

    Only the embeddings actually involved in the batch are penalised, in the
    LightGCN sense: regularising the whole table every step would shrink the
    embeddings of entities the batch never saw and make the penalty depend on
    the number of nodes rather than on the batch.

    Args:
        *embeddings: Tensors of shape ``(n_triples, dim)``.

    Returns:
        Scalar ``sum(||e||^2) / (2 * n_triples)``.
    """
    if not embeddings:
        raise ValueError("l2_regularization can it nhat mot tensor")
    n_triples = embeddings[0].shape[0]
    if n_triples == 0:
        return embeddings[0].sum() * 0.0
    total = sum(tensor.pow(2).sum() for tensor in embeddings)
    return total / (2 * n_triples)


def _assert_same_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(f"shape lech nhau: {tuple(left.shape)} vs {tuple(right.shape)}")


#: Loss functions selectable from ``cfg.training.loss``.
LOSS_BY_NAME = {"bpr": bpr_loss, "weighted_bpr": weighted_bpr_loss}
