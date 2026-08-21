"""Symmetric normalisation of the weighted adjacency matrix.

Formula (3.24)-(3.26) of the thesis, inherited from Kipf & Welling (ICLR 2017)
and kept parameter-free in the LightGCN sense (He et al., SIGIR 2020)::

    d_v   = sum over neighbours j of omega(v, j)      # weighted degree
    A_hat = D^(-1/2) . A . D^(-1/2)

Normalising by the weighted degree of *both* endpoints stops high-degree nodes
-- a popular item, or a PropertyValue shared by thousands of items -- from
dominating propagation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.utils.logging import get_logger

log = get_logger(__name__)


def weighted_degree(adjacency: sp.spmatrix) -> np.ndarray:
    """Weighted degree ``d_v`` of every node."""
    return np.asarray(adjacency.sum(axis=1)).ravel()


def symmetric_normalize(adjacency: sp.spmatrix, dtype: str = "float32") -> sp.csr_matrix:
    """Return ``A_hat = D^(-1/2) . A . D^(-1/2)``.

    Isolated nodes (degree 0) would make ``d^(-1/2)`` infinite. Their scaling
    factor is set to 0 instead: an isolated node has no neighbour to receive
    anything from, so its propagated value is legitimately zero and the matrix
    stays finite. The count of such nodes is logged -- on a correctly built
    graph it must be 0 (KG_DESIGN.md muc 8).

    Args:
        adjacency: Symmetric weighted adjacency matrix.
        dtype: Storage dtype of the result.

    Raises:
        ValueError: If the matrix is not square or holds a negative weight.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency phai vuong, dang co {adjacency.shape}")

    matrix = adjacency.tocsr()
    if matrix.nnz and matrix.data.min() < 0:
        raise ValueError(f"trong so am trong ma tran ke: min={matrix.data.min()}")

    degree = weighted_degree(matrix)
    n_isolated = int((degree == 0).sum())
    if n_isolated:
        log.warning("co %s node bac 0 — he so chuan hoa dat bang 0", f"{n_isolated:,}")

    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    scaler = sp.diags(d_inv_sqrt.astype("float64"))
    normalized = (scaler @ matrix @ scaler).tocsr().astype(dtype)
    normalized.eliminate_zeros()

    log.info(
        "chuan hoa doi xung: %s node, %s phan tu khac 0, gia tri in [%.6g, %.6g]",
        f"{normalized.shape[0]:,}", f"{normalized.nnz:,}",
        normalized.data.min() if normalized.nnz else 0.0,
        normalized.data.max() if normalized.nnz else 0.0,
    )
    return normalized


def assert_symmetric(adjacency: sp.spmatrix, tolerance: float = 1e-6) -> None:
    """Raise if the adjacency matrix is not symmetric.

    Propagation assumes an undirected graph; an asymmetric matrix would send a
    signal one way only and silently change the model.
    """
    difference = (adjacency - adjacency.T)
    max_gap = float(abs(difference).max()) if difference.nnz else 0.0
    if max_gap > tolerance:
        raise ValueError(f"ma tran ke khong doi xung: lech toi da {max_gap}")
