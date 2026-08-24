"""LightGCN -- the published academic baseline (Buoc 8).

He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation*, SIGIR 2020. Plain graph collaborative filtering: user-item
edges only, uniform weights, parameter-free propagation.

Two components are removed relative to ``bt_dkgrec``, and both removals happen
outside this file:

============  ===========================  ==================================
              where it is decided          effect
============  ===========================  ==================================
side info     ``configs/models/lightgcn``   ``use_side_info: false`` -- Buoc 4
              (``use_side_info: false``)    builds a graph with no Category or
                                            PropertyValue block at all
edge weight   ``WEIGHTING_BY_MODEL``        ``UniformWeighting`` -- w = 1.0
============  ===========================  ==================================

So ``lightgcn`` differs from ``bt_dkgrec`` in **two** variables, while
``static_kg_gcn`` differs in one. Reading the three together separates the two
contributions: ``lightgcn -> static_kg_gcn`` isolates side information, and
``static_kg_gcn -> bt_dkgrec`` isolates behavior-time weighting.

Must be trained to convergence
------------------------------
CLAUDE.md rule 9, and the reason it exists: the v11 thesis trained every model
for 10 epochs, while the LightGCN reference implementation trains for 1000.
Beating a baseline that never converged proves nothing, and this is the question
a committee is most likely to ask. The evidence is ``curves.csv`` -- the
validation curve must visibly plateau before early stopping fires. If it is
still climbing at the last epoch, ``max_epochs`` was too low and the run does
not count as a converged baseline.
"""

from __future__ import annotations

from src.models.bt_dkgrec import BTDKGRec


class LightGCN(BTDKGRec):
    """Graph CF baseline. Shares every line of ``bt_dkgrec``; only the graph differs."""

    name = "lightgcn"
