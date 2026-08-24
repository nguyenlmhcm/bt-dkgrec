"""Ablation: the knowledge graph without behavior-time weighting (Buoc 7).

This is the load-bearing comparison of the thesis. ``static_kg_gcn`` and
``bt_dkgrec`` must differ in **exactly one** variable -- the edge weight -- so
that any measured gap is attributable to it and to nothing else.

Where that one variable actually lives
--------------------------------------
Not here. The weighting is chosen in :data:`src.graph.weighting.WEIGHTING_BY_MODEL`
and applied when Buoc 4 builds the adjacency matrix::

    bt_dkgrec      BehaviorTimeWeighting   w = alpha_b * exp(-lambda * dt)   (3.17)
    static_kg_gcn  UniformWeighting        w = 1.0

``UniformWeighting`` itself inherits ``BehaviorTimeWeighting`` and overrides only
``edge_weight()``. So the ablation is a single overridden method, one layer down,
and this class is left with nothing to say but its own name.

That emptiness is the point, and it is asserted rather than trusted:
``tests/test_ablation.py`` parses this file and fails if the class body ever
grows anything besides ``name``. The moment a second difference appears, the
comparison stops being controlled and the central claim loses its evidence.

Not a reimplementation of anything
----------------------------------
``static_kg_gcn`` is **not** KGAT, KGCN, or any published model. It is this
thesis's own model with one component removed. CLAUDE.md requires this to be
stated explicitly in the write-up so a reader never mistakes it for a
reproduction and asks why the numbers differ from a paper.
"""

from __future__ import annotations

from src.models.bt_dkgrec import BTDKGRec


class StaticKGGCN(BTDKGRec):
    """``bt_dkgrec`` with uniform edge weights. Everything else is inherited."""

    name = "static_kg_gcn"
