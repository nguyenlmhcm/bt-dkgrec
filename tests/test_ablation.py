"""The ablation is controlled — asserted, not trusted (Buoc 7, Buoc 8).

CLAUDE.md states the central requirement of the thesis in prose:

    Lap luan cot loi cua luan van nam o cap `static_kg_gcn` vs `bt_dkgrec`.
    Hai mo hinh phai khac DUNG MOT BIEN: ham `edge_weight()`.

Prose cannot fail a build. This module turns it into assertions that run on
every commit, and they are deliberately written so that they hold **without
torch**: they read source text, config, and the graph artefacts on disk rather
than importing the models. That matters because the VPS has no torch (D28), so
these are the ablation checks that still run there -- the place where the code
is actually edited and where the invariant is most likely to be broken.

Three independent levels are checked, because a second difference could sneak in
at any of them:

======  =====================================  =========================
level   what is compared                       fails if
======  =====================================  =========================
source  the class bodies                       a subclass grows a method
config  the resolved YAML for each model       a key other than the
                                               intended ones differs
data    the built adjacency matrices           the sparsity patterns
                                               stop matching
======  =====================================  =========================
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.graph.weighting import BehaviorTimeWeighting, UniformWeighting, WEIGHTING_BY_MODEL
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "models"

#: Names Python puts in every class body; not a difference anyone wrote.
DUNDERS = {"__module__", "__qualname__", "__doc__", "__firstlineno__", "__static_attributes__"}


def _class_body(path: Path, class_name: str) -> ast.ClassDef:
    """Parse one class out of a source file without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"khong tim thay class {class_name} trong {path.name}")


# ══ Level 1 — source ═════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("filename", "class_name", "model_name"),
    [
        ("static_kg_gcn.py", "StaticKGGCN", "static_kg_gcn"),
        ("lightgcn.py", "LightGCN", "lightgcn"),
    ],
)
def test_the_subclass_adds_nothing_but_a_name(filename, class_name, model_name) -> None:
    """★ The executable form of "git diff khac dung mot bien".

    A subclass that grows a method has stopped being the same model with one
    component removed, and the comparison stops being controlled.
    """
    node = _class_body(MODELS_DIR / filename, class_name)

    assert [base.id for base in node.bases] == ["BTDKGRec"]  # type: ignore[attr-defined]

    statements = [s for s in node.body if not _is_docstring(s)]
    assert len(statements) == 1, (
        f"{class_name} co {len(statements)} lenh ngoai docstring — "
        "chi duoc phep dung mot dong `name = ...`"
    )
    assignment = statements[0]
    assert isinstance(assignment, ast.Assign)
    assert [t.id for t in assignment.targets] == ["name"]  # type: ignore[attr-defined]
    assert assignment.value.value == model_name  # type: ignore[attr-defined]


def _is_docstring(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def test_the_parent_never_hardcodes_its_own_name() -> None:
    """A literal ``"bt_dkgrec"`` anywhere else would not inherit correctly.

    Every other reference must go through ``self.name``, or the subclasses would
    silently load the parent's graph, read the parent's stats, and report the
    parent's identity -- while looking perfectly healthy.
    """
    source = (MODELS_DIR / "bt_dkgrec.py").read_text(encoding="utf-8")
    occurrences = source.count('"bt_dkgrec"')
    assert occurrences == 1, (
        f'chuoi "bt_dkgrec" xuat hien {occurrences} lan trong bt_dkgrec.py; '
        "chi duoc phep mot lan, o dong `name = \"bt_dkgrec\"`"
    )
    assert 'name = "bt_dkgrec"' in source


# ══ Level 2 — config ═════════════════════════════════════════════════════


def _model_section(model: str) -> dict:
    return json.loads(load_config(model=model, cohort="original", seed=2020).model_dump_json())


def _differing_keys(left: dict, right: dict) -> set[str]:
    return {k for k in set(left) | set(right) if left.get(k) != right.get(k)}


def test_the_ablation_pair_differs_in_no_config_key_at_all() -> None:
    """★ static_kg_gcn vs bt_dkgrec: identical everywhere except the name.

    Not embedding_dim, not num_layers, not learning rate, not the graph limits.
    If any of those differed, the gap between the two would no longer be
    attributable to the edge weight alone.
    """
    dynamic = _model_section("bt_dkgrec")
    static = _model_section("static_kg_gcn")

    assert _differing_keys(dynamic, static) == {"model"}
    assert _differing_keys(dynamic["model"], static["model"]) == {"name"}


def test_lightgcn_differs_from_the_proposed_model_in_exactly_two_things() -> None:
    """lightgcn removes side info as well, so the chain separates two effects.

    lightgcn -> static_kg_gcn isolates side information;
    static_kg_gcn -> bt_dkgrec isolates behavior-time weighting.
    """
    light = _model_section("lightgcn")
    dynamic = _model_section("bt_dkgrec")

    assert _differing_keys(light, dynamic) == {"model"}
    assert _differing_keys(light["model"], dynamic["model"]) == {"name", "use_side_info"}
    assert light["model"]["use_side_info"] is False


def test_the_weighting_of_each_model_is_decided_in_code_not_yaml() -> None:
    """docs/DECISIONS.md muc D4: no config switch can disable the contribution."""
    assert WEIGHTING_BY_MODEL["bt_dkgrec"] is BehaviorTimeWeighting
    assert WEIGHTING_BY_MODEL["static_kg_gcn"] is UniformWeighting
    assert WEIGHTING_BY_MODEL["lightgcn"] is UniformWeighting


# ══ Level 3 — the built graphs on disk ═══════════════════════════════════


def _graph_dir(cohort: str, model: str) -> Path:
    return REPO_ROOT / "data" / "processed" / cohort / model


def _built(cohort: str, *models: str) -> bool:
    return all((_graph_dir(cohort, m) / "adjacency.npz").exists() for m in models)


@pytest.mark.parametrize("cohort", ["original", "active"])
def test_the_two_kg_graphs_share_a_sparsity_pattern_and_differ_only_in_values(cohort) -> None:
    """★ The ablation, verified on the real artefacts rather than a fixture.

    Same nodes, same edges, same positions in the matrix -- different numbers.
    That is what "one variable" looks like once it reaches the data.
    """
    if not _built(cohort, "bt_dkgrec", "static_kg_gcn"):
        pytest.skip(f"chua dung graph cho cohort {cohort} — chay `make graph COHORT={cohort}`")

    dynamic = sp.load_npz(_graph_dir(cohort, "bt_dkgrec") / "adjacency.npz").tocsr()
    static = sp.load_npz(_graph_dir(cohort, "static_kg_gcn") / "adjacency.npz").tocsr()

    assert dynamic.shape == static.shape
    assert np.array_equal(dynamic.indptr, static.indptr)
    assert np.array_equal(dynamic.indices, static.indices)
    assert not np.allclose(dynamic.data, static.data)


@pytest.mark.parametrize("cohort", ["original", "active"])
def test_the_lightgcn_graph_really_has_no_side_information(cohort) -> None:
    """Otherwise 'graph CF without side info' would be a claim, not a fact."""
    if not _built(cohort, "lightgcn", "bt_dkgrec"):
        pytest.skip(f"chua dung graph cho cohort {cohort}")

    light = json.loads((_graph_dir(cohort, "lightgcn") / "graph_stats.json").read_text())
    full = json.loads((_graph_dir(cohort, "bt_dkgrec") / "graph_stats.json").read_text())

    assert light["n_category"] == 0 and light["n_property_value"] == 0
    assert set(light["edges"]) == {"interacted_with"}
    assert full["n_category"] > 0 and full["n_property_value"] > 0
    # Same users and items on both sides — only the extra node types are gone.
    assert light["n_visitor"] == full["n_visitor"]
    assert light["n_item"] == full["n_item"]


@pytest.mark.parametrize("cohort", ["original", "active"])
def test_uniform_weighting_really_produced_weight_one_per_event(cohort) -> None:
    """static_kg_gcn and lightgcn must show W(u,i) = number of events, exactly."""
    if not _built(cohort, "static_kg_gcn"):
        pytest.skip(f"chua dung graph cho cohort {cohort}")

    stats = json.loads((_graph_dir(cohort, "static_kg_gcn") / "graph_stats.json").read_text())
    assert stats["weighting"] == "uniform"
    assert stats["interaction_weight_min"] == 1.0
    assert stats["interaction_weight_max"] == float(stats["max_events_per_pair"])
