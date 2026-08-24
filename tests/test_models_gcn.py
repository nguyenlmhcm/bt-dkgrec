"""Tests for the graph model and its training loop (Buoc 6).

Two of these tests defend claims the thesis will have to make out loud:

* :func:`test_only_layer_zero_holds_learnable_parameters` -- the propagation is
  parameter-free, so a gain over ``lightgcn`` cannot be explained by extra
  network capacity.
* :func:`test_early_stopping_restores_the_epoch_that_validated_best` -- the run
  reports the epoch chosen on *validation*, which is the fix for the v11 habit
  of training every model for a fixed 10 epochs.

The graph is synthetic but goes through the real builder, so the model is fed
exactly the artefact ``scripts/02_build_graph.py`` produces.
"""

from __future__ import annotations

import json

import pytest

# Torch khong nam trong requirements.txt: VPS chi sua code va chay test, khong
# train (D28). Nhung bai test duoi day chay duong huan luyen that, nen chung
# chay tren Colab — noi co torch ban CUDA — va tu bo qua o may khong co torch.
# Bo qua KHAC voi bo sot: o 15 cua notebook la noi chung bat buoc phai xanh.
pytest.importorskip("torch", reason="can torch — chay tren Colab (notebook o 15)")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.data.mapping import IdMapping
from src.graph.builder import build_graph, save_graph
from src.graph.normalize import symmetric_normalize
from src.graph.weighting import weighting_for_model
from src.models.base import ModelContext, NotFittedError
from src.models.bt_dkgrec import BTDKGRec
from src.utils.config import load_config

DAY = 86_400_000
T_TRAIN = 200 * DAY
BEHAVIORS = ("view", "addtocart", "transaction")
N_VISITORS = 60
N_ITEMS = 40
N_BLOCKS = 4
ITEMS_PER_VISITOR = 5


def _synthetic_interim(directory) -> None:
    """Write a complete `data/interim/<cohort>/` with learnable block structure.

    Visitor ``v`` interacts only with items of block ``v % 4``, deterministically
    chosen so every item is covered. A model that learns anything at all must
    score same-block items above the rest, which is what makes "the loss fell"
    a meaningful assertion rather than a tautology.
    """
    visitors, items, stamps, behaviors = [], [], [], []
    block_size = N_ITEMS // N_BLOCKS
    for visitor in range(N_VISITORS):
        block = visitor % N_BLOCKS
        for step in range(ITEMS_PER_VISITOR):
            offset = (visitor // N_BLOCKS + step) % block_size
            visitors.append(visitor)
            items.append(block * block_size + offset)
            stamps.append(T_TRAIN - (visitor * 3 + step) * DAY)
            behaviors.append(BEHAVIORS[(visitor + step) % len(BEHAVIORS)])

    pd.DataFrame(
        {
            "timestamp": np.array(stamps, dtype="int64"),
            "visitorid": np.array(visitors, dtype="int32"),
            "itemid": np.array(items, dtype="int32"),
            "behavior": pd.Categorical(behaviors, categories=list(BEHAVIORS)),
            "split": pd.Categorical(
                ["train"] * len(visitors), categories=["train", "valid", "test"]
            ),
        }
    ).to_parquet(directory / "events.parquet", index=False)

    pd.DataFrame(
        {"visitor_id": np.arange(N_VISITORS), "idx": np.arange(N_VISITORS)}
    ).to_parquet(directory / "visitors.parquet", index=False)
    pd.DataFrame({"item_id": np.arange(N_ITEMS), "idx": np.arange(N_ITEMS)}).to_parquet(
        directory / "items.parquet", index=False
    )

    empty = {
        "side_item_category.parquet": ["item_idx", "category_id", "category_idx", "valid_from"],
        "side_item_property.parquet": ["item_idx", "pv_idx", "valid_from"],
        "side_property_values.parquet": ["pv_idx", "prop_key", "prop_value", "freq", "pv_id"],
        "side_categories.parquet": ["category_id", "idx", "depth", "is_root"],
        "side_category_parent.parquet": ["category_idx", "parent_idx"],
    }
    for filename, columns in empty.items():
        frame = pd.DataFrame({c: np.array([], dtype="int64") for c in columns})
        frame.to_parquet(directory / filename, index=False)

    (directory / "split.json").write_text(
        json.dumps(
            {"t_min": 0, "t_max": 220 * DAY, "t_train": T_TRAIN,
             "t_valid_end": 210 * DAY, "boundary_inclusive": True}
        ),
        encoding="utf-8",
    )


def _config(**training):
    """Config for a fast but honest run: real YAML, only the loop size overridden."""
    overrides = {
        "training": {
            "device": "cpu", "batch_size": 128, "eval_every": 5, "patience": 20, **training
        }
    }
    return load_config(model="bt_dkgrec", cohort="original", seed=2020, overrides=overrides)


@pytest.fixture
def setup(tmp_path):
    """A built graph plus the ModelContext the training script would pass."""
    interim = tmp_path / "interim"
    interim.mkdir()
    _synthetic_interim(interim)

    cfg = _config()
    graph, edges = build_graph(cfg, interim, weighting_for_model(cfg))
    graph_dir = tmp_path / "graph"
    save_graph(graph, edges, graph_dir)

    mapping = IdMapping.load(interim)
    events = pd.read_parquet(interim / "events.parquet")
    events["visitor_idx"] = mapping.visitor_index(events["visitorid"])
    events["item_idx"] = mapping.item_index(events["itemid"])

    def make_context(cfg_override=None):
        return ModelContext(
            cfg=cfg_override or cfg,
            mapping=mapping,
            train_events=events[["visitor_idx", "item_idx", "behavior", "timestamp"]],
            interaction_edges=edges,
            t_train=T_TRAIN,
            graph_dir=graph_dir,
        )

    return make_context, graph, graph_dir


# ══ Structure of the model ═══════════════════════════════════════════════


def test_only_layer_zero_holds_learnable_parameters(setup) -> None:
    """★ Propagation is parameter-free — there is no weight matrix to explain a gain."""
    make_context, graph, _ = setup
    model = BTDKGRec()
    model._prepare(make_context())

    parameters = list(model.embeddings.parameters())
    assert len(parameters) == 1
    assert tuple(parameters[0].shape) == (graph.node_space.total, 32)


def test_propagation_is_the_mean_over_all_layers(setup) -> None:
    """Formulas (3.25)-(3.26), checked against scipy rather than against itself."""
    make_context, graph, _ = setup
    model = BTDKGRec()
    model._prepare(make_context())

    a_hat = symmetric_normalize(graph.adjacency).astype("float64")
    h0 = model.embeddings.weight.detach().numpy().astype("float64")
    expected = (h0 + a_hat @ h0 + a_hat @ (a_hat @ h0)) / 3.0

    produced = model.propagate().detach().numpy()
    assert produced.shape == expected.shape
    np.testing.assert_allclose(produced, expected, rtol=1e-4, atol=1e-5)


def test_score_returns_one_column_per_train_item(setup) -> None:
    make_context, _, _ = setup
    model = BTDKGRec()
    model._prepare(make_context())
    model.refresh_embeddings()

    scores = model.score(np.array([0, 1, 2]))
    assert scores.shape == (3, N_ITEMS)
    assert np.isfinite(scores).all()


def test_score_refuses_a_cold_visitor_instead_of_inventing_one(setup) -> None:
    """A visitor with no row in E has no embedding; a number here would be fiction."""
    make_context, _, _ = setup
    model = BTDKGRec()
    model._prepare(make_context())
    model.refresh_embeddings()

    with pytest.raises(ValueError, match="khong phuc vu visitor cold"):
        model.score(np.array([0, -1]))


def test_score_before_fit_raises_rather_than_returning_zeros(setup) -> None:
    with pytest.raises(NotFittedError):
        BTDKGRec().score(np.array([0]))


# ══ Loading the right graph ══════════════════════════════════════════════


def test_a_graph_built_for_another_model_is_refused(setup, tmp_path) -> None:
    """Training bt_dkgrec on the static_kg_gcn graph would silently erase the ablation."""
    make_context, _, graph_dir = setup
    stats = json.loads((graph_dir / "graph_stats.json").read_text(encoding="utf-8"))
    stats["model"] = "static_kg_gcn"
    (graph_dir / "graph_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="duoc dung cho"):
        BTDKGRec()._prepare(make_context())


def test_a_missing_graph_names_the_step_that_builds_it(setup, tmp_path) -> None:
    make_context, _, _ = setup
    context = ModelContext(
        cfg=make_context().cfg,
        mapping=make_context().mapping,
        train_events=make_context().train_events,
        interaction_edges=make_context().interaction_edges,
        t_train=T_TRAIN,
        graph_dir=tmp_path / "khong-ton-tai",
    )
    with pytest.raises(FileNotFoundError, match="make graph"):
        BTDKGRec()._prepare(context)


# ══ Training ═════════════════════════════════════════════════════════════


def test_fit_lowers_the_bpr_loss(setup) -> None:
    """★ Buoc 6 completion criterion: train runs and the loss actually falls."""
    make_context, _, _ = setup
    torch.manual_seed(2020)
    model = BTDKGRec()
    model.fit(make_context(_config(max_epochs=300)))

    result = model.training_result
    assert result is not None
    assert result.losses[0] > result.losses[-1]
    assert result.losses[-1] < 0.5 * result.losses[0]


def _block_preference(model: BTDKGRec) -> float:
    """Share of visitors who score their own item block above every other block."""
    block_size = N_ITEMS // N_BLOCKS
    visitors = np.arange(N_VISITORS)
    scores = model.score(visitors)
    own_block = np.zeros(N_ITEMS, dtype=bool)

    better = 0
    for visitor in visitors:
        own_block[:] = False
        start = (visitor % N_BLOCKS) * block_size
        own_block[start : start + block_size] = True
        better += scores[visitor][own_block].mean() > scores[visitor][~own_block].mean()
    return better / N_VISITORS


def test_fitting_ranks_better_than_propagation_over_random_embeddings(setup) -> None:
    """The honest form of "the model learned something".

    A two-layer propagation over a block-structured graph already separates the
    blocks even with *random* layer-0 embeddings -- ``A_hat^2`` carries the
    community structure regardless of what it multiplies. So "the fitted model
    prefers the right block" on its own proves nothing about training. What has
    to be shown is that fitting improves on the untrained propagation, measured
    from the same initialisation.
    """
    make_context, _, _ = setup
    context = make_context(_config(max_epochs=300))

    torch.manual_seed(2020)
    untrained = BTDKGRec()
    untrained._prepare(context)
    untrained.refresh_embeddings()
    before = _block_preference(untrained)

    torch.manual_seed(2020)          # same draw, so only training differs
    model = BTDKGRec()
    model.fit(context)
    after = _block_preference(model)

    assert after > before
    assert after >= 0.95


def test_training_is_kept_when_no_validation_ever_chose_an_epoch(setup) -> None:
    """Regression: restoring the epoch-0 snapshot would erase the whole run.

    Without a validation callback nothing is ever recorded as "best", so an
    unconditional restore hands back the random initialisation -- behind a loss
    curve that still looks perfectly converged.
    """
    make_context, _, _ = setup
    context = make_context(_config(max_epochs=300))

    torch.manual_seed(2020)
    model = BTDKGRec()
    model._prepare(context)
    initial = model.embeddings.weight.detach().clone()

    torch.manual_seed(2020)          # fit() re-initialises from the same draw
    model.fit(context)

    assert not torch.allclose(model.embeddings.weight.detach(), initial)
    assert model.training_result.best_epoch == 300


def test_curves_carry_one_row_per_evaluation(setup) -> None:
    """curves.csv is the evidence of convergence CLAUDE.md requires."""
    make_context, _, _ = setup
    model = BTDKGRec()
    model.fit(make_context(_config(max_epochs=300)))

    curves = model.training_result.curves
    assert list(curves.columns) == ["epoch", "loss", "valid_ndcg@20", "note"]
    assert len(curves) == 300 // 5
    assert curves["epoch"].iloc[0] == 5 and curves["epoch"].iloc[-1] == 300
    assert curves["loss"].notna().all()


def test_rule_6_is_asserted_once_per_epoch(setup) -> None:
    """Leakage rule 6 becomes real here, and the count is auditable."""
    make_context, _, _ = setup
    model = BTDKGRec()
    model.fit(make_context(_config(max_epochs=300)))

    facts = model.describe()["negative_sampling"]
    assert facts["rule_6_checks"] == model.training_result.n_epochs
    assert facts["strategy"] == "uniform_over_train_items"


def test_early_stopping_restores_the_epoch_that_validated_best(setup) -> None:
    """★ The v11 fix: the reported model is the one validation chose, not the last."""
    make_context, _, _ = setup
    sequence = iter([0.10, 0.30, 0.20, 0.15, 0.05] + [0.01] * 50)

    model = BTDKGRec()
    model.attach_validation(lambda _: next(sequence))
    model.fit(make_context(_config(max_epochs=300, eval_every=1, patience=3)))

    result = model.training_result
    assert result.stopped_early
    assert result.best_epoch == 2          # the 0.30 evaluation
    assert result.best_value == pytest.approx(0.30)
    assert result.n_epochs == 5            # 3 evaluations without a gain
    assert result.describe()["monitor"] == "ndcg@20"


def test_training_without_a_validation_hook_still_produces_curves(setup) -> None:
    """A run with no validation must not silently look like a selected one."""
    make_context, _, _ = setup
    model = BTDKGRec()
    model.fit(make_context(_config(max_epochs=300)))

    result = model.training_result
    assert result.best_value is None
    assert not result.stopped_early
    assert result.n_epochs == 300
    assert result.best_epoch == 300      # the last epoch, not the untrained one


def test_fit_refuses_an_empty_edge_table(setup) -> None:
    make_context, _, graph_dir = setup
    context = make_context()
    empty = ModelContext(
        cfg=context.cfg,
        mapping=context.mapping,
        train_events=context.train_events,
        interaction_edges=pd.DataFrame(columns=["visitor_idx", "item_idx", "weight"]),
        t_train=T_TRAIN,
        graph_dir=graph_dir,
    )
    with pytest.raises(ValueError, match="khong co canh tuong tac"):
        BTDKGRec().fit(empty)


def test_describe_records_what_a_reader_needs_to_reproduce_the_run(setup) -> None:
    make_context, _, _ = setup
    model = BTDKGRec()
    model.fit(make_context(_config(max_epochs=300)))

    record = model.describe()
    assert record["model"] == "bt_dkgrec"
    assert record["supports_cold_start"] is False
    assert record["embedding_dim"] == 32 and record["num_layers"] == 2
    assert record["weighting"] == "behavior_time"
    assert record["training"]["n_epochs"] == 300
    json.dumps(record)  # must survive the metrics.json round trip


# ══ Buoc 7 & 8: the two subclasses ═══════════════════════════════════════


def _variant_config(model: str, **training):
    overrides = {
        "training": {
            "device": "cpu", "batch_size": 128, "eval_every": 5, "patience": 20, **training
        }
    }
    return load_config(model=model, cohort="original", seed=2020, overrides=overrides)


@pytest.fixture
def all_three_graphs(tmp_path):
    """Build the graph of every variant from ONE interim directory.

    Same source data for all three, exactly as ``scripts/02_build_graph.py --all``
    does it -- so nothing can differ except what the variant itself decides.
    """
    from src.models.lightgcn import LightGCN
    from src.models.static_kg_gcn import StaticKGGCN

    interim = tmp_path / "interim"
    interim.mkdir()
    _synthetic_interim(interim)

    mapping = IdMapping.load(interim)
    events = pd.read_parquet(interim / "events.parquet")
    events["visitor_idx"] = mapping.visitor_index(events["visitorid"])
    events["item_idx"] = mapping.item_index(events["itemid"])
    train = events[["visitor_idx", "item_idx", "behavior", "timestamp"]]

    built = {}
    for model_class in (LightGCN, StaticKGGCN, BTDKGRec):
        cfg = _variant_config(model_class.name)
        graph, edges = build_graph(cfg, interim, weighting_for_model(cfg))
        graph_dir = tmp_path / model_class.name
        save_graph(graph, edges, graph_dir)
        built[model_class.name] = (model_class, cfg, graph_dir, edges, mapping, train)
    return built


def test_each_variant_loads_the_graph_built_for_itself(all_three_graphs) -> None:
    for name, (model_class, cfg, graph_dir, edges, mapping, train) in all_three_graphs.items():
        model = model_class()
        model._prepare(
            ModelContext(cfg=cfg, mapping=mapping, train_events=train,
                         interaction_edges=edges, t_train=T_TRAIN, graph_dir=graph_dir)
        )
        assert model.name == name
        assert model._graph_stats["model"] == name


def test_a_variant_refuses_another_variants_graph(all_three_graphs) -> None:
    """★ Without this, an ablation could silently train on the wrong matrix.

    The failure would be invisible: the run completes, the metrics look
    plausible, and the reported comparison is between two copies of the same
    model.
    """
    _, cfg, _, edges, mapping, train = all_three_graphs["static_kg_gcn"]
    wrong_dir = all_three_graphs["bt_dkgrec"][2]

    from src.models.static_kg_gcn import StaticKGGCN

    with pytest.raises(ValueError, match="duoc dung cho"):
        StaticKGGCN()._prepare(
            ModelContext(cfg=cfg, mapping=mapping, train_events=train,
                         interaction_edges=edges, t_train=T_TRAIN, graph_dir=wrong_dir)
        )


def test_each_variant_records_its_own_weighting_in_the_run_artifact(all_three_graphs) -> None:
    """metrics.json must say which weighting produced the numbers."""
    expected = {"bt_dkgrec": "behavior_time", "static_kg_gcn": "uniform", "lightgcn": "uniform"}
    for name, (model_class, cfg, graph_dir, edges, mapping, train) in all_three_graphs.items():
        model = model_class()
        model._prepare(
            ModelContext(cfg=cfg, mapping=mapping, train_events=train,
                         interaction_edges=edges, t_train=T_TRAIN, graph_dir=graph_dir)
        )
        record = model.describe()
        assert record["model"] == name
        assert record["weighting"] == expected[name]
        assert record["supports_cold_start"] is False


def test_the_registry_offers_all_five_models() -> None:
    from src.models.registry import available_models, build_model

    assert set(available_models()) == {
        "popularity", "recent_popularity", "lightgcn", "static_kg_gcn", "bt_dkgrec"
    }
    for name in ("lightgcn", "static_kg_gcn", "bt_dkgrec"):
        assert build_model(_variant_config(name)).name == name
