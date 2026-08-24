"""Single place that turns ``cfg.model.name`` into a model instance.

Torch is imported **lazily**, inside :func:`_graph_models`, and that is
deliberate. ``requirements.txt`` -- the environment the VPS runs preprocessing,
graph building and the test suite in -- does not contain torch; only
``requirements-colab.txt`` does. Importing :mod:`src.models.bt_dkgrec` at module
level would make ``python scripts/03_train.py --model popularity`` fail on a
machine that has no business needing a deep-learning framework to count item
frequencies.

Buoc 7 and Buoc 8 register their models by adding one line each to
:func:`_graph_models`; nothing else in the pipeline learns their names.
"""

from __future__ import annotations

from src.models.base import Recommender
from src.models.popularity import HEURISTIC_MODELS
from src.utils.config import Config


def _graph_models() -> dict[str, type[Recommender]]:
    """Graph models, imported on demand because they require torch."""
    from src.models.bt_dkgrec import BTDKGRec

    return {BTDKGRec.name: BTDKGRec}


def available_models() -> tuple[str, ...]:
    """Every model name the training script can currently instantiate."""
    return tuple(sorted(HEURISTIC_MODELS)) + tuple(sorted(_graph_models()))


def build_model(cfg: Config) -> Recommender:
    """Instantiate the model named by ``cfg``.

    Raises:
        SystemExit: If the model is a valid config name but not yet implemented,
            with the roadmap step that will add it.
    """
    name = cfg.model.name
    if name in HEURISTIC_MODELS:
        return HEURISTIC_MODELS[name]()

    graph_models = _graph_models()
    if name in graph_models:
        return graph_models[name]()

    raise SystemExit(
        f"model {name!r} chua duoc trien khai (hien co: {', '.join(available_models())})"
    )


def is_trainable(cfg: Config) -> bool:
    """Whether the model learns parameters and therefore needs the trainer."""
    return bool(cfg.model.trainable)
