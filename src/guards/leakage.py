"""Executable data-leakage guards.

CLAUDE.md lists seven inviolable rules. This module turns each one into a pure
assertion function so it can be called from two places: while a structure is
still in memory (during preprocessing or graph building) and again against the
Parquet files after they have been written.

**Never disable a guard to make the pipeline pass.** A failing guard is a real
bug: results produced past a silenced guard have no academic value.

The rules split into two families, and the assertions are deliberately kept
separate so a failure names the actual defect:

Temporal leaks -- the model sees the future
    Rule 1  :func:`assert_temporal_split`
    Rule 3  :func:`assert_side_info_cutoff`, :func:`assert_edges_within_train`
    Rule 4  :func:`assert_single_category_per_item`,
            :func:`assert_latest_record_selected`

Identity leaks -- future entities acquire an identity in the model
    Rule 2  :func:`assert_train_only_mapping`
    Rule 5  :func:`assert_candidate_scope`
    Rule 6  :func:`assert_negatives_in_train`
    Rule 7  :func:`assert_model_selection_scope`

Every violation raises :class:`LeakageError` carrying the count, the share, and
a sample of offending rows -- an empty ``AssertionError`` is not diagnosable at
three in the morning.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

#: How many offending rows are shown inside a LeakageError message.
EXAMPLE_ROWS = 5

TEMPORAL_RULES = (1, 3, 4)
IDENTITY_RULES = (2, 5, 6, 7)


class LeakageError(RuntimeError):
    """Raised when a data-leakage rule is violated.

    Attributes:
        rule: The rule identifier from CLAUDE.md, e.g. ``"3"`` or ``"4 (item-category)"``.
        kind: ``"temporal"`` or ``"identity"``.
        n_violations: Number of offending rows.
    """

    def __init__(
        self,
        rule: str,
        kind: str,
        title: str,
        n_violations: int,
        n_total: int,
        facts: dict[str, object] | None = None,
        examples: pd.DataFrame | None = None,
    ) -> None:
        self.rule = rule
        self.kind = kind
        self.n_violations = n_violations
        share = 100 * n_violations / n_total if n_total else 0.0

        lines = [
            "",
            f"[QUY TAC {rule} — leak {'thoi gian' if kind == 'temporal' else 'danh tinh'}] {title}",
            f"  So dong vi pham : {n_violations:,} / {n_total:,} ({share:.4f}%)",
        ]
        for key, value in (facts or {}).items():
            # Counts get thousands separators; timestamps are passed as str by the
            # call sites so they stay copy-pasteable when debugging.
            formatted = f"{value:,}" if isinstance(value, int) else value
            lines.append(f"  {key:<16}: {formatted}")
        if examples is not None and len(examples):
            lines.append(f"  Vi du {min(len(examples), EXAMPLE_ROWS)} dong dau:")
            lines.extend(
                "    " + line
                for line in examples.head(EXAMPLE_ROWS).to_string(index=False).splitlines()
            )
        lines.append("  KHONG duoc tat guard de pipeline chay qua — guard fail = bug that.")
        super().__init__("\n".join(lines))


def _fail(
    rule: str, kind: str, title: str, mask: np.ndarray, frame: pd.DataFrame,
    facts: dict[str, object] | None = None,
) -> None:
    """Raise :class:`LeakageError` describing the rows selected by ``mask``."""
    raise LeakageError(
        rule=rule, kind=kind, title=title,
        n_violations=int(mask.sum()), n_total=int(len(mask)),
        facts=facts, examples=frame.loc[mask],
    )


# ══════════════════════════════════════════════════════════════════════════
# Temporal leaks
# ══════════════════════════════════════════════════════════════════════════


def assert_temporal_split(events: pd.DataFrame, t_train: int, t_valid_end: int) -> None:
    """Rule 1 -- the split is chronological, never random.

    Checks that every train event precedes the train boundary, every validation
    event falls inside its window, and no test event predates a train event.

    Args:
        events: Frame with ``timestamp`` and a categorical ``split`` column.
        t_train: Inclusive upper bound of the train window.
        t_valid_end: Inclusive upper bound of the validation window.

    Raises:
        LeakageError: If any event sits in the wrong window.
    """
    windows = {"train": (None, t_train), "valid": (t_train, t_valid_end), "test": (t_valid_end, None)}
    for split, (lower, upper) in windows.items():
        subset = events[events["split"] == split]
        if subset.empty:
            continue
        bad = np.zeros(len(subset), dtype=bool)
        if lower is not None:
            bad |= subset["timestamp"].to_numpy() <= lower
        if upper is not None:
            bad |= subset["timestamp"].to_numpy() > upper
        if bad.any():
            _fail(
                "1", "temporal", f"su kien thuoc split {split!r} nam ngoai cua so thoi gian",
                bad, subset[["timestamp", "visitorid", "itemid", "split"]],
                {"cua so": f"({lower}, {upper}]", "T_train": str(t_train), "T_valid_end": str(t_valid_end)},
            )
    log.info("guard rule 1 OK: split theo dung truc thoi gian")


def assert_side_info_cutoff(
    frame: pd.DataFrame, t_train: int, name: str, column: str = "valid_from"
) -> None:
    """Rule 3 -- side information may not carry records newer than ``T_train``.

    This is the rule the v11 thesis violated: item attributes were loaded from
    every timestamp, so the training graph saw the future state of products
    (docs/DECISIONS.md muc D11). The guard fails on a single offending edge.

    Args:
        frame: Side-information edges.
        t_train: Inclusive cutoff.
        name: Table name used in the error message.
        column: Column holding the record timestamp.

    Raises:
        LeakageError: If any row is newer than ``t_train``, or ``column`` is absent.
    """
    if column not in frame.columns:
        raise LeakageError(
            rule="3", kind="temporal",
            title=f"bang {name!r} khong co cot {column!r} nen khong the kiem chung moc thoi gian",
            n_violations=len(frame), n_total=len(frame),
            facts={"cot hien co": ", ".join(frame.columns)},
        )
    if frame.empty:
        log.info("guard rule 3 OK: %s rong", name)
        return
    stamps = frame[column].to_numpy()
    bad = stamps > t_train
    if bad.any():
        _fail(
            "3", "temporal", f"canh side info {name!r} co {column} > T_train",
            bad, frame,
            {"T_train": str(t_train), "max(ts)": str(int(stamps.max())),
             "tre hon T_train (ms)": str(int(stamps[bad].max() - t_train))},
        )
    log.info(
        "guard rule 3 OK: %s — max(%s)=%d <= T_train=%d",
        name, column, int(stamps.max()), t_train,
    )


def assert_edges_within_train(
    edges: pd.DataFrame, t_train: int, name: str = "INTERACTED_WITH", column: str = "timestamp"
) -> None:
    """Rule 3 -- interaction edges are built from train events only."""
    if edges.empty:
        return
    stamps = edges[column].to_numpy()
    bad = stamps > t_train
    if bad.any():
        _fail(
            "3", "temporal", f"canh tuong tac {name!r} sinh tu su kien sau T_train",
            bad, edges, {"T_train": str(t_train), "max(ts)": str(int(stamps.max()))},
        )
    log.info("guard rule 3 OK: %s — moi canh <= T_train", name)


def assert_single_category_per_item(item_category: pd.DataFrame) -> None:
    """Rule 4 -- an item holds exactly one category snapshot at ``T_train``.

    More than one row per item means the temporal snapshot was not resolved and
    the graph would carry stale and current categories side by side.
    """
    if item_category.empty:
        return
    counts = item_category["item_idx"].value_counts()
    offenders = counts[counts > 1]
    if len(offenders):
        duplicated = item_category["item_idx"].isin(offenders.index).to_numpy()
        _fail(
            "4", "temporal", "item co nhieu hon mot category tai T_train",
            duplicated, item_category.sort_values("item_idx"),
            {"item bi trung": int(len(offenders)), "so ban ghi max": int(offenders.max())},
        )
    log.info("guard rule 4 OK: %s item, moi item dung mot category", f"{len(item_category):,}")


def assert_latest_record_selected(
    keys: Sequence[np.ndarray], timestamps: np.ndarray, selected: np.ndarray, rule: str = "4"
) -> None:
    """Rule 4 -- the record kept per group is the newest one, not the last line.

    The classic trap: ``item_properties`` holds many rows per item at different
    timestamps, and taking the final line of the file silently reads the future.
    This verifies the selection against a per-group maximum, and that exactly one
    row survives per group.

    Args:
        keys: Grouping columns, e.g. ``[item_ids]`` or ``[item_ids, prop_codes]``.
        timestamps: Record timestamps, aligned with ``keys``.
        selected: Positions chosen as the newest record of each group.
        rule: Label used in the error message.

    Raises:
        LeakageError: If a selected row is not its group's maximum, or if the
            number of selected rows differs from the number of groups.
    """
    if len(timestamps) == 0:
        return
    frame = pd.DataFrame({f"k{i}": k for i, k in enumerate(keys)})
    frame["timestamp"] = timestamps
    group_columns = [c for c in frame.columns if c != "timestamp"]
    group_max = frame.groupby(group_columns, sort=False)["timestamp"].transform("max").to_numpy()

    chosen = np.zeros(len(frame), dtype=bool)
    chosen[selected] = True
    stale = chosen & (timestamps < group_max)
    if stale.any():
        report = frame.copy()
        report["max_cua_nhom"] = group_max
        _fail(
            rule, "temporal", "ban ghi duoc giu KHONG phai ban ghi moi nhat cua nhom",
            stale, report, {"so nhom": int(frame[group_columns].drop_duplicates().shape[0])},
        )

    n_groups = int(frame[group_columns].drop_duplicates().shape[0])
    if len(selected) != n_groups:
        raise LeakageError(
            rule=rule, kind="temporal",
            title="so ban ghi giu lai khac so nhom (thua hoac thieu snapshot)",
            n_violations=abs(len(selected) - n_groups), n_total=n_groups,
            facts={"so nhom": n_groups, "so ban ghi giu": len(selected)},
        )


# ══════════════════════════════════════════════════════════════════════════
# Identity leaks
# ══════════════════════════════════════════════════════════════════════════


def assert_train_only_mapping(
    visitor_ids: np.ndarray,
    item_ids: np.ndarray,
    events: pd.DataFrame,
) -> None:
    """Rule 2 -- identifier mappings are built from train events only.

    Two failure modes are checked: an id present in the mapping but absent from
    train (it leaked in from validation or test), and an id present in train but
    missing from the mapping (the mapping is incomplete, which silently drops
    supervision).

    Args:
        visitor_ids: Mapped visitor ids.
        item_ids: Mapped item ids.
        events: All cohort events, carrying a ``split`` column.

    Raises:
        LeakageError: On either failure mode.
    """
    train = events[events["split"] == "train"]
    for label, mapped, column in (
        ("visitor", visitor_ids, "visitorid"),
        ("item", item_ids, "itemid"),
    ):
        in_train = pd.unique(train[column])
        extra = np.setdiff1d(mapped, in_train, assume_unique=False)
        missing = np.setdiff1d(in_train, mapped, assume_unique=False)
        if len(extra):
            future = events[events[column].isin(extra[:EXAMPLE_ROWS])]
            raise LeakageError(
                rule="2", kind="identity",
                title=f"{label} co trong mapping nhung KHONG co trong train",
                n_violations=len(extra), n_total=len(mapped),
                facts={f"{label} id vi du": ", ".join(map(str, extra[:EXAMPLE_ROWS]))},
                examples=future[["timestamp", column, "split"]],
            )
        if len(missing):
            raise LeakageError(
                rule="2", kind="identity",
                title=f"{label} co trong train nhung THIEU trong mapping",
                n_violations=len(missing), n_total=len(in_train),
                facts={f"{label} id vi du": ", ".join(map(str, missing[:EXAMPLE_ROWS]))},
            )
    log.info(
        "guard rule 2 OK: mapping dung bang tap train (%s visitor, %s item)",
        f"{len(visitor_ids):,}", f"{len(item_ids):,}",
    )


def assert_index_within_mapping(
    frame: pd.DataFrame, column: str, n_entities: int, name: str
) -> None:
    """Rule 2 -- index columns hold matrix positions, not raw identifiers.

    A column named ``*_idx`` that still carries raw ids points outside the
    mapping and would silently address the wrong rows of the adjacency matrix.
    """
    if frame.empty:
        return
    values = frame[column].to_numpy()
    bad = (values < 0) | (values >= n_entities)
    if bad.any():
        _fail(
            "2", "identity", f"cot {column!r} cua {name!r} nam ngoai khong gian chi so",
            bad, frame,
            {"so entity": n_entities, "max gia tri": int(values.max()), "min gia tri": int(values.min())},
        )
    log.info("guard rule 2 OK: %s.%s trong [0, %s)", name, column, f"{n_entities:,}")


def assert_candidate_scope(candidates: np.ndarray, train_item_ids: np.ndarray) -> None:
    """Rule 5 -- the candidate set is a subset of ``I_train``.

    A candidate outside the train universe has no learned embedding, so ranking
    it would either crash or fabricate a score.
    """
    outside = np.setdiff1d(candidates, train_item_ids)
    if len(outside):
        raise LeakageError(
            rule="5", kind="identity",
            title="candidate set chua item khong thuoc I_train",
            n_violations=len(outside), n_total=len(candidates),
            facts={
                "|I_train|": int(len(train_item_ids)),
                "item la vi du": ", ".join(map(str, outside[:EXAMPLE_ROWS])),
            },
        )
    log.info("guard rule 5 OK: %s candidate deu thuoc I_train", f"{len(candidates):,}")


def assert_negatives_in_train(negatives: np.ndarray, train_item_ids: np.ndarray) -> None:
    """Rule 6 -- negative samples are drawn from ``I_train`` only.

    Sampling a negative from a future item would teach the model about items it
    is not allowed to know yet.
    """
    outside = np.setdiff1d(np.unique(negatives), train_item_ids)
    if len(outside):
        raise LeakageError(
            rule="6", kind="identity",
            title="negative sampling lay item ngoai I_train",
            n_violations=len(outside), n_total=int(len(np.unique(negatives))),
            facts={
                "|I_train|": int(len(train_item_ids)),
                "item la vi du": ", ".join(map(str, outside[:EXAMPLE_ROWS])),
            },
        )


def assert_model_selection_scope(monitor: str, consulted_splits: Iterable[str]) -> None:
    """Rule 7 -- model selection reads validation metrics, never test.

    Args:
        monitor: The metric key driving early stopping and checkpoint choice.
        consulted_splits: Every split whose metrics were read while selecting.

    Raises:
        LeakageError: If the monitored metric or any consulted split is test.
    """
    consulted = sorted(set(consulted_splits))
    offending = [s for s in consulted if "test" in s.lower()]
    if "test" in monitor.lower() or offending:
        raise LeakageError(
            rule="7", kind="identity",
            title="model selection doc metric cua tap test",
            n_violations=len(offending) or 1, n_total=max(len(consulted), 1),
            facts={"monitor": monitor, "split da doc": ", ".join(consulted) or "(khong)"},
        )
    log.info("guard rule 7 OK: model selection theo %r tren %s", monitor, consulted)


# ══════════════════════════════════════════════════════════════════════════
# Gate
# ══════════════════════════════════════════════════════════════════════════


def run_preprocess_guards(
    events: pd.DataFrame,
    visitor_ids: np.ndarray,
    item_ids: np.ndarray,
    item_category: pd.DataFrame,
    item_property: pd.DataFrame,
    t_train: int,
    t_valid_end: int,
    monitor: str,
) -> list[str]:
    """Run every guard checkable at preprocessing time.

    Called as a gate from ``scripts/01_preprocess.py`` so the rules run on every
    single preprocess, not only when someone remembers to run the test suite.

    Rules 5 and 6 are **not** included: no candidate set or negative sampler
    exists at preprocessing time, and asserting ``I_train subset of I_train``
    would print a PASS that verifies nothing. They are enforced where their
    artefacts appear -- the evaluator (Buoc 5) and the sampler (Buoc 6).

    Returns:
        Human-readable names of the guards that passed.

    Raises:
        LeakageError: On the first violation found.
    """
    assert_temporal_split(events, t_train, t_valid_end)
    assert_train_only_mapping(visitor_ids, item_ids, events)
    assert_side_info_cutoff(item_category, t_train, "item_category")
    assert_side_info_cutoff(item_property, t_train, "item_property")
    assert_single_category_per_item(item_category)
    assert_index_within_mapping(item_category, "item_idx", len(item_ids), "item_category")
    assert_index_within_mapping(item_property, "item_idx", len(item_ids), "item_property")
    assert_edges_within_train(
        events[events["split"] == "train"], t_train, name="INTERACTED_WITH (train events)"
    )
    assert_model_selection_scope(monitor, consulted_splits=["valid"])

    return [
        "rule 1 — split theo thoi gian",
        "rule 2 — mapping chi tu train",
        "rule 3 — side info <= T_train",
        "rule 3 — canh tuong tac <= T_train",
        "rule 4 — mot category moi nhat / item",
        "rule 2 — chi so side info nam trong mapping",
        "rule 7 — model selection theo valid",
    ]
