"""Guards that an artifact says what actually happened.

Separate from :mod:`src.guards.leakage`: nothing here is about the model seeing
data it should not. These assertions catch the other failure family -- a run
that completes, writes a healthy-looking artifact, and reports a number that
does not correspond to the model it claims to describe.

That is not hypothetical. An earlier version of :mod:`src.training.trainer`
restored an epoch-0 snapshot whenever validation produced no number, silently
throwing away the whole run while ``curves.csv`` still showed a falling loss.
The bug survived because nothing compared the reported metric against the
trained parameters. This module makes that comparison a step of the pipeline.
"""

from __future__ import annotations

from src.utils.logging import get_logger

log = get_logger(__name__)

#: Above this relative gap the two numbers cannot be the same parameters
#: scoring the same data, so the run is wrong rather than merely imprecise.
FATAL_RELATIVE_GAP = 1e-3

#: Below this, the gap is ordinary float noise and not worth a line in the log.
NOISE_RELATIVE_GAP = 1e-6


class ConsistencyError(RuntimeError):
    """An artifact's numbers contradict each other."""


def assert_selection_restored(
    monitor: str, best_epoch: int, best_value: float, reevaluated: float | None
) -> dict[str, object]:
    """Check the restored parameters reproduce the metric validation selected.

    After training, the script re-scores the validation split with the restored
    parameters. Those are the same parameters that produced ``best_value`` at
    ``best_epoch``, scoring the same data, so the two numbers must agree.

    Sparse CUDA kernels are not bit-exact (CLAUDE.md muc "Non-determinism"), so
    disagreement in the last digits is expected: it is recorded, not raised.

    Args:
        monitor: Metric that drove selection.
        best_epoch: Epoch whose parameters were restored.
        best_value: Metric measured during training at ``best_epoch``.
        reevaluated: Metric measured after restoring, or ``None`` if the split
            could not be scored at all.

    Returns:
        The comparison, to be recorded inside ``metrics.json``.

    Raises:
        ConsistencyError: If the split cannot be scored any more, or the two
            numbers differ by more than :data:`FATAL_RELATIVE_GAP`.
    """
    if reevaluated is None:
        raise ConsistencyError(
            f"train da chon epoch {best_epoch} theo valid {monitor} = {best_value:.6f}, "
            "nhung cham lai sau khi khoi phuc thi khong ra so — tham so khong duoc nap lai"
        )

    relative = abs(reevaluated - best_value) / max(abs(best_value), 1e-12)
    record: dict[str, object] = {
        "monitor": monitor,
        "best_epoch": best_epoch,
        "value_during_training": best_value,
        "value_after_restore": reevaluated,
        "relative_difference": relative,
    }

    if relative > FATAL_RELATIVE_GAP:
        raise ConsistencyError(
            f"valid {monitor} luc train = {best_value:.6f} nhung cham lai sau khi "
            f"khoi phuc = {reevaluated:.6f} (lech {relative:.2%}) — tham so duoc bao cao "
            "KHONG phai tham so ma validation da chon"
        )
    if relative > NOISE_RELATIVE_GAP:
        log.warning(
            "valid %s lech %.2e giua luc train va luc cham lai — trong nguong nhieu cua "
            "kernel sparse CUDA, van ghi lai trong metrics.json",
            monitor, relative,
        )
    else:
        log.info("guard nhat quan OK: valid %s khop sau khi khoi phuc tham so", monitor)
    return record
