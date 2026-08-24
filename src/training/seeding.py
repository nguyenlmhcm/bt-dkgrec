"""Seed control for reproducible runs.

Every model runs on the same seed list ``[2020, 2021, 2022]``
(``experiments/seeds.json``). Results are reported as mean +/- std over those
three runs, and models are compared with Welch's t-test.

Bit-exact reproducibility is **not** claimed on GPU: sparse operations on CUDA
are not bit-exact even with a fixed seed. That is why the protocol uses an
independent two-sample test rather than a paired one (CLAUDE.md muc
"Non-determinism").
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every random source used by the project.

    Args:
        seed: The run seed.
        deterministic_torch: Ask cuDNN for deterministic kernels where possible.
            Sparse CUDA kernels remain non-deterministic regardless.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # torch exists only where training happens: Colab, not the VPS (D28)
        import torch
    except ImportError:
        log.info("seed=%d (numpy, random) — torch chua cai, bo qua", seed)
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info("seed=%d (numpy, random, torch)", seed)
