from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int = 0,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    deterministic=True can improve repeatability,
    although some deterministic operations may
    run more slowly.
    """

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True
        )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True