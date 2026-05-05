"""
Seed Utilities
==============
Deterministic seeding for reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for: Python random, NumPy, PyTorch (CPU + CUDA).

    Args:
        seed: The random seed value.
        deterministic_cudnn: If True, set CuDNN to deterministic mode
            (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
