"""
Seed management for reproducibility across experiments.
"""
import random
import numpy as np
import os


GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    """
    Set random seed for reproducibility across Python, NumPy, and environment.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # XGBoost uses numpy's random state
    # If using PyTorch/TensorFlow, add their seed setters here

    print(f"[Seed] Set global seed to {seed}")


def get_seed() -> int:
    """Return the global seed value."""
    return GLOBAL_SEED
