"""Fase 2.6 - Reprodutibilidade fechada por seed."""
from __future__ import annotations
import os
import random


def configurar_ambiente_determinismo() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seeds(seed: int = 42) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
