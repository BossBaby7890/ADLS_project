"""
src.models — Architecture definitions for APQ-Lite experiments.

Role in the pipeline
--------------------
Provides clean, registerable model factories so that any stage of the
APQ-Lite pipeline (profiling, QAT, evaluation) can instantiate the same
architecture from a config string without hard-coding import paths.

Adding a new architecture
    1. Create ``src/models/<arch_name>.py`` and define a ``build_<arch_name>``
       factory function.
    2. Register it in ``MODEL_REGISTRY`` below.
    3. Reference it in ``configs/base_config.yaml`` as ``model.architecture``.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

# Registry: config name → factory(num_classes) -> nn.Module
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model factory under ``name``."""
    def decorator(fn: Callable) -> Callable:
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(architecture: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Instantiate a model by its registry name.

    Parameters
    ----------
    architecture:
        Key in ``MODEL_REGISTRY``, e.g. ``"resnet20"``.
    num_classes:
        Number of output classes.

    Raises
    ------
    KeyError
        If ``architecture`` is not in the registry.
    """
    if architecture not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Unknown architecture '{architecture}'. Available: {available}"
        )
    return MODEL_REGISTRY[architecture](num_classes=num_classes, **kwargs)


# ---- Import concrete architectures so they register themselves ----
from . import resnet  # noqa: E402, F401

__all__ = ["build_model", "register_model", "MODEL_REGISTRY"]
