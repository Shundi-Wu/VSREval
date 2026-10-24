"""The metric registry.  Heavy dependencies are imported only when a metric is created."""
from __future__ import annotations

from typing import Callable

import torch

from .base import Metric
from .iqa import FULL_REFERENCE, NO_REFERENCE


def _pyiqa(name: str) -> Callable[[torch.device], Metric]:
    def make(device: torch.device) -> Metric:
        from .iqa import PyiqaMetric

        return PyiqaMetric(name, device)

    return make


def _video(module: str, cls: str) -> Callable[[torch.device], Metric]:
    def make(device: torch.device) -> Metric:
        import importlib

        return getattr(importlib.import_module(f"{__package__}.{module}"), cls)(device)

    return make


REGISTRY: dict[str, Callable[[torch.device], Metric]] = {
    **{n: _pyiqa(n) for n in FULL_REFERENCE + NO_REFERENCE},
    "dover": _video("dover", "Dover"),
    "fastervqa": _video("fastervqa", "FasterVQA"),
    "ewarp": _video("ewarp", "EWarp"),
}
NEEDS_GT = set(FULL_REFERENCE)
DEFAULT_METRICS = list(REGISTRY)


def create(name: str, device: torch.device) -> Metric:
    if name not in REGISTRY:
        raise KeyError(f"unknown metric '{name}'; available: {', '.join(REGISTRY)}")
    return REGISTRY[name](device)
