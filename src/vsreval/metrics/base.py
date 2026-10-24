"""The metric interface and the per-sample inputs shared by all metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import torch

from ..video import Sample, align, canonicalize, read_frames


class SampleData:
    """Lazily loaded inputs of one sample.  Prediction and ground truth are converted to the canonical video format (see `video.py`), so
    both sides of a full-reference metric went through the same RGB to YUV rounding, and every metric reads those files: the frames for
    image metrics, the prediction's file itself for video metrics."""

    def __init__(self, sample: Sample, cache_dir: Path):
        self.sample, self.cache_dir = sample, cache_dir

    @cached_property
    def _frames(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = read_frames(self.video)
        if self.sample.gt is None:
            return pred, None
        return align(pred, read_frames(canonicalize(self.sample.gt, self.cache_dir / "gt")))

    @property
    def pred(self) -> torch.Tensor:
        """(T, 3, H, W) float32 RGB in [0, 1]."""
        return self._frames[0]

    @property
    def gt(self) -> torch.Tensor:
        if self._frames[1] is None:
            raise ValueError(f"'{self.sample.name}' has no ground truth")
        return self._frames[1]

    @cached_property
    def video(self) -> Path:
        return canonicalize(self.sample.pred, self.cache_dir / "pred")


class Metric(ABC):
    name: ClassVar[str]
    needs_gt: ClassVar[bool] = False

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def __call__(self, data: SampleData) -> float:
        """The score of one sample."""
