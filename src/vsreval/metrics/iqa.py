"""Full- and no-reference image metrics through pyiqa, averaged over the frames of a video."""
from __future__ import annotations

import numpy as np
import torch

from .base import Metric, SampleData

FULL_REFERENCE = ("psnr", "ssim", "lpips", "dists")
NO_REFERENCE = ("musiq", "clipiqa", "niqe", "maniqa")


class PyiqaMetric(Metric):
    def __init__(self, name: str, device: torch.device):
        import pyiqa

        super().__init__(device)
        self.name = name
        self.needs_gt = name in FULL_REFERENCE
        self.model = pyiqa.create_metric(name, device=device).eval()

    @torch.no_grad()
    def __call__(self, data: SampleData) -> float:
        # frame by frame: a whole high-resolution clip overflows the 32-bit indexing of some CUDA kernels
        pred = data.pred
        if self.needs_gt:
            gt = data.gt
            scores = [self.model(pred[i:i + 1].to(self.device), gt[i:i + 1].to(self.device)).item() for i in range(len(pred))]
        else:
            scores = [self.model(pred[i:i + 1].to(self.device)).item() for i in range(len(pred))]
        return float(np.mean(scores))
