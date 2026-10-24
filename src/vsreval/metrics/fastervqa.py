"""FasterVQA (Wu et al., TPAMI 2023) score, as the repository's `vqa.py -m FasterVQA` computes it: `FAST_VQA_3D_1_1.pth`, the `f3dvqa-b.yml`
fragment sampling and the sigmoid rescaling of the raw prediction."""
from __future__ import annotations

import decord
import numpy as np
import torch
import torch.nn as nn

from .._vendor.dover.datasets import get_spatial_fragments
from .._vendor.dover.models import SwinTransformer3D, VQAHead
from .._vendor.fastvqa import sample_frame_indices
from ..weights import ensure
from .base import Metric, SampleData

# the config of f3dvqa-b.yml; as in the FAST-VQA repository's vqa.py the temporal sampler is built from clip_len and num_clips only (frame_interval = 1)
SAMPLE = dict(fragments_h=7, fragments_w=7, fsize_h=32, fsize_w=32, aligned=8, clip_len=32, frame_interval=2, t_frag=8, num_clips=1)
SEED = 0  # the fragment positions inside every grid cell are drawn at random (torch.randint), also at test time
MEAN = torch.tensor([123.675, 116.28, 103.53])
STD = torch.tensor([58.395, 57.12, 57.375])
SCORE_MEAN, SCORE_STD = 0.14759505, 0.03613452


class _FasterVQANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fragments_backbone = SwinTransformer3D()
        self.vqa_head = VQAHead(in_channels=768, hidden_channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FAST-VQA's backbone runs with adaptive window sizes by default (DOVER's copy of the backbone does not)
        return self.vqa_head(self.fragments_backbone(x, multi=False, layer=-1, adaptive_window_size=True))


class FasterVQA(Metric):
    name = "fastervqa"

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.net = _FasterVQANet()
        self.net.load_state_dict(torch.load(ensure("fastervqa"), map_location="cpu")["state_dict"])
        self.net.to(device).eval()

    @torch.no_grad()
    def __call__(self, data: SampleData) -> float:
        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(str(data.video))
        idx = sample_frame_indices(len(reader), SAMPLE["clip_len"], SAMPLE["num_clips"])
        frames = {int(i): reader[int(i)] for i in np.unique(idx)}
        video = torch.stack([frames[int(i)] for i in idx], 0).permute(3, 0, 1, 2)  # (C, T, H, W)
        with torch.random.fork_rng(devices=[]):  # same crops for every model and run, global RNG untouched
            torch.manual_seed(SEED)
            x = get_spatial_fragments(video, **SAMPLE)
        x = ((x.permute(1, 2, 3, 0) - MEAN) / STD).permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], SAMPLE["num_clips"], -1, *x.shape[2:]).transpose(0, 1)
        raw = self.net(x.to(self.device)).mean().item()
        return float(1 / (1 + np.exp(-(raw - SCORE_MEAN) / SCORE_STD)))
