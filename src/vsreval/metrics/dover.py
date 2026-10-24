"""DOVER overall score (Wu et al., ICCV 2023): the `val-l1080p` sampling of `dover.yml` and the score fusion of
`evaluate_a_set_of_videos.py`, weights `DOVER.pth`."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .._vendor.dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from .._vendor.dover.models import SwinTransformer3D, VQAHead, convnext_3d_tiny
from ..weights import ensure
from .base import Metric, SampleData

SAMPLE_TYPES = {
    "technical": dict(fragments_h=7, fragments_w=7, fsize_h=32, fsize_w=32, aligned=32, clip_len=32, frame_interval=2, num_clips=3),
    "aesthetic": dict(size_h=224, size_w=224, clip_len=32, frame_interval=2, t_frag=32, num_clips=1),
}
SEED = 0  # the fragment positions inside every grid cell are drawn at random (torch.randint), also at test time
MEAN = torch.tensor([123.675, 116.28, 103.53])
STD = torch.tensor([58.395, 57.12, 57.375])


class _DoverNet(nn.Module):
    """The two-branch DOVER of the released checkpoint: a technical (Swin-T, fragments) and an aesthetic (ConvNeXt-T) branch."""

    def __init__(self):
        super().__init__()
        self.technical_backbone = SwinTransformer3D()
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=False)  # every weight comes from DOVER.pth
        self.technical_head = VQAHead(in_channels=768, hidden_channels=64)
        self.aesthetic_head = VQAHead(in_channels=768, hidden_channels=64)

    def forward(self, views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: getattr(self, f"{k}_head")(getattr(self, f"{k}_backbone")(v, multi=False, layer=-1)) for k, v in views.items()}


def _sampler(opt: dict) -> UnifiedFrameSampler:
    if "t_frag" not in opt:
        return UnifiedFrameSampler(opt["clip_len"], opt["num_clips"], opt["frame_interval"])
    return UnifiedFrameSampler(opt["clip_len"] // opt["t_frag"], opt["t_frag"], opt["frame_interval"], opt["num_clips"])


def fuse(aesthetic: float, technical: float) -> float:
    t, a = (technical - 0.1107) / 0.07355, (aesthetic + 0.08285) / 0.03774
    return float(1 / (1 + np.exp(-(t * 0.6104 + a * 0.3896))))


class Dover(Metric):
    name = "dover"

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.net = _DoverNet()
        self.net.load_state_dict(torch.load(ensure("dover"), map_location="cpu"))
        self.net.to(device).eval()
        self.samplers = {k: _sampler(v) for k, v in SAMPLE_TYPES.items()}

    @torch.no_grad()
    def __call__(self, data: SampleData) -> float:
        with torch.random.fork_rng(devices=[]):  # same crops for every model and run, global RNG untouched
            torch.manual_seed(SEED)
            np_state = np.random.get_state()
            np.random.seed(SEED)  # the temporal sampler also draws random offsets, for videos longer than ~190 frames
            try:
                views, _ = spatial_temporal_view_decomposition(str(data.video), SAMPLE_TYPES, self.samplers)
            finally:
                np.random.set_state(np_state)
        batch = {}
        for key in ("aesthetic", "technical"):
            v = ((views[key].permute(1, 2, 3, 0) - MEAN) / STD).permute(3, 0, 1, 2)[None].to(self.device)  # (1, C, T, H, W)
            b, c, t, h, w = v.shape
            n = SAMPLE_TYPES[key]["num_clips"]
            batch[key] = v.reshape(b, c, n, t // n, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b * n, c, t // n, h, w)
        out = self.net(batch)
        return fuse(aesthetic=float(out["aesthetic"].mean()), technical=float(out["technical"].mean()))
