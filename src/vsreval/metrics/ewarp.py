"""E*warp: warping error of a video, after the RAFT-based `compute_video_warping_error` of EvalCrafter.

For every pair of consecutive frames (downscaled by 2): forward and backward RAFT flow, occlusion / motion-boundary mask from the
forward-backward consistency, and the squared difference between the first frame and the second frame warped onto it, over the
non-occluded pixels; averaged over the pairs and multiplied by 1000.

`Resample2d` (a compiled FlowNet2 CUDA extension) is replaced by `resample`, the same bilinear, border-clamped sampling in plain
PyTorch (checked against the kernel's index arithmetic in `tests/test_resample.py`).
"""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

from .._vendor.raft.raft import RAFT
from .._vendor.raft.utils import InputPadder
from ..video import read_frames
from ..weights import ensure
from .base import Metric, SampleData

RAFT_INPUT_SCALE = 255.0
SCORE_SCALE = 1000.0


def resample(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """`img` (1, C, H, W) sampled at (x + flow_x, y + flow_y), bilinear, coordinates outside the image clamped to its border."""
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
    gx = 2.0 * (xs + flow[:, 0]) / max(w - 1, 1) - 1.0
    gy = 2.0 * (ys + flow[:, 1]) / max(h - 1, 1) - 1.0
    return F.grid_sample(img, torch.stack((gx, gy), dim=-1), mode="bilinear", padding_mode="border", align_corners=True)


def _magnitude(flow: np.ndarray) -> np.ndarray:
    return flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2


def _gradients(flow: np.ndarray) -> tuple[np.ndarray, ...]:
    h, w = flow.shape[:2]
    x_du, x_dv, y_du, y_dv = (np.zeros((h, w)) for _ in range(4))
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    x_du[:, :-1] = fx[:, :-1] - fx[:, 1:]
    x_dv[:-1, :] = fx[:-1, :] - fx[1:, :]
    y_du[:, :-1] = fy[:, :-1] - fy[:, 1:]
    y_dv[:-1, :] = fy[:-1, :] - fy[1:, :]
    return x_du, x_dv, y_du, y_dv


def occlusion_and_warp(fw_flow: torch.Tensor, bw_flow: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Occlusion mask (H, W) on the grid of frame 1 and frame 2 warped onto frame 1.
    `fw_flow`: frame 1 -> 2, `bw_flow`: frame 2 -> 1, both (1, 2, H, W)."""
    warped = resample(img2, fw_flow)
    bw_warped = resample(bw_flow, fw_flow)  # the backward flow at the position each pixel moves to
    to_hwc = lambda t: t[0].permute(1, 2, 0).cpu().numpy()  # noqa: E731
    fw, bw_w = to_hwc(fw_flow), to_hwc(bw_warped)

    fw_mag, bw_w_mag = _magnitude(fw), _magnitude(bw_w)
    inconsistent = _magnitude(bw_w + fw) > 0.01 * (bw_w_mag + fw_mag) + 0.5
    x_du, x_dv, y_du, y_dv = _gradients(fw)
    motion_boundary = (x_du ** 2 + x_dv ** 2 + y_du ** 2 + y_dv ** 2) > 0.01 * fw_mag + 0.002
    occlusion = torch.from_numpy(np.logical_or(inconsistent, motion_boundary).astype(np.float32)).to(img2.device)
    return occlusion, warped


class EWarp(Metric):
    name = "ewarp"

    def __init__(self, device: torch.device):
        super().__init__(device)
        args = Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
        self.model = RAFT(args)
        state = torch.load(ensure("raft"), map_location="cpu")
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, data: SampleData) -> float:
        frames = read_frames(data.video)
        if len(frames) < 2:
            raise ValueError(f"E*warp needs at least two frames ({data.sample.name})")
        total = 0.0
        for i in range(len(frames) - 1):
            img1, img2 = (F.interpolate(f[None].to(self.device), scale_factor=0.5, mode="bilinear", align_corners=False)
                          for f in (frames[i], frames[i + 1]))
            img1, img2 = InputPadder(img1.shape).pad(img1, img2)
            _, fw_flow = self.model(img1 * RAFT_INPUT_SCALE, img2 * RAFT_INPUT_SCALE, iters=20, test_mode=True)
            _, bw_flow = self.model(img2 * RAFT_INPUT_SCALE, img1 * RAFT_INPUT_SCALE, iters=20, test_mode=True)
            occlusion, warped = occlusion_and_warp(fw_flow, bw_flow, img2)
            non_occluded = 1 - occlusion
            squared = ((warped - img1) * non_occluded) ** 2
            n = non_occluded.sum()
            total += float(squared.sum() / (n if n != 0 else squared.numel()))
        return total / (len(frames) - 1) * SCORE_SCALE
