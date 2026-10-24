import cv2
import numpy as np
import torch

from vsreval._vendor.fastvqa import sample_frame_indices
from vsreval.metrics import DEFAULT_METRICS, NEEDS_GT
from vsreval.metrics.dover import fuse
from vsreval.video import align, find_samples, read_frames


def _write_frames(directory, n, h, w, seed=0):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        cv2.imwrite(str(directory / f"{i:05d}.png"), rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_fuse_matches_the_published_constants():
    assert abs(fuse(aesthetic=-0.08285, technical=0.1107) - 0.5) < 1e-12  # both normalised scores are 0
    assert fuse(1.0, 1.0) > 0.99 and fuse(-1.0, -1.0) < 0.01


def test_fastervqa_sampling():
    # a 32-frame clip with clip_len 32: all frames in order
    assert sample_frame_indices(32, clip_len=32, num_clips=1).tolist() == list(range(32))
    # a longer video: one clip, shifted by half of the spare length
    idx = sample_frame_indices(100, clip_len=32, num_clips=1)
    assert idx[0] == 34 and idx[-1] == 65
    # too short for the clip: wraps around
    assert sample_frame_indices(10, clip_len=32, num_clips=1).max() < 10


def test_find_samples_matches_ground_truth_by_name(tmp_path):
    for name in ("b", "a"):
        _write_frames(tmp_path / "pred" / name, 2, 8, 8)
        _write_frames(tmp_path / "gt" / name, 2, 8, 8, seed=1)
    samples = find_samples(tmp_path / "pred", tmp_path / "gt")
    assert [s.name for s in samples] == ["a", "b"]
    assert all(s.gt is not None and s.gt.name == s.name for s in samples)
    assert [s.gt for s in find_samples(tmp_path / "pred")] == [None, None]


def test_read_frames_and_align(tmp_path):
    _write_frames(tmp_path / "p", 5, 10, 12)
    _write_frames(tmp_path / "g", 4, 8, 12, seed=1)
    pred, gt = read_frames(tmp_path / "p"), read_frames(tmp_path / "g")
    assert pred.shape == (5, 3, 10, 12) and pred.dtype == torch.float32 and 0 <= pred.min() and pred.max() <= 1
    pred, gt = align(pred, gt)
    assert pred.shape == gt.shape == (4, 3, 8, 12)


def test_registry_is_consistent():
    assert set(NEEDS_GT) <= set(DEFAULT_METRICS)
    assert {"dover", "fastervqa", "ewarp"} <= set(DEFAULT_METRICS)
