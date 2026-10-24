"""Test-mode frame sampling of FAST-VQA / FasterVQA (SampleFrames._get_test_clips + __call__), the only piece of the original
`fastvqa` package that FasterVQA inference needs.  Upstream: VQAssessment/FAST-VQA-and-FasterVQA @ 8db452e (S-Lab License 1.0)."""
import numpy as np


def sample_frame_indices(total_frames: int, clip_len: int, num_clips: int, frame_interval: int = 1) -> np.ndarray:
    ori_clip_len = clip_len * frame_interval
    avg_interval = (total_frames - ori_clip_len + 1) / float(num_clips)
    if total_frames > ori_clip_len - 1:
        base_offsets = np.arange(num_clips) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
    else:
        clip_offsets = np.zeros((num_clips,), dtype=np.int32)
    frame_inds = clip_offsets[:, None] + np.arange(clip_len)[None, :] * frame_interval
    frame_inds = np.mod(frame_inds.reshape(-1), total_frames)
    return frame_inds.astype(np.int32)
