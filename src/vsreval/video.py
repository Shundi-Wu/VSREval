"""Reading predictions / ground truth (video files or directories of frames) and converting them to the one format every metric reads.

The canonical format is libx264, yuv444p, lossless (crf 0), 8 fps.  Whatever an input is, the
metrics see the same pixels: `canonicalize` returns the input itself when it already is such a file, and otherwise decodes it to RGB and
encodes it to that format.  A lossy input stays lossy (nothing brings the lost detail back); `probe` tells what an input is so that
this can be reported.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
CANONICAL_FPS = 8
LOSSLESS_CODECS = {"ffv1", "rawvideo", "png", "utvideo", "huffyuv", "ffvhuff", "magicyuv"}


@dataclass(frozen=True)
class Sample:
    name: str
    pred: Path
    gt: Path | None = None


def _is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def _is_frames(p: Path) -> bool:
    return p.is_dir() and any(c.suffix.lower() in IMAGE_EXTS for c in p.iterdir())


def _find_gt(gt_dir: Path, name: str) -> Path:
    for cand in [gt_dir / (name + ext) for ext in VIDEO_EXTS] + [gt_dir / name]:
        if _is_video(cand) or _is_frames(cand):
            return cand
    raise FileNotFoundError(f"no ground truth for '{name}' in {gt_dir}")


def find_samples(pred_dir: str | Path, gt_dir: str | Path | None = None) -> list[Sample]:
    """One sample per video file / frame directory in `pred_dir`; the ground truth is matched by name."""
    pred_dir = Path(pred_dir)
    entries = sorted(p for p in pred_dir.iterdir() if _is_video(p) or _is_frames(p))
    if not entries:
        raise FileNotFoundError(f"no videos or frame directories in {pred_dir}")
    gt_dir = Path(gt_dir) if gt_dir else None
    return [Sample(p.stem if p.is_file() else p.name, p, _find_gt(gt_dir, p.stem if p.is_file() else p.name) if gt_dir else None)
            for p in entries]


def _frame_files(path: Path) -> list[Path]:
    return sorted(c for c in path.iterdir() if c.suffix.lower() in IMAGE_EXTS)


def read_uint8(path: Path) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB."""
    if path.is_dir():
        frames = [cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in _frame_files(path)]
    else:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return np.stack(frames)


def read_frames(path: Path) -> torch.Tensor:
    """(T, 3, H, W) float32 RGB in [0, 1]."""
    return torch.from_numpy(read_uint8(path)).permute(0, 3, 1, 2).float() / 255.0


def align(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Common number of frames and, when the sizes differ, a common centre crop."""
    t = min(len(pred), len(gt))
    pred, gt = pred[:t], gt[:t]
    h, w = min(pred.shape[-2], gt.shape[-2]), min(pred.shape[-1], gt.shape[-1])

    def crop(x: torch.Tensor) -> torch.Tensor:
        top, left = (x.shape[-2] - h) // 2, (x.shape[-1] - w) // 2
        return x[..., top:top + h, left:left + w]

    return crop(pred), crop(gt)


# ---------------------------------------------------------------------------------------------------------------------------------
# what an input is
# ---------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    kind: str       # "frames" or "video"
    codec: str      # video codec, or the image format of a frame directory
    pix_fmt: str
    width: int
    height: int
    frames: int
    lossless: bool  # False also when it cannot be told (e.g. HEVC, VP9)
    detail: str     # encoder setting when known, e.g. "crf=18.0"

    @property
    def canonical(self) -> bool:
        return self.kind == "video" and self.codec == "h264" and self.pix_fmt == "yuv444p" and self.lossless

    def __str__(self) -> str:
        return " ".join(x for x in (self.codec, self.pix_fmt, self.detail) if x)


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _x264_options(path: Path) -> dict[str, str]:
    """The encoder settings libx264 writes into the stream (`rc=cqp qp=0` for lossless, `rc=crf crf=18.0` ...)."""
    with open(path, "rb") as f:
        head = f.read(4 << 20)
    i = head.find(b"x264 - core")
    if i < 0:
        return {}
    text = head[i:i + 2000].split(b"\x00")[0].decode("latin1")
    return dict(kv.split("=", 1) for kv in text.split() if "=" in kv)


@lru_cache(maxsize=None)
def probe(path: Path) -> Source:
    if path.is_dir():
        files = _frame_files(path)
        h, w = cv2.imread(str(files[0])).shape[:2]
        png = files[0].suffix.lower() == ".png" and all(f.suffix.lower() == ".png" for f in files)
        return Source("frames", "png" if png else "jpeg", "rgb24", w, h, len(files), png, "")
    text = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Video: (\w+).*?, (\w+)(?:\([^)]*\))?, (\d+)x(\d+)", text)
    if not m:
        raise ValueError(f"cannot read the video stream of {path}")
    codec, pix_fmt, w, h = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    lossless, detail = codec in LOSSLESS_CODECS, ""
    if codec == "h264":
        opts = _x264_options(path)
        lossless = opts.get("rc") == "cqp" and opts.get("qp") == "0"
        detail = "lossless" if lossless else next((f"{k}={opts[k]}" for k in ("crf", "qp", "bitrate") if k in opts), "")
    return Source("video", codec, pix_fmt, w, h, frames, lossless, detail)


def describe(path: Path) -> dict:
    """The record of an input in `results.json`."""
    return {**asdict(probe(path)), "canonical": probe(path).canonical}


# ---------------------------------------------------------------------------------------------------------------------------------
# the canonical format
# ---------------------------------------------------------------------------------------------------------------------------------
def _fingerprint(path: Path) -> str:
    files = _frame_files(path) if path.is_dir() else [path]
    stats = [f.stat() for f in files]
    return f"{path.resolve()}|{len(files)}|{sum(s.st_size for s in stats)}|{max(s.st_mtime_ns for s in stats)}"


def canonicalize(path: Path, cache_dir: Path) -> Path:
    """The file the metrics read for `path`: libx264, yuv444p, lossless.  An input of that kind is returned as it is, any other is decoded
    to RGB and encoded into `cache_dir` (re-encoded when the input changed)."""
    if probe(path).canonical:
        return path
    import imageio.v2 as imageio

    out = cache_dir / f"{path.stem if path.is_file() else path.name}.mp4"
    stamp = out.with_suffix(".json")
    fingerprint = _fingerprint(path)
    if out.exists() and stamp.exists() and json.loads(stamp.read_text()).get("source") == fingerprint:
        return out
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    with imageio.get_writer(str(tmp), format="FFMPEG", mode="I", fps=CANONICAL_FPS, codec="libx264", pixelformat="yuv444p",
                            macro_block_size=None, ffmpeg_params=["-crf", "0"]) as w:
        for frame in read_uint8(path):
            w.append_data(frame)
    os.replace(tmp, out)
    stamp.write_text(json.dumps({"source": fingerprint}) + "\n")
    return out
