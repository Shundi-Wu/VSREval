import json

import cv2
import imageio.v2 as imageio
import numpy as np

from vsreval.runner import PROTOCOL, evaluate
from vsreval.video import canonicalize, probe, read_frames, read_uint8


def _frames(n=4, h=32, w=48, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8)


def _write_dir(directory, frames):
    directory.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(directory / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return directory


def _write_video(path, frames, pixelformat, *params, codec="libx264"):
    with imageio.get_writer(str(path), format="FFMPEG", mode="I", fps=8, codec=codec, pixelformat=pixelformat,
                            macro_block_size=None, ffmpeg_params=list(params)) as w:
        for f in frames:
            w.append_data(f)
    return path


def test_probe_tells_what_an_input_is(tmp_path):
    frames = _frames()
    lossy = probe(_write_video(tmp_path / "lossy.mp4", frames, "yuv420p", "-crf", "18"))
    assert (lossy.codec, lossy.pix_fmt, lossy.lossless, lossy.canonical) == ("h264", "yuv420p", False, False)
    assert lossy.detail == "crf=18.0" and (lossy.width, lossy.height, lossy.frames) == (48, 32, 4)
    # the same pixel format is still not canonical when it is lossy
    assert not probe(_write_video(tmp_path / "lossy444.mp4", frames, "yuv444p", "-crf", "18")).canonical
    canonical = probe(_write_video(tmp_path / "canonical.mp4", frames, "yuv444p", "-crf", "0"))
    assert canonical.canonical and canonical.lossless
    png = probe(_write_dir(tmp_path / "png", frames))
    assert (png.kind, png.codec, png.lossless, png.frames) == ("frames", "png", True, 4)


def test_canonical_files_are_used_as_they_are(tmp_path):
    canonical = _write_video(tmp_path / "canonical.mp4", _frames(), "yuv444p", "-crf", "0")
    assert canonicalize(canonical, tmp_path / "cache") == canonical
    assert not (tmp_path / "cache").exists()


def test_a_frame_directory_and_the_video_saved_from_it_give_the_same_pixels(tmp_path):
    frames = _frames()
    from_dir = read_frames(canonicalize(_write_dir(tmp_path / "frames", frames), tmp_path / "cache"))
    # the same frames saved straight to the canonical format
    canonical = _write_video(tmp_path / "canonical.mp4", frames, "yuv444p", "-crf", "0")
    assert (from_dir == read_frames(canonicalize(canonical, tmp_path / "cache"))).all()
    assert abs(from_dir * 255 - torch_frames(frames)).max() <= 2  # the RGB to YUV rounding (white noise is its worst case)


def torch_frames(frames):
    import torch

    return torch.from_numpy(frames).permute(0, 3, 1, 2).float()


def test_lossless_rgb_input_is_converted_to_the_canonical_format(tmp_path):
    frames = _frames()
    rgb = _write_video(tmp_path / "rgb.mkv", frames, "rgb24", "-qp", "0", codec="libx264rgb")
    assert probe(rgb).lossless and not probe(rgb).canonical
    out = canonicalize(rgb, tmp_path / "cache")
    assert probe(out).canonical
    assert np.abs(read_uint8(out).astype(int) - frames.astype(int)).max() <= 2


def test_the_converted_file_follows_its_source(tmp_path):
    src = _write_dir(tmp_path / "frames", _frames(seed=0))
    first = read_uint8(canonicalize(src, tmp_path / "cache")).copy()
    _write_dir(src, _frames(seed=1))  # same name, new content
    second = read_uint8(canonicalize(src, tmp_path / "cache"))
    assert np.abs(first.astype(int) - second.astype(int)).max() > 50


def test_evaluate_records_inputs_and_recomputes_outdated_results(tmp_path, capsys):
    frames = _frames()
    _write_dir(tmp_path / "pred" / "a", frames)
    _write_dir(tmp_path / "gt" / "a", _frames(seed=1))
    _write_video(tmp_path / "pred" / "b.mp4", frames, "yuv420p", "-crf", "18")
    _write_dir(tmp_path / "gt" / "b", _frames(seed=2))
    out = tmp_path / "out"
    results = evaluate(tmp_path / "pred", tmp_path / "gt", ["psnr"], out, device="cpu")
    assert results["protocol"] == PROTOCOL
    assert results["inputs"]["a"]["pred"]["lossless"] and not results["inputs"]["b"]["pred"]["lossless"]
    assert len(results["warnings"]) == 1 and "1 of 2 predictions are lossy" in results["warnings"][0]
    assert "warning" in capsys.readouterr().out

    stored = json.loads((out / "psnr.json").read_text())
    assert stored["protocol"] == PROTOCOL
    stored.update(protocol="an-earlier-protocol", average=-1.0)  # a result of an earlier protocol must not be reused
    (out / "psnr.json").write_text(json.dumps(stored))
    assert evaluate(tmp_path / "pred", tmp_path / "gt", ["psnr"], out, device="cpu")["average"]["psnr"] > 0
