# vsreval

Evaluation bench for video super-resolution: eleven metrics behind one command, one environment and one JSON format.

| Metric | Kind | Implementation |
|---|---|---|
| `psnr` `ssim` `lpips` `dists` | full-reference, per frame | [pyiqa](https://github.com/chaofengc/IQA-PyTorch) |
| `musiq` `clipiqa` `niqe` `maniqa` | no-reference, per frame | pyiqa (`maniqa`: the KonIQ-10k checkpoint, pyiqa default) |
| `dover` | no-reference, video | DOVER (Wu et al., ICCV 2023), `DOVER.pth`, overall score |
| `fastervqa` | no-reference, video | FasterVQA (Wu et al., TPAMI 2023), `FAST_VQA_3D_1_1.pth` |
| `ewarp` | temporal, video | E\*warp: RAFT warping error with forward-backward occlusion mask, ×1000 |

Image metrics are the mean over the frames of a video; every metric is then averaged over the videos.

## Install and use

```bash
# inside a clone: one environment for all metrics
git clone https://github.com/Shundi-Wu/vsreval && cd vsreval && uv sync
uv run vsreval weights --download                     # DOVER / FasterVQA / RAFT weights -> ~/.cache/vsreval/weights ($VSREVAL_WEIGHTS)

uv run vsreval eval --pred results/UDM10 --gt datasets/UDM10/GT-Video --out results/UDM10/vsreval
uv run vsreval eval --pred results/UDM10 --metrics dover,fastervqa,ewarp          # no ground truth needed

# or as a command available everywhere, in its own isolated environment
OVERRIDES=https://raw.githubusercontent.com/Shundi-Wu/vsreval/main/overrides.txt
uv tool install git+https://github.com/Shundi-Wu/vsreval --overrides $OVERRIDES
uvx --from git+https://github.com/Shundi-Wu/vsreval --overrides $OVERRIDES vsreval eval ...   # without installing
```

`--overrides` is needed outside a clone: pyiqa requires `openai-clip`, whose package is empty and shadows the CLIP code (CLIP-IQA fails to
import), and `filterpy`, which does not build; no metric uses either. Inside a clone `pyproject.toml` applies the same overrides.

`--pred` / `--gt` are directories whose entries are video files (`.mp4 .mkv .avi .mov`) or directories of frames (`.png .jpg`), matched by
name. If the sizes of prediction and ground truth differ, both are cropped to the common centre region.

Python: `vsreval.evaluate(pred, gt, metrics=None, out="vsreval_out", device="cuda")`.

Output in `--out`: `<metric>.json` per metric (`per_sample`, `average`; skipped on the next run unless `--force`) and `results.json` with
all metrics, the protocol name (`vsreval-v1`), the library versions, `inputs` and `warnings`. A metric file is reused only when it was
computed under the same protocol for the same set of samples; otherwise it is recomputed.

## One input format for every metric

Whatever is passed in, every metric, image metrics included, reads the same kind of file, for the prediction and for the ground truth:
libx264 / yuv444p / lossless (`-crf 0`). An input is converted when it is not already that:

- a **frame directory** or a video of **any other codec or pixel format** is decoded to RGB and encoded to that format (into `<out>/_videos`);
- a video that **already is** libx264 yuv444p lossless is used as it is. So a directory of PNG frames and a lossless yuv444p mp4 saved from the
  same frames give bit-identical pixels, hence identical scores (the RGB to YUV rounding changes a pixel by at most one level);
- the **ground truth** goes through the same conversion, so both sides of a full-reference metric carry the same RGB to YUV rounding.
  Compared with reading the ground truth as it is, the difference is small (two models on three UDM10 videos: PSNR −0.01 dB, SSIM −0.002,
  LPIPS +0.002, DISTS +0.001, and the gap between the two models changes by at most 0.01 dB / 0.0007 LPIPS);
- the frame rate of the input does not matter (the metrics index frames, not time); the colour matrix tags of a stream are not interpreted
  (decoding is OpenCV's).

The RGB to YUV rounding is not neutral for CLIP-IQA: relative to lossless RGB frames it raised the score by 0.03 on average (per video
0.002 to 0.06) for two models on three UDM10 videos, while the other image metrics moved by amounts that are the same for both models.
CLIP-IQA scores computed on files of different formats are therefore not comparable.

**A lossy input stays lossy.** Converting a `yuv420p` / `crf 18` video does not bring back what the encoder removed, so its scores differ from
those of the lossless original (two UDM10 videos: PSNR about 0.4 dB, FasterVQA about 2 %). `vsreval` therefore reads what every input is
(codec, pixel format, x264 quantiser from the stream), prints a warning for each lossy kind, and stores the description of every prediction and
ground truth under `inputs` in `results.json` together with the `warnings`. Write predictions losslessly, or pass frames.

## Implementation notes

DOVER and FasterVQA are the authors' code, vendored and trimmed. Three things in it depend on library versions or on chance and are set
explicitly:

1. **DOVER resize.** DOVER's aesthetic branch resizes frames to 224×224 with `torchvision.transforms.Resize`, which anti-aliases tensors
   since torchvision 0.17 but did not before. With the new default the raw aesthetic score of a test video was 0.076 instead of 0.226.
   `antialias=False` is set, which gives the original DOVER numbers on current torchvision.
2. **Random fragment crops.** In DOVER's technical branch and in FasterVQA the crop position inside every grid cell is drawn with
   `torch.randint`, also at test time and without a seed. Scores therefore change from run to run. Here the global RNG is untouched and the
   crops use a fixed seed, so a video gets the same crops for every model and every run. Per-video standard deviation over seeds on two
   test videos: DOVER 0.004 to 0.007, FasterVQA 0.009 to 0.031; the average over ten videos is about √10 times smaller. Differences between
   models smaller than that are not meaningful.
3. **FasterVQA frame interval.** The repository's `vqa.py` builds `SampleFrames(clip_len, num_clips)` and so ignores `frame_interval` from the
   config; the same is done here.

E\*warp follows EvalCrafter's `compute_video_warping_error`; the compiled FlowNet2 `Resample2d` operator is replaced by the same bilinear,
border-clamped sampling in plain PyTorch (max difference to the compiled operator 7e-6).

## Validation

Two UDM10 outputs against the standalone scripts: PSNR, SSIM, LPIPS, DISTS, MUSIQ identical to four decimals, CLIP-IQA within 2e-4;
DOVER and FasterVQA within their sampling noise (DOVER 0.8464 vs 0.8502 and 0.7496 vs 0.7547; FasterVQA 0.7874 vs 0.7891 and 0.6777 vs
0.6765). `pytest tests` covers the resampling operator against the kernel's index arithmetic, the DOVER fusion, the FasterVQA frame
sampling, the input format conversion and the file handling.

## Third-party code and weights

`src/vsreval/_vendor` holds trimmed copies of DOVER, FAST-VQA and RAFT with their licences; see `_vendor/NOTICE.md` for sources, commits
and modifications. DOVER and FAST-VQA are under the S-Lab License 1.0 (non-commercial); keep the licence files when redistributing.
Weights are downloaded on first use and verified by SHA-256 (`weights.py`); they are not part of the repository.

## License

The code of this repository outside `src/vsreval/_vendor` is MIT licensed (`LICENSE`). The vendored DOVER and FAST-VQA code is under the
S-Lab License 1.0 (non-commercial) and RAFT under BSD-3-Clause; their licence files stay next to the code, so using `vsreval` for the
video metrics is subject to those terms. The model weights have their own licences.
