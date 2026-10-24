# Vendored third-party code

Trimmed copies of the parts the metrics need, with the upstream licence next to each. All three upstream licences restrict use to
non-commercial purposes (S-Lab License 1.0 for DOVER and FAST-VQA; BSD-3-Clause for RAFT). Keep these files when redistributing.

| Directory | Upstream | Commit | Licence | What was kept / changed |
|---|---|---|---|---|
| `dover/` | https://github.com/VQAssessment/DOVER | `f1ddc96` (2024-08-12) | S-Lab License 1.0 (`dover/LICENSE`) | `models/{swin_backbone,conv_backbone,head}.py` and `datasets/dover_datasets.py` (sampling and view decomposition only). Removed: the evaluator wrappers, the dataset class, the `.yuv` reader (scikit-video), `thop`/`__main__` test blocks. `timm.models.layers` / `timm.models.registry` imports updated to `timm.layers` / `timm.models`. `Resize` in `get_resize_function` made `antialias=False` explicitly (the behaviour of torchvision < 0.17, which DOVER was built with). |
| `fastvqa/` | https://github.com/VQAssessment/FAST-VQA-and-FasterVQA | `8db452e` (2024-08-12) | S-Lab License 1.0 (`fastvqa/LICENSE`) | Only the test-mode frame sampling (`SampleFrames`). FasterVQA reuses DOVER's copy of the Swin-3D backbone and head, which are identical to FAST-VQA's apart from the default `adaptive_window_size`, passed explicitly in `metrics/fastervqa.py`. |
| `raft/` | https://github.com/princeton-vl/RAFT (as copied by EvalCrafter) | EvalCrafter `1275bda` (2024-10-03) | BSD-3-Clause (`raft/LICENSE`) | `core/{raft,update,extractor,corr,utils}.py` with relative imports. Removed: `ipdb`, scipy's `forward_interpolate`, the optional `alt_cuda_corr` import. |

Weights are not vendored; see `src/vsreval/weights.py`.
