"""Evaluating a directory of predictions: models are loaded once, every sample is read once, results are cached per metric."""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import __version__
from .metrics import DEFAULT_METRICS, NEEDS_GT, create
from .metrics.base import SampleData
from .video import describe, find_samples

PROTOCOL = "vsreval-v1"  # every input is converted to libx264 / yuv444p / lossless before any metric reads it


def _versions() -> dict[str, str]:
    out = {"vsreval": __version__}
    for pkg in ("torch", "numpy", "decord", "pyiqa", "opencv-python-headless", "timm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _inputs(samples) -> tuple[dict, list[str]]:
    """What every prediction / ground truth is, and a warning per kind of lossy input."""
    record, lossy = {}, {}
    for s in samples:
        record[s.name] = {role: describe(path) for role, path in (("pred", s.pred), ("gt", s.gt)) if path is not None}
        for role, info in record[s.name].items():
            if not info["lossless"]:
                lossy.setdefault((role, info["codec"], info["pix_fmt"], info["detail"]), []).append(s.name)
    warnings = [f"{len(names)} of {len(samples)} {'predictions' if role == 'pred' else 'ground-truth videos'} are lossy "
                f"({' '.join(x for x in (codec, pix_fmt, detail) if x)}): scores are computed on the lossy pixels and are not comparable "
                f"with scores of lossless inputs" for (role, codec, pix_fmt, detail), names in lossy.items()]
    return record, warnings


def _stored(path: Path, names: set[str]) -> dict | None:
    """The result file of a metric if it was computed by this protocol for exactly these samples, else None."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("protocol") == PROTOCOL and set(data.get("per_sample", {})) == names else None


def evaluate(pred: str | Path, gt: str | Path | None = None, metrics: list[str] | None = None, out: str | Path = "vsreval_out",
             device: str = "cuda", force: bool = False, limit: int | None = None) -> dict:
    """Score every video / frame directory of `pred`.  Writes `<out>/<metric>.json` per metric (skipped when it was computed
    by the current protocol for the same samples, unless `force`) and `<out>/results.json` with all metrics; returns the latter."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = find_samples(pred, gt)[:limit]
    names = metrics or DEFAULT_METRICS
    if gt is None:
        skipped = [m for m in names if m in NEEDS_GT]
        names = [m for m in names if m not in NEEDS_GT]
        if skipped:
            print(f"[vsreval] no --gt: skipping the full-reference metrics {', '.join(skipped)}")
    inputs, warnings = _inputs(samples)
    for w in warnings:
        print(f"[vsreval] warning: {w}")
    sample_names = {s.name for s in samples}
    todo = [m for m in names if force or _stored(out / f"{m}.json", sample_names) is None]

    if todo:
        dev = torch.device(device)
        models = {m: create(m, dev) for m in todo}
        scores: dict[str, dict[str, float]] = {m: {} for m in todo}
        for sample in tqdm(samples, desc="samples"):
            data = SampleData(sample, out / "_videos")
            for m, model in models.items():
                scores[m][sample.name] = float(model(data))
            del data
            torch.cuda.empty_cache()
        for m, per_sample in scores.items():
            payload = {"metric": m, "protocol": PROTOCOL, "per_sample": per_sample,
                       "average": float(np.mean(list(per_sample.values()))), "count": len(per_sample)}
            (out / f"{m}.json").write_text(json.dumps(payload, indent=2) + "\n")

    per_metric = {m: json.loads((out / f"{m}.json").read_text()) for m in names}
    results = {
        "protocol": PROTOCOL,
        "versions": _versions(),
        "count": len(samples),
        "inputs": inputs,
        "warnings": warnings,
        "average": {m: d["average"] for m, d in per_metric.items()},
        "per_sample": {s.name: {m: d["per_sample"][s.name] for m, d in per_metric.items() if s.name in d["per_sample"]} for s in samples},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
