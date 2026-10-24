"""Model weights: resolved from a local cache, downloaded (and checksummed) on first use.

The cache directory is `$VSREVAL_WEIGHTS` or `~/.cache/vsreval/weights`.  Weights are never part of the package or the repository.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Weight:
    filename: str
    sha256: str
    urls: tuple[str, ...]
    archive_member: str | None = None  # the URL points to a zip; extract this member


WEIGHTS: dict[str, Weight] = {
    "dover": Weight(
        "DOVER.pth", "f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838",
        ("https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth",
         "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth")),
    "fastervqa": Weight(
        "FAST_VQA_3D_1_1.pth", "8c3108647653fd48e31f3bebbe03a344d624c806d3f1af9478a4e9f5aa3038ab",
        ("https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth",)),
    "raft": Weight(
        "raft-things.pth", "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
        ("https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",), archive_member="raft-things.pth"),
}


def weights_dir() -> Path:
    return Path(os.environ.get("VSREVAL_WEIGHTS", Path.home() / ".cache" / "vsreval" / "weights"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dst)


def ensure(name: str) -> Path:
    """Path of the weight file `name`; downloads and verifies it when it is not cached yet."""
    w = WEIGHTS[name]
    path = weights_dir() / w.filename
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in w.urls:
        try:
            print(f"[vsreval] downloading {w.filename} from {url}", flush=True)
            if w.archive_member:
                archive = path.with_suffix(".zip")
                _download(url, archive)
                with zipfile.ZipFile(archive) as z:
                    member = next(n for n in z.namelist() if n.endswith(w.archive_member))
                    with z.open(member) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                archive.unlink()
            else:
                _download(url, path)
            if sha256(path) != w.sha256:
                path.unlink()
                raise RuntimeError("checksum mismatch")
            return path
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"could not obtain {w.filename}:\n  " + "\n  ".join(errors) + f"\nPlace the file in {path.parent} by hand.")


def verify() -> dict[str, bool]:
    """Checksum of every cached weight file (False = missing or corrupt)."""
    return {k: (weights_dir() / w.filename).exists() and sha256(weights_dir() / w.filename) == w.sha256 for k, w in WEIGHTS.items()}
