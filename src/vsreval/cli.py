"""Command line: `vsreval eval | list | weights`."""
from __future__ import annotations

import argparse

from . import __version__
from .metrics import DEFAULT_METRICS, NEEDS_GT


def _metric_list(text: str) -> list[str]:
    return DEFAULT_METRICS if text == "all" else [m.strip() for m in text.split(",") if m.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vsreval", description="Evaluation bench for video super-resolution.")
    parser.add_argument("--version", action="version", version=f"vsreval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="score a directory of predictions")
    ev.add_argument("--pred", required=True, help="directory of predicted videos or frame directories")
    ev.add_argument("--gt", help="directory of ground-truth videos or frame directories, matched by name (needed for psnr/ssim/lpips/dists)")
    ev.add_argument("--metrics", type=_metric_list, default="all", help="comma-separated list, or 'all' (default)")
    ev.add_argument("--out", default="vsreval_out", help="output directory (default: vsreval_out)")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--force", action="store_true", help="recompute metrics whose result file already exists")
    ev.add_argument("--limit", type=int, help="only the first N samples")

    sub.add_parser("list", help="list the available metrics")
    wt = sub.add_parser("weights", help="show the state of the weight cache")
    wt.add_argument("--download", action="store_true", help="download the missing weights")

    args = parser.parse_args(argv)
    if args.command == "list":
        for m in DEFAULT_METRICS:
            print(f"{m:10s} {'full-reference' if m in NEEDS_GT else 'no-reference' if m not in ('dover', 'fastervqa', 'ewarp') else 'video'}")
    elif args.command == "weights":
        from . import weights

        if args.download:
            for name in weights.WEIGHTS:
                weights.ensure(name)
        print(f"cache: {weights.weights_dir()}")
        for name, ok in weights.verify().items():
            print(f"{name:10s} {'ok' if ok else 'missing or corrupt'}")
    else:
        import json

        from .runner import evaluate

        try:
            results = evaluate(args.pred, args.gt, args.metrics, args.out, args.device, args.force, args.limit)
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise SystemExit(f"vsreval: {e}")
        print(json.dumps({"count": results["count"], "average": results["average"]}, indent=2))


if __name__ == "__main__":
    main()
