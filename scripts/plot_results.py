#!/usr/bin/env python3
"""
Plot evaluation metrics and overlay bounding boxes for a run.
Usage:
    python scripts/plot_results.py --run-dir outputs/<timestamp>/<cfg_hash> [--sample-idx 0]
"""

import argparse
import json
from pathlib import Path

from nova_retrieval_vlm.visualization.plotting import plot_metrics
from nova_retrieval_vlm.visualization.plotting import plot_overlays


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics and overlays for a given run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory containing run.json, preds.jsonl, refs.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, help="Directory to save plots (default: <run-dir>/viz)"
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to overlay")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot metrics
    metrics_file = run_dir / "run.json"
    if metrics_file.exists():
        metrics = json.load(metrics_file.open())
        plot_metrics(metrics, out_dir / "metrics.png", title=f"Metrics: {run_dir.name}")
        print(f"Saved metrics plot to {out_dir / 'metrics.png'}")
    else:
        print(f"No metrics file at {metrics_file}")

    # Generate overlay for a sample
    try:
        plot_overlays(run_dir, out_dir, sample_idx=args.sample_idx)
        print(f"Saved overlay image for sample {args.sample_idx} to {out_dir}")
    except Exception as e:
        print(f"Could not generate overlays: {e}")


if __name__ == "__main__":
    main()
