#!/usr/bin/env python
"""Evaluation script for GEMeX-ThinkVG.

Evaluates a trained model on the GEMeX dataset using verifiable rewards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import RewardWeights  # now exported from src/__init__
from src import load_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GEMeX-ThinkVG model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to GEMeX JSONL dataset file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (e.g., gpt-4o, or path to checkpoint)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default="0.4,0.3,0.3",
        help="Comma-separated weights for answer,location,bbox rewards",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )

    args = parser.parse_args()

    # Parse reward weights
    weights = [float(x) for x in args.reward_weights.split(",")]
    if len(weights) != 3:
        raise ValueError("--reward-weights must have 3 values")
    reward_weights = RewardWeights(answer=weights[0], location=weights[1], bbox=weights[2])

    # Load environment (pass reward_weights so the rubric uses them)
    env = load_environment(
        dataset_path=args.dataset,
        max_turns=args.max_turns,
        reward_weights=reward_weights,
    )

    # Sample dataset if requested
    dataset = env.dataset
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    print(f"Evaluating on {len(dataset)} samples with model {args.model}")

    # Evaluate (placeholder for actual evaluation loop)
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "num_samples": len(dataset),
        "max_turns": args.max_turns,
        "reward_weights": {
            "answer": reward_weights.answer,
            "location": reward_weights.location,
            "bbox": reward_weights.bbox,
        },
        # These would be populated by actual evaluation
        "metrics": {
            "mean_reward": 0.0,
            "answer_accuracy": 0.0,
            "location_accuracy": 0.0,
            "mean_iou": 0.0,
            "iou_50": 0.0,
            "iou_30": 0.0,
        },
        "sample_results": [],
    }

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"gemex_eval_{args.model.replace('/', '_')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")
    print("\nNote: Actual evaluation requires verifiers package integration")
    print("This script shows the evaluation structure.")
    print("\nTo run evaluation:")
    print("1. Ensure dependencies are installed (verifiers is core): `uv sync` or `pip install -e .`")
    print("2. Use verifiers evaluation loop with the loaded environment")
    print("3. Collect detailed results using GEMeXRewardFunction")


if __name__ == "__main__":
    main()
