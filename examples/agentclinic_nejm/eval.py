#!/usr/bin/env python
"""Evaluation script for AgentClinic NEJM.

Evaluates a trained model on NEJM diagnostic reasoning cases.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src import load_environment

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate AgentClinic NEJM model")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to NEJM JSONL dataset file",
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
        default=10,
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
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )

    args = parser.parse_args()

    # Load environment
    env = load_environment(
        dataset_path=args.dataset,
        max_turns=args.max_turns,
    )

    # Sample dataset if requested
    dataset = env.dataset
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    logger.info("Evaluating on %d samples with model %s", len(dataset), args.model)

    # Evaluate (placeholder for actual evaluation loop)
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "num_samples": len(dataset),
        "max_turns": args.max_turns,
        # These would be populated by actual evaluation
        "metrics": {
            "accuracy": 0.0,
            "mean_turns": 0.0,
            "completion_rate": 0.0,
        },
        "sample_results": [],
    }

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"agentclinic_eval_{args.model.replace('/', '_')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", results_file)
    logger.info("Note: Actual evaluation requires verifiers package integration")
    logger.info("This script shows the evaluation structure.")
    logger.info("To run evaluation:")
    logger.info(
        "1. Ensure dependencies are installed (verifiers is core): uv sync or pip install -e ."
    )
    logger.info("2. Use verifiers evaluation loop with the loaded environment")
    logger.info("3. The environment includes the accuracy_reward function")


if __name__ == "__main__":
    main()
