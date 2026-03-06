#!/usr/bin/env python
"""Evaluation script for AgentClinic NEJM.

Loads the environment and dataset for evaluation. Wire the environment
into a verifiers evaluation loop to run actual inference.
See https://github.com/primeintellect-ai/verifiers for API details.
"""

from __future__ import annotations

import argparse
import logging

from src import load_environment

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate AgentClinic NEJM model")
    parser.add_argument("--dataset", type=str, help="Path to NEJM JSONL dataset file")
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument(
        "--num-samples", type=int, default=-1, help="Number of samples (-1 for all)"
    )
    args = parser.parse_args()

    env = load_environment(dataset_path=args.dataset, max_turns=args.max_turns)

    dataset = env.dataset
    if 0 < args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    logger.info("Loaded %d samples for model %s", len(dataset), args.model)
    logger.info(
        "To evaluate, pass this environment to a verifiers evaluation loop. "
        "See https://github.com/primeintellect-ai/verifiers for API details."
    )


if __name__ == "__main__":
    main()
