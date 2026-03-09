#!/usr/bin/env python
"""Prepare AgentClinic NEJM training configuration for verifiers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from examples.agentclinic_nejm.src import load_environment

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train AgentClinic NEJM model")
    parser.add_argument("--dataset", type=str, help="Path to NEJM JSONL dataset file")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument("--model", type=str, required=True, help="Model name for training")
    parser.add_argument("--output", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-rollouts", type=int, default=4, help="Rollouts per sample")
    parser.add_argument("--eval-freq", type=int, default=100, help="Eval frequency (steps)")
    args = parser.parse_args()

    env = load_environment(dataset_path=args.dataset, max_turns=args.max_turns)
    logger.info("Loaded environment with %d samples", len(env.dataset))

    config = {
        "model": args.model,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_rollouts": args.max_rollouts,
        "max_turns": args.max_turns,
        "eval_freq": args.eval_freq,
        "output_dir": args.output,
    }

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Config saved to %s/config.json", args.output)
    logger.info("Environment and config prepared. Pass them to your verifiers training loop.")


if __name__ == "__main__":
    main()
