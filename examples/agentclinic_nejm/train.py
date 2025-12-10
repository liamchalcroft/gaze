#!/usr/bin/env python
"""Training script for AgentClinic NEJM RL fine-tuning.

Uses the verifiers package for multi-turn diagnostic reasoning training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import load_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgentClinic NEJM model")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to NEJM JSONL dataset file",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for training (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=4,
        help="Maximum rollouts per sample",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100,
        help="Evaluation frequency (steps)",
    )

    args = parser.parse_args()

    # Load environment
    env = load_environment(
        dataset_path=args.dataset,
        max_turns=args.max_turns,
    )

    print(f"Loaded environment with {len(env.dataset)} samples")

    # Configure training
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

    # Save config
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Training config: {json.dumps(config, indent=2)}")
    print("\nNote: Actual training implementation requires verifiers package integration")
    print("This script shows the configuration structure for training.")
    print("\nTo run training:")
    print("1. Ensure dependencies are installed (verifiers is core): `uv sync` or `pip install -e .`")
    print("2. Set up your model with the verifiers training loop")
    print("3. Use the loaded environment and accuracy reward function")


if __name__ == "__main__":
    main()
