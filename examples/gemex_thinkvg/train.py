#!/usr/bin/env python
"""Prepare GEMeX-ThinkVG training config or dispatch to the real evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.gemex_thinkvg.eval import run_evaluation as run_gemex_eval
from examples.gemex_thinkvg.src.processor import GEMeXProcessor
from examples.gemex_thinkvg.src.rewards import RewardWeights
from examples.gemex_thinkvg.src.verifiers import GEMeXThinkVGToolEnv


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train/evaluate GEMeX-ThinkVG model with verifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
        help="Run mode: train or eval",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to GEMeX JSONL dataset file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Base directory for resolving image paths",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name (OpenRouter format)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server when running --mode eval",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Enable visual manipulation tools",
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable web search tools",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate in --mode eval (-1 for all)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable reasoning mode when using OpenAI/OpenRouter for --mode eval",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs for --mode eval",
    )

    # Reward
    parser.add_argument(
        "--answer-weight",
        type=float,
        default=0.4,
        help="Weight for answer reward component",
    )
    parser.add_argument(
        "--location-weight",
        type=float,
        default=0.3,
        help="Weight for location reward component",
    )
    parser.add_argument(
        "--bbox-weight",
        type=float,
        default=0.3,
        help="Weight for bounding box reward component",
    )

    # Training-specific
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/gemex",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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

    return parser.parse_args()


def load_cases(dataset_path: str) -> list[dict[str, Any]]:
    """Load cases from JSONL file."""
    cases = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_training(
    processor: GEMeXProcessor,
    env: GEMeXThinkVGToolEnv,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Prepare training config and environment details for verifiers.

    Args:
        processor: Configured GEMeX processor
        env: Configured GEMeX environment
        config: Training configuration
        output_dir: Directory for checkpoints
    """
    print(f"Loaded {len(env.dataset)} training cases")

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nTraining Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Max Rollouts: {config['max_rollouts']}")
    print(
        f"  Reward Weights: answer={config['answer_weight']}, "
        f"location={config['location_weight']}, bbox={config['bbox_weight']}"
    )

    print("\nTraining config written.")
    print("Use the saved config and the prepared environment with your verifiers trainer.")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate reward weights
    weights_sum = args.answer_weight + args.location_weight + args.bbox_weight
    if abs(weights_sum - 1.0) > 1e-6:
        raise ValueError(f"Reward weights must sum to 1.0, got {weights_sum}")

    reward_weights = RewardWeights(
        answer=args.answer_weight,
        location=args.location_weight,
        bbox=args.bbox_weight,
    )

    # Create processor (still useful for reward function introspection)
    processor = GEMeXProcessor(
        model_name=args.model,
        use_tools=args.use_tools,
        use_web_search=args.use_web_search,
        max_turns=args.max_turns,
        reward_weights=reward_weights,
    )

    # Load cases
    cases = load_cases(args.dataset)
    print(f"Loaded {len(cases)} cases from {args.dataset}")

    # Create GEMeXThinkVGToolEnv directly — same env class used by eval.py
    # to guarantee training/evaluation parity.
    env = GEMeXThinkVGToolEnv(
        cases=cases,
        max_turns=args.max_turns,
        reward_weights=reward_weights,
    )

    output_dir = Path(args.output)

    if args.mode == "eval":
        asyncio.run(
            run_gemex_eval(
                SimpleNamespace(
                    dataset=args.dataset,
                    image_dir=Path(args.image_dir) if args.image_dir else None,
                    model=args.model,
                    base_url=args.base_url,
                    mode="single_turn" if args.max_turns == 1 else "agentic",
                    max_turns=args.max_turns,
                    use_tools=args.use_tools,
                    use_web_search=args.use_web_search,
                    num_samples=args.num_samples,
                    output=output_dir,
                    reward_weights=(
                        f"{args.answer_weight},{args.location_weight},{args.bbox_weight}"
                    ),
                    reasoning=args.reasoning,
                    verbose=args.verbose,
                )
            )
        )
    else:
        config = {
            "model": args.model,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_rollouts": args.max_rollouts,
            "max_turns": args.max_turns,
            "answer_weight": args.answer_weight,
            "location_weight": args.location_weight,
            "bbox_weight": args.bbox_weight,
        }
        run_training(processor, env, config, output_dir)


if __name__ == "__main__":
    main()
