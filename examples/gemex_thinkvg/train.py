#!/usr/bin/env python
"""Training script for GEMeX-ThinkVG RL fine-tuning.

Uses the radiant_harness verifiers integration for multi-turn training
with verifiable rewards. Supports both training and evaluation modes.

Example:
    # Training
    python train.py --dataset data/train.jsonl --model gpt-4o --mode train

    # Evaluation
    python train.py --dataset data/test.jsonl --model gpt-4o --mode eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import verifiers as vf

from src.processor import GEMeXProcessor
from src.rewards import RewardWeights


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


async def run_evaluation(
    processor: GEMeXProcessor,
    env_cls: type[vf.MultiTurnEnv],
    output_dir: Path,
) -> dict[str, float]:
    """Run evaluation using verifiers environment.

    Args:
        processor: Configured GEMeX processor
        env_cls: Verifiers environment class
        output_dir: Directory for saving results

    Returns:
        Dictionary of evaluation metrics
    """
    env = env_cls()
    print(f"Loaded {len(env.dataset)} evaluation cases")

    # Run evaluation episodes
    results = []
    total_reward = 0.0

    for i, (prompt, info) in enumerate(zip(env.dataset["prompt"], env.dataset["info"], strict=True)):
        print(f"Evaluating case {i + 1}/{len(env.dataset)}...")

        # Initialize episode
        state = env.build_initial_state(prompt, info)
        messages = list(prompt)  # Copy initial messages

        # Run episode (simplified - real impl would use model generation)
        while not await env.is_completed(messages, state, info):
            # In real training, this would call the model
            # For demo, we just show the structure
            break

        # Compute reward for final response (placeholder)
        # In real eval, completion would come from model
        reward = 0.0  # reward_fn(prompt, completion, info)
        total_reward += reward

        results.append({
            "case_id": i,
            "reward": reward,
            "turns": state.get("turn", 0),
        })

    # Aggregate metrics
    n = len(results)
    metrics = {
        "num_samples": n,
        "mean_reward": total_reward / n if n > 0 else 0.0,
        "mean_turns": sum(r["turns"] for r in results) / n if n > 0 else 0.0,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
    print(f"  Mean Turns: {metrics['mean_turns']:.2f}")
    print(f"  Results saved to: {output_dir / 'eval_results.json'}")

    return metrics


def run_training(
    processor: GEMeXProcessor,
    env_cls: type[vf.MultiTurnEnv],
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Run training using verifiers.

    Args:
        processor: Configured GEMeX processor
        env_cls: Verifiers environment class
        config: Training configuration
        output_dir: Directory for checkpoints
    """
    env = env_cls()
    print(f"Loaded {len(env.dataset)} training cases")

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Max Rollouts: {config['max_rollouts']}")
    print(f"  Reward Weights: answer={config['answer_weight']}, "
          f"location={config['location_weight']}, bbox={config['bbox_weight']}")

    print("\n" + "=" * 60)
    print("Training Setup Complete")
    print("=" * 60)
    print("""
To run actual training, integrate with your training framework:

    import verifiers as vf

    # Create trainer (example with hypothetical API)
    trainer = vf.Trainer(
        model=config['model'],
        environment=env,
        rubric=rubric,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
    )

    # Run training
    trainer.train(epochs=config['epochs'])

    # Save model
    trainer.save(output_dir / 'model')

For GRPO/PPO training, see verifiers documentation:
https://github.com/willccbb/verifiers
""")


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

    # Create processor with mixin
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

    # Create verifiers environment from processor
    image_base_path = Path(args.image_dir) if args.image_dir else None
    env_cls = processor.as_verifiers_env(
        max_turns=args.max_turns,
        cases=cases,
        image_base_path=image_base_path,
        model_name=args.model,
        use_tools=args.use_tools,
        use_web_search=args.use_web_search,
        reward_weights=reward_weights,
    )

    output_dir = Path(args.output)

    if args.mode == "eval":
        asyncio.run(run_evaluation(processor, env_cls, output_dir))
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
        run_training(processor, env_cls, config, output_dir)


if __name__ == "__main__":
    main()
