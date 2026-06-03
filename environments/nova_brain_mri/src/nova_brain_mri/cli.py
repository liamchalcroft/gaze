"""CLI entry point for NOVA Brain MRI environment.

Provides medarc-eval compatible CLI for evaluating models on NOVA benchmark.

Usage:
    medarc-eval nova-brain-mri -m gpt-4o -n 100

    # Or directly:
    python -m nova_brain_mri.cli --model gpt-4o --num-examples 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from . import NOVABrainMRIEnv, load


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on NOVA Brain MRI benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., gpt-4o, claude-3-opus)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY/OPENROUTER_API_KEY env var)",
    )

    # Dataset arguments
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to NOVA dataset directory",
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["caption", "diagnosis", "localization", "all"],
        default="all",
        help="NOVA task to evaluate",
    )

    # Environment arguments
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for localization matching (0.5 = NOVA ACC50)",
    )

    # Output arguments
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Stream results as they complete",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print environment configuration schema and exit",
    )

    return parser.parse_args()


def get_api_client(args: argparse.Namespace) -> Any:
    """Create OpenAI-compatible API client."""
    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key
    base_url = args.base_url

    # Try environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: No API key provided. Set OPENAI_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Use OpenRouter if model has provider prefix
    if "/" in args.model and not base_url:
        base_url = "https://openrouter.ai/api/v1"
        # Prefer OPENROUTER_API_KEY when routing through OpenRouter
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            api_key = openrouter_key

    return OpenAI(api_key=api_key, base_url=base_url)


def print_env_schema() -> None:
    """Print environment configuration schema."""
    schema = {
        "type": "object",
        "properties": {
            "split": {
                "type": "string",
                "enum": ["train", "validation", "test"],
                "default": "test",
            },
            "task": {
                "type": "string",
                "enum": ["caption", "diagnosis", "localization", "all"],
                "default": "all",
            },
            "max_turns": {"type": "integer", "default": 10},
            "iou_threshold": {"type": "number", "default": 0.5},
            "data_dir": {"type": "string", "default": None},
        },
    }
    print(json.dumps(schema, indent=2))


def _score_completion(rubric: Any, completion: Any, info: dict[str, Any]) -> float:
    """Weighted score of a completion using a NOVA rubric's reward functions.

    Calls the rubric's individual reward functions with the
    ``(prompt, completion, info)`` keyword contract they declare, then combines
    them with the rubric weights.
    """
    funcs = rubric._get_individual_reward_funcs()
    weights = rubric._get_individual_reward_weights()
    total = 0.0
    for func, weight in zip(funcs, weights, strict=False):
        total += weight * float(func(prompt="", completion=completion, info=info))
    return total


async def run_evaluation(
    client: Any,
    model: str,
    env: NOVABrainMRIEnv,
    num_examples: int | None = None,
    verbose: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    """Run evaluation on the NOVA environment.

    Args:
        client: OpenAI-compatible client
        model: Model name
        env: NOVA environment instance
        num_examples: Number of examples (None for all)
        verbose: Print verbose output
        stream: Stream results

    Returns:
        Evaluation results dictionary
    """
    results = []
    rubric = env.rubric

    dataset = env.dataset
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    total = len(dataset)
    rewards = []

    for idx, example in enumerate(dataset):
        if verbose:
            print(f"Evaluating example {idx + 1}/{total}...")

        prompt = example["prompt"]
        info = example["info"]

        state: dict[str, Any] = {"turn": 0, "info": info, "is_complete": False}
        messages = list(prompt)

        while not await env.is_completed(state):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }
                messages.append(assistant_msg)

                env_msgs = await env.env_response(messages, state)
                messages.extend(env_msgs)

            except Exception as e:  # noqa: BLE001 - surface API/client errors per case
                if verbose:
                    print(f"  Error: {e}")
                break

        reward = _score_completion(rubric, messages, info)
        rewards.append(reward)

        result = {
            "case_index": idx,
            "reward": reward,
            "turns": state.get("turn", 0),
        }
        results.append(result)

        if stream:
            print(json.dumps(result))

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "model": model,
        "task": env.config.task,
        "split": env.split,
        "num_examples": len(results),
        "mean_reward": mean_reward,
        "results": results,
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Handle schema printing
    if args.schema:
        print_env_schema()
        return 0

    if not args.model:
        print("Error: --model is required (unless using --schema)", file=sys.stderr)
        return 1

    # Create client
    client = get_api_client(args)

    # Load environment
    env = load(
        split=args.split,
        task=args.task,
        max_turns=args.max_turns,
        iou_threshold=args.iou_threshold,
        data_dir=args.data_dir,
    )

    if args.verbose:
        print(f"Loaded NOVA environment: {len(env.dataset)} examples")
        print(f"Task: {args.task}, Max turns: {args.max_turns}")

    # Run evaluation
    results = asyncio.run(
        run_evaluation(
            client=client,
            model=args.model,
            env=env,
            num_examples=args.num_examples,
            verbose=args.verbose,
            stream=args.stream,
        )
    )

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if args.verbose:
            print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
