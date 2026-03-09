#!/usr/bin/env python
"""Evaluate AgentClinic NEJM with a real multi-turn inference loop."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

from loguru import logger

from examples.agentclinic_nejm.src.environment import _brace_content
from examples.agentclinic_nejm.src.environment import accuracy_reward
from examples.agentclinic_nejm.src.environment import combined_reward
from examples.agentclinic_nejm.src.environment import load_environment
from radiant_harness import LMStudioAdapter
from radiant_harness import OpenAIAdapter
from radiant_harness import require_lmstudio_model
from radiant_harness.verifiers import TokenF1Reward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AgentClinic NEJM with Radiant Harness adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, help="Path to NEJM JSONL dataset file")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (OpenAI/OpenRouter format, or local ID for --base-url)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server (audit endpoint: http://192.168.1.138:1234/v1)",
    )
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum conversation turns")
    parser.add_argument(
        "--num-samples", type=int, default=-1, help="Number of samples (-1 for all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results"),
        help="Directory for summary output",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--reasoning", action="store_true", help="Enable reasoning mode for OpenAI/OpenRouter"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    return parser.parse_args()


def _requested_information(text: str) -> bool:
    """Match the environment's request-trigger keywords."""
    text_lower = text.lower()
    return bool(
        "history" in text_lower
        or "symptom" in text_lower
        or any(
            keyword in text_lower
            for keyword in [
                "exam",
                "physical",
                "test",
                "lab",
                "result",
                "imaging",
                "x-ray",
                "ct",
                "mri",
                "image",
                "photo",
                "picture",
            ]
        )
    )


def _aggregate_metrics(sample_results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-case evaluation metrics."""
    if not sample_results:
        return {
            "mean_reward": 0.0,
            "accuracy": 0.0,
            "token_f1": 0.0,
            "mean_turns": 0.0,
            "mean_tokens": 0.0,
            "requested_info_rate": 0.0,
            "diagnosis_completion_rate": 0.0,
        }

    n = len(sample_results)

    def _mean(key: str) -> float:
        return sum(float(sample[key]) for sample in sample_results) / n

    return {
        "mean_reward": _mean("reward"),
        "accuracy": _mean("accuracy"),
        "token_f1": _mean("token_f1"),
        "mean_turns": _mean("num_turns"),
        "mean_tokens": _mean("total_tokens"),
        "requested_info_rate": _mean("requested_info"),
        "diagnosis_completion_rate": _mean("completed"),
    }


async def _run_case(
    adapter: LMStudioAdapter | OpenAIAdapter,
    env: Any,
    prompt: list[dict[str, Any]],
    info: dict[str, Any],
    *,
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Run one AgentClinic case end-to-end."""
    messages = copy.deepcopy(prompt)
    state: dict[str, Any] = {"info": info}
    state = await env.setup_state(state)
    state["info"] = info

    total_tokens = 0
    turn_logs: list[dict[str, Any]] = []
    token_f1_reward = TokenF1Reward()
    answer_texts = [str(opt.get("text", "")) for opt in info.get("answers", [])]
    final_text = ""
    completed = False

    for turn_idx in range(max_turns):
        response_text, _, generation = await adapter.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=None,
            response_format=None,
        )
        total_tokens += generation.tokens
        final_text = response_text
        messages.append({"role": "assistant", "content": response_text})

        requested_info = _requested_information(response_text)
        if requested_info:
            state["asked"] = True

        completed = bool(state.get("asked", False) and _brace_content(response_text, answer_texts))
        turn_logs.append(
            {
                "turn": turn_idx + 1,
                "assistant": response_text,
                "requested_info": requested_info,
                "tokens": generation.tokens,
            }
        )
        if completed:
            break

        env_reply = await env.env_response(messages, state)
        messages.extend(env_reply)
        turn_logs[-1]["environment"] = env_reply

    accuracy = accuracy_reward("", final_text, info)
    token_f1 = token_f1_reward("", final_text, {"gold": info.get("gold", "")})
    reward = combined_reward("", final_text, info)

    return {
        "final_text": final_text,
        "prediction": _brace_content(final_text, answer_texts) or final_text,
        "gold": info.get("gold", ""),
        "reward": reward,
        "accuracy": accuracy,
        "token_f1": token_f1,
        "num_turns": len(turn_logs),
        "total_tokens": total_tokens,
        "requested_info": 1.0 if state.get("asked", False) else 0.0,
        "completed": 1.0 if completed else 0.0,
        "turn_log": turn_logs,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    loaded_models: list[str] | None = None
    if args.base_url is not None:
        loaded_models = await require_lmstudio_model(model_name=args.model, base_url=args.base_url)
        logger.info(f"LM Studio ready at {args.base_url} with models: {loaded_models}")
        adapter: LMStudioAdapter | OpenAIAdapter = LMStudioAdapter(
            model_name=args.model,
            base_url=args.base_url,
        )
    else:
        adapter = OpenAIAdapter(
            model_name=args.model,
            reasoning_enabled=args.reasoning,
        )

    env = load_environment(dataset_path=args.dataset, max_turns=args.max_turns)
    dataset = env.dataset
    if 0 < args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))

    logger.info(f"Loaded {len(dataset)} samples for model {args.model}")

    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        prompts = dataset["prompt"]
        infos = dataset["info"]
        for idx, (prompt, info) in enumerate(zip(prompts, infos, strict=True)):
            logger.info(f"Processing sample {idx + 1}/{len(dataset)}")
            try:
                result = await _run_case(
                    adapter,
                    env,
                    prompt,
                    info,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                sample_results.append({"sample_id": idx, **result})
            except Exception as exc:
                logger.error(f"Sample {idx} failed: {exc}")
                failures.append(
                    {
                        "sample_id": idx,
                        "question": info.get("question", ""),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        await adapter.aclose()

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "base_url": args.base_url,
        "lmstudio_models": loaded_models,
        "max_turns": args.max_turns,
        "num_samples_total": len(dataset),
        "num_samples_evaluated": len(sample_results),
        "num_failures": len(failures),
        "metrics": _aggregate_metrics(sample_results),
        "failures": failures,
        "sample_results": sample_results,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / f"agentclinic_eval_{args.model.replace('/', '_')}.json"
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    logger.info(f"Results saved to {results_file}")
    return results


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logger.enable("radiant_harness")
        logger.enable("examples.agentclinic_nejm")
    else:
        logger.disable("examples.agentclinic_nejm")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
