"""Export NOVA evaluation runs to Hugging Face Agent Traces format.

Converts sample_*.json outputs from the NOVA CLI into Claude Code JSONL
sessions enriched with per-sample evaluation metadata. The resulting
directory is ready for ``huggingface-cli upload``.

Usage:
    python -m examples.nova.src.export_hf_traces \
        --run-dir ./runs/gpt-4o__agentic__turns10 \
        --output-dir ./hf_traces_export

The script re-computes per-sample evaluation metrics so each trace is
self-contained. Diagnosis semantic matching re-uses cached judgment logs
when available to avoid redundant LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running from repo root without install
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from beartype import beartype

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_model_name(name: str) -> str:
    """Sanitise model name for use in filenames."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).strip("_")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _build_claude_code_jsonl(
    sample: dict[str, Any],
    session_id: str,
    eval_metrics: dict[str, Any],
    global_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert a single NOVA sample into a list of Claude Code JSONL events."""
    turns = sample.get("turns", [])
    metadata = sample.get("metadata", {})
    ground_truth = sample.get("ground_truth", {})
    run_config = sample.get("run_config", {})
    response = sample.get("response", {})

    # Session-level metadata attached to the first event
    nova_metadata: dict[str, Any] = {
        "trace_id": session_id,
        "dataset_split": metadata.get("split", "unknown"),
        "case_id": metadata.get("image_id", sample.get("sample_id", "unknown")),
        "image_ids": [metadata.get("image_id", "")],
        "image_source": "gaze-vlm/nova-brain-mri",
        "clinical_history": metadata.get("clinical_history", ""),
        "task_type": global_config.get("task", "all"),
        "model_config": {
            "model_name": run_config.get("model_name", global_config.get("model", "unknown")),
            "adapter": global_config.get("adapter", "openai"),
            "max_turns": run_config.get("max_turns", global_config.get("max_turns", 10)),
            "max_tokens": run_config.get("max_tokens", global_config.get("max_tokens")),
            "temperature": run_config.get("temperature", 0.0),
            "seed": run_config.get("seed", global_config.get("seed")),
            "use_tools": global_config.get("use_tools", True),
            "use_web_search": global_config.get("use_web_search", True),
            "disabled_tools": [],
            "reasoning_enabled": global_config.get("reasoning_enabled", False),
            "reasoning_effort": global_config.get("reasoning_effort", "high"),
        },
        "outcome": {
            "correct_caption": bool(eval_metrics.get("caption", {}).get("bleu", 0.0) > 0.3),
            "correct_diagnosis_top1": bool(eval_metrics.get("diagnosis", {}).get("top1", 0)),
            "correct_diagnosis_top5": bool(eval_metrics.get("diagnosis", {}).get("top5", 0)),
            "correct_localization_iou50": bool(
                eval_metrics.get("localization", {}).get("acc50", 0)
            ),
            "stopped_early": sample.get("num_turns", 0) < run_config.get("max_turns", 10),
            "nudges_required": 0,
            "force_finalized": False,
        },
        "evaluation": eval_metrics,
        "aggregate_stats": {
            "num_turns": sample.get("num_turns", 0),
            "num_tool_calls": sum(len(t.get("tool_calls", [])) for t in turns),
            "tools_used": sample.get("tools_used", []),
            "total_tokens": sample.get("total_tokens", 0),
            "input_tokens": sample.get("total_tokens", 0),
            "output_tokens": 0,
            "total_latency_ms": 0,
            "confidence": sample.get("confidence", 0.0),
        },
        "ground_truth": {
            "caption": ground_truth.get("caption", ""),
            "diagnosis": ground_truth.get("final_diagnosis", ""),
            "localizations": ground_truth.get("localizations", []),
        },
        "final_response": dict(response)
        if isinstance(response, (dict, type({}.keys().__class__)))
        else {},
    }

    events: list[dict[str, Any]] = []
    parent_uuid: str | None = None

    # Initial user turn with metadata
    user_content = metadata.get("clinical_history", "")
    first_event: dict[str, Any] = {
        "type": "user",
        "message": {"role": "user", "content": user_content or "Analyze this brain MRI."},
        "uuid": f"{session_id}::user::0",
        "parentUuid": None,
        "sessionId": session_id,
        "timestamp": metadata.get("timestamp", ""),
        "nova_metadata": nova_metadata,
    }
    events.append(first_event)
    parent_uuid = first_event["uuid"]

    # Convert each NOVA turn to assistant / tool_result events
    for turn_idx, turn in enumerate(turns):
        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])
        tool_results = turn.get("tool_results", [])

        turn_meta: dict[str, Any] = {
            "turn_index": turn_idx,
            "turn_type": "tool_planning" if tool_calls else "reasoning",
            "tool_calls": [
                {
                    "id": tc.get("id", f"call_{turn_idx}_{i}"),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                }
                for i, tc in enumerate(tool_calls)
            ],
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": run_config.get("model_name", global_config.get("model", "")),
            "stop_reason": "tool_calls" if tool_calls else "end_turn",
            "continue_flag": turn_idx < len(turns) - 1,
            "coord_space_modified": any(
                tc.get("name", "") in {"crop", "zoom", "rotate", "flip_horizontal", "flip_vertical"}
                for tc in tool_calls
            ),
            "intensity_space_modified": any(
                tc.get("name", "")
                in {
                    "threshold",
                    "window_level",
                    "equalize_histogram",
                    "adaptive_equalize",
                    "invert",
                    "detect_edges",
                    "symmetry_diff",
                    "morphological",
                    "denoise",
                    "adjust_contrast",
                    "adjust_brightness",
                    "adjust_sharpness",
                }
                for tc in tool_calls
            ),
        }

        assistant_event: dict[str, Any] = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": run_config.get("model_name", global_config.get("model", "")),
                "content": [{"type": "text", "text": content}],
            },
            "uuid": f"{session_id}::assistant::{turn_idx}",
            "parentUuid": parent_uuid,
            "sessionId": session_id,
            "timestamp": "",
            "nova_turn_meta": turn_meta,
        }
        if tool_calls:
            assistant_event["message"]["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{turn_idx}_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {}))
                        if isinstance(tc.get("arguments"), dict)
                        else str(tc.get("arguments", "")),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
        events.append(assistant_event)
        parent_uuid = assistant_event["uuid"]

        # Tool result turn(s)
        for tr_idx, tr in enumerate(tool_results):
            tr_meta: dict[str, Any] = {
                "turn_index": turn_idx,
                "turn_type": "tool_execution",
                "tool_results": [
                    {
                        "tool_call_id": tr_idx,
                        "tool_name": tr.get("tool_name", ""),
                        "success": tr.get("success", True),
                        "produced_image": tr.get("produced_image", False),
                        "description_preview": tr.get("description", "")[:200],
                        "error": tr.get("error"),
                    }
                ],
                "execution_latency_ms": 0,
            }
            tool_event: dict[str, Any] = {
                "type": "tool_result",
                "message": {
                    "role": "tool",
                    "tool_call_id": tr_idx,
                    "content": tr.get("description", ""),
                },
                "uuid": f"{session_id}::tool::{turn_idx}::{tr_idx}",
                "parentUuid": parent_uuid,
                "sessionId": session_id,
                "timestamp": "",
                "nova_turn_meta": tr_meta,
            }
            events.append(tool_event)
            parent_uuid = tool_event["uuid"]

    return events


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


@beartype
def _eval_caption_sample(pred: str, ref: str) -> dict[str, Any]:
    from examples.nova.src.evaluation.caption import evaluate_caption

    return evaluate_caption([pred], [ref])


@beartype
async def _eval_diagnosis_sample(
    pred: list[str],
    ref: str,
    judge_model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    from examples.nova.src.evaluation.diagnosis import evaluate_diagnosis_nova_official

    kwargs: dict[str, Any] = {"seed": seed}
    if judge_model:
        kwargs["model_name"] = judge_model
    return await evaluate_diagnosis_nova_official([pred], [ref], **kwargs)


@beartype
def _eval_localization_sample(
    pred_boxes: dict[str, Any],
    ref_boxes: dict[str, Any],
) -> dict[str, Any]:
    from examples.nova.src.evaluation.detection import evaluate_detection

    return evaluate_detection([pred_boxes], [ref_boxes])


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportConfig:
    run_dir: Path
    output_dir: Path
    judge_model: str | None = None
    seed: int | None = None


async def export_run(config: ExportConfig) -> None:
    """Convert a single NOVA run directory to HF traces."""
    run_dir = config.run_dir
    out_dir = config.output_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")

    with summary_path.open() as f:
        summary = json.load(f)

    global_config: dict[str, Any] = summary.get("config", {})
    model_name = global_config.get("model", "unknown")
    safe_name = _safe_model_name(model_name)
    mode = global_config.get("mode", "agentic")
    max_turns = global_config.get("max_turns", 10)
    use_tools = global_config.get("use_tools", True)

    # Collect sample files
    sample_files = sorted(run_dir.glob("sample_*.json"))
    if not sample_files:
        raise ValueError(f"No sample_*.json files found in {run_dir}")

    # Guard against legacy runs that did not persist turns
    _first = json.loads(sample_files[0].read_text())
    if "turns" not in _first:
        raise ValueError(
            "Legacy run detected: sample files lack 'turns' field. "
            "Re-run evaluation with the updated CLI, or pass --legacy to skip trace generation."
        )

    print(f"Found {len(sample_files)} samples in {run_dir}")

    # Prepare traces subdirectory
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema": "claude_code_jsonl",
        "source_run": str(run_dir),
        "model": model_name,
        "mode": mode,
        "max_turns": max_turns,
        "use_tools": use_tools,
        "num_samples": len(sample_files),
        "files": [],
    }

    # Process samples with bounded concurrency
    sem = asyncio.Semaphore(8)

    async def _process_one(sample_path: Path) -> dict[str, Any]:
        async with sem:
            with sample_path.open() as f:
                sample = json.load(f)

            sample_id = sample.get("sample_id", 0)
            session_id = f"{safe_name}__{mode}__turns{max_turns}__sample{sample_id:04d}"

            # Build prediction / reference objects for evaluation
            response = sample.get("response", {})
            gt = sample.get("ground_truth", {})

            eval_metrics: dict[str, Any] = {}

            # Caption
            pred_caption = ""
            caption_data = response.get("caption", {})
            if isinstance(caption_data, dict):
                parts = [caption_data.get("description", "")]
                if seq := caption_data.get("sequence_characteristics"):
                    parts.append(str(seq))
                if orient := caption_data.get("orientation"):
                    parts.append(str(orient))
                parts.extend(str(f) for f in caption_data.get("findings", []))
                pred_caption = " ".join(parts)
            elif isinstance(caption_data, str):
                pred_caption = caption_data

            gt_caption = gt.get("caption", "")
            if gt_caption:
                try:
                    caption_metrics = _eval_caption_sample(pred_caption, gt_caption)
                    eval_metrics["caption"] = {
                        k: v for k, v in caption_metrics.items() if k not in {"bert_model"}
                    }
                except Exception as exc:
                    print(f"  Sample {sample_id}: caption eval failed: {exc}")
                    eval_metrics["caption"] = {}

            # Diagnosis
            diag_data = response.get("diagnosis", {})
            pred_diagnosis: list[str] = []
            if isinstance(diag_data, dict):
                primary = diag_data.get("primary_diagnosis", "")
                if primary:
                    pred_diagnosis.append(primary)
                for dd in diag_data.get("differential_diagnoses", []):
                    if isinstance(dd, dict):
                        name = dd.get("diagnosis", "")
                        if name:
                            pred_diagnosis.append(name)
                    elif isinstance(dd, str) and dd:
                        pred_diagnosis.append(dd)
            elif isinstance(diag_data, str):
                pred_diagnosis = [diag_data]

            gt_diagnosis = gt.get("final_diagnosis", "")
            if gt_diagnosis and pred_diagnosis:
                try:
                    diag_metrics = await _eval_diagnosis_sample(
                        pred_diagnosis,
                        gt_diagnosis,
                        judge_model=config.judge_model,
                        seed=config.seed,
                    )
                    eval_metrics["diagnosis"] = {
                        "top1": diag_metrics.get("top1", 0.0),
                        "top5": diag_metrics.get("top5", 0.0),
                        "coverage": diag_metrics.get("coverage", 0.0),
                        "entropy": diag_metrics.get("entropy", 0.0),
                        "judgment_method": "llm",
                        "semantic_match_model": diag_metrics.get(
                            "semantic_match_model", config.judge_model or "unknown"
                        ),
                        "normalized_pred": pred_diagnosis[0] if pred_diagnosis else "",
                        "normalized_ref": gt_diagnosis,
                    }
                except Exception as exc:
                    print(f"  Sample {sample_id}: diagnosis eval failed: {exc}")
                    eval_metrics["diagnosis"] = {}

            # Localization
            loc_data = response.get("localization", {})
            pred_boxes: list[list[float]] = []
            pred_scores: list[float] = []
            if isinstance(loc_data, dict):
                for loc in loc_data.get("localizations", []):
                    bbox = loc.get("bounding_box")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        pred_boxes.append([float(c) for c in bbox[:4]])
                        pred_scores.append(loc.get("confidence", 1.0))

            gt_boxes: list[list[float]] = []
            for loc in gt.get("localizations", []):
                bbox = loc.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    gt_boxes.append([float(c) for c in bbox[:4]])

            if pred_boxes or gt_boxes:
                try:
                    loc_metrics = _eval_localization_sample(
                        {
                            "boxes": pred_boxes,
                            "scores": pred_scores,
                            "labels": [0] * len(pred_boxes),
                        },
                        {
                            "boxes": gt_boxes,
                            "scores": [1.0] * len(gt_boxes),
                            "labels": [0] * len(gt_boxes),
                        },
                    )
                    eval_metrics["localization"] = {
                        "map30": loc_metrics.get("map30", 0.0),
                        "map50": loc_metrics.get("map50", 0.0),
                        "map50_95": loc_metrics.get("map50_95", 0.0),
                        "acc50": loc_metrics.get("acc50", 0.0),
                        "recall50": loc_metrics.get("recall50", 0.0),
                        "precision50": loc_metrics.get("precision50", 0.0),
                        "tp30": loc_metrics.get("tp30", 0),
                        "fp30": loc_metrics.get("fp30", 0),
                    }
                except Exception as exc:
                    print(f"  Sample {sample_id}: localization eval failed: {exc}")
                    eval_metrics["localization"] = {}

            # Build Claude Code JSONL events
            events = _build_claude_code_jsonl(
                sample=sample,
                session_id=session_id,
                eval_metrics=eval_metrics,
                global_config=global_config,
            )

            file_name = f"{session_id}.jsonl"
            file_path = traces_dir / file_name
            with file_path.open("w") as f:
                for evt in events:
                    f.write(json.dumps(evt) + "\n")

            return {
                "file": file_name,
                "sample_id": sample_id,
                "session_id": session_id,
                "num_events": len(events),
                "eval_metrics": eval_metrics,
            }

    results = await asyncio.gather(*(_process_one(p) for p in sample_files))
    manifest["files"] = results

    # Write manifest
    _write_json(out_dir / "manifest.json", manifest)

    # Write README.md with dataset card metadata
    readme = f"""---
tags:
- agent-traces
- medical-imaging
- radiology
- vlm
- tool-calling
- brain-mri
- nova
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
---

# NOVA Brain-MRI Agent Traces — {safe_name}

Multi-turn agentic traces from the NOVA brain-MRI benchmark, generated with
**{model_name}** in **{mode}** mode (max {max_turns} turns).

## What's inside

- **{len(sample_files)} sessions**, one per case
- Full conversation turns (user, assistant, tool results)
- Per-sample evaluation metrics (caption, diagnosis, localization)
- Model configuration metadata for reproducibility

## Format

Sessions use the [Claude Code JSONL schema](https://huggingface.co/changelog/agent-trace-viewer)
with custom ``nova_metadata`` and ``nova_turn_meta`` blocks for medical-imaging
context and evaluation scores.

## Evaluation

Metrics follow the NOVA benchmark protocol:

- **Caption**: BLEU, BERTScore F1, METEOR, clinical/modality keyword F1
- **Diagnosis**: Top-1 / Top-5 accuracy via LLM semantic matching
- **Localization**: mAP@0.5, ACC50, recall/precision

## Loading

```python
from datasets import load_dataset

ds = load_dataset("<your-org>/nova-traces-{safe_name}", split="train")
```

## Source

Generated with [GAZE](https://github.com/liamchalcroft/gaze)
using the NOVA benchmark pipeline.
"""
    (out_dir / "README.md").write_text(readme)

    print(f"\nExported {len(results)} traces to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    print(f"Traces dir: {traces_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NOVA run to Hugging Face Agent Traces format",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing sample_*.json and summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for HF-ready traces",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model for diagnosis semantic matching (default: from env or openai/gpt-5-nano)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible diagnosis evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.judge_model:
        os.environ["NOVA_SEMANTIC_MATCH_MODEL"] = args.judge_model

    config = ExportConfig(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        seed=args.seed,
    )

    asyncio.run(export_run(config))


if __name__ == "__main__":
    main()
