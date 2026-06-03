#!/usr/bin/env python
"""Build the local GEMeX-ThinkVG JSONL that ``eval.py`` consumes.

The evaluation script (``examples/gemex_thinkvg/eval.py``) reads a local JSONL
dataset whose lines carry, at minimum, an ``image_path`` plus the question and
the ground-truth answer / location / bounding box. The HuggingFace source
``BoKelvin/GEMeX-ThinkVG`` stores the ground truth inside an XML-style
``response`` column, so this script downloads that dataset, parses each
``response`` exactly as ``GEMeXDataset`` does, and writes one JSONL line per
sample with the explicit fields ``eval.py`` expects.

The JSONL line schema mirrors the keys ``eval.py`` reads (see
``_resolve_image_path`` and ``_reference_from_case`` in eval.py):

    {
      "image_path": "p10/p10000032/.../xray.jpg",  # relative MIMIC-CXR path
      "question": "...",                            # eval.py line 299
      "question_type": "open_ended_questions",      # eval.py line 301 (raw HF label)
      "options": [],                                # eval.py line 303
      "answer": "...",                              # eval.py line 88
      "location_reference": "right lower lobe",     # eval.py lines 91-93
      "bbox": [x1, y1, x2, y2],                      # eval.py line 86
      "response": "<response>...</response>"         # eval.py line 84 (fallback parser)
    }

``image_path`` is written as the dataset's native (relative) MIMIC-CXR path.
The MIMIC-CXR-JPG images themselves require PhysioNet credentialed access and
are NOT downloaded here; pass the image root to ``eval.py`` via ``--image-dir``.
GEMeX-ThinkVG currently exposes only a ``train`` split, so the default output is
``data/train.jsonl``.

Requires the ``gemex`` extra (HuggingFace ``datasets``):

    uv sync --extra gemex
    uv run --extra gemex python -m examples.gemex_thinkvg.prepare_data --split train

Usage:
    python prepare_data.py [--split train] [--output-dir ./data]
                           [--max-samples N] [--question-type TYPE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BoKelvin/GEMeX-ThinkVG to the local JSONL eval.py reads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split to export (GEMeX-ThinkVG currently exposes only 'train')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for the generated <split>.jsonl",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of exported samples (default: all)",
    )
    parser.add_argument(
        "--question-type",
        type=str,
        default=None,
        choices=[
            "open_ended",
            "closed_ended",
            "single_choice",
            "multi_choice",
        ],
        help="Optional question-type filter (default: keep all types)",
    )
    return parser.parse_args()


def _case_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Map one ``GEMeXDataset`` sample to a JSONL line for ``eval.py``.

    ``GEMeXDataset`` already parses the XML ``response`` into ``answer``,
    ``location_reference`` and ``bbox`` (see its ``_parse_ground_truth``), so we
    take those parsed fields directly and additionally retain the raw
    ``raw_response`` under the ``response`` key that ``eval.py`` falls back to.
    """
    return {
        # eval.py resolves this relative path against --image-dir.
        "image_path": sample.get("image_path_relative", ""),
        "question": sample.get("question", ""),
        # Keep the raw HF label; eval.py normalises it itself.
        "question_type": sample.get("metadata", {}).get(
            "question_type_raw", sample.get("question_type", "open_ended_questions")
        ),
        "options": sample.get("options", []),
        "answer": sample.get("answer", ""),
        "location_reference": sample.get("location_reference", ""),
        "bbox": sample.get("bbox", [0, 0, 0, 0]),
        # Raw XML response: eval.py's _reference_from_case parses this when the
        # explicit fields above are empty.
        "response": sample.get("raw_response", ""),
    }


def main() -> None:
    args = _parse_args()

    # Lazy import so `python -m py_compile` and `--help` work without the
    # `gemex` extra (datasets/torch) installed.
    from examples.gemex_thinkvg.src.dataset import GEMeXDataset

    dataset = GEMeXDataset(
        mimic_cxr_root=None,
        split=args.split,
        max_samples=args.max_samples,
        question_type=args.question_type,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.split}.jsonl"

    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in dataset:
            line = _case_from_sample(sample)
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} GEMeX-ThinkVG cases to {out_path}")  # noqa: T201
    print(  # noqa: T201
        "Pass the MIMIC-CXR-JPG image root to eval.py via --image-dir "
        "(images are not downloaded by this script)."
    )


if __name__ == "__main__":
    main()
