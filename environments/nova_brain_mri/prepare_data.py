#!/usr/bin/env python
"""Build the NOVA brain-MRI JSONL (+ images) the standalone environment reads.

``NOVABrainMRIEnv._load_cases`` (in ``src/nova_brain_mri/__init__.py``) looks for
``data/nova_<split>.jsonl`` but nothing produces it. This script downloads the
NOVA dataset from HuggingFace (``c-i-ber/Nova``), extracts metadata and ground
truth from its parquet file, copies the referenced images into a local
``data/images`` directory, and writes ``data/nova_<split>.jsonl``.

The field extraction mirrors ``examples/nova``'s ``NovaDataset`` (parquet path,
caption/diagnosis/clinical-history/bbox columns) so the standalone environment
and the sibling example agree on ground truth.

JSONL line schema (grounded in the consuming code):

    {
      "image_path": "/abs/path/data/images/case0001_001.png",  # __init__.py line 244
      "clinical_history": "...",                                # __init__.py line 245
      "modality": "MRI",                                        # __init__.py line 246
      "caption": "Ground-truth caption",                        # rewards.py caption_reward line 223
      "diagnosis": "Ground-truth final diagnosis",     # rewards.py diagnosis_reward line 287
      "boxes": [[x1, y1, x2, y2], ...]                 # rewards.py _extract_ref_boxes line 357
    }

``image_path`` is an absolute filesystem path; the environment prepends
``file://`` itself when building the user message (``__init__.py`` line 266),
so the path here must NOT carry a scheme.

NOVA only ships a single pool of cases (no official train/val/test split in the
parquet), so by default the whole dataset is written to the requested split
filename (``--split test`` -> ``nova_test.jsonl``). Use ``--max-samples`` for a
quick smoke subset.

Dependencies (declared in this package's pyproject ``[project]`` /
``[project.optional-dependencies].dev`` plus pandas/huggingface-hub for
extraction):

    pip install -e .
    pip install pandas huggingface-hub          # parquet + snapshot download

Usage:
    python prepare_data.py [--split test] [--output-dir ./data]
                           [--max-samples N] [--data-dir /local/Nova]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

HF_REPO_ID = "c-i-ber/Nova"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export c-i-ber/Nova to nova_<split>.jsonl for the standalone environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Split filename to write (nova_<split>.jsonl); NOVA ships one case pool",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for nova_<split>.jsonl and the copied images/ subdir",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of exported cases (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local NOVA repo directory to use instead of downloading from HuggingFace",
    )
    return parser.parse_args()


def _resolve_repo_dir(data_dir: str | None) -> Path:
    """Return the NOVA repo directory, downloading from HuggingFace if needed."""
    if data_dir and Path(data_dir).expanduser().exists():
        return Path(data_dir).expanduser()

    from huggingface_hub import snapshot_download

    print(f"Downloading NOVA dataset from {HF_REPO_ID} ...")  # noqa: T201
    repo_dir = Path(snapshot_download(HF_REPO_ID, repo_type="dataset"))
    print(f"NOVA dataset cached at {repo_dir}")  # noqa: T201
    return repo_dir


def _gold_boxes(bboxes_raw: Any) -> list[list[float]]:
    """Convert NOVA gold bboxes from (x, y, width, height) to [x1, y1, x2, y2].

    Mirrors NovaDataset._extract_row_metadata: only ``source == 'gold'`` boxes
    are kept.
    """
    boxes: list[list[float]] = []
    if bboxes_raw is None:
        return boxes
    for bbox in bboxes_raw:
        if not isinstance(bbox, dict):
            continue
        if bbox.get("source") != "gold":
            continue
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["width"])
        h = float(bbox["height"])
        boxes.append([x, y, x + w, y + h])
    return boxes


def main() -> None:
    args = _parse_args()

    # Lazy import: keep `python -m py_compile` / `--help` working without
    # pandas / huggingface-hub installed.
    import pandas as pd

    repo_dir = _resolve_repo_dir(args.data_dir)
    parquet_path = repo_dir / "data" / "nova-v1.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found at {parquet_path}. "
            "Ensure the NOVA dataset downloaded correctly."
        )

    df = pd.read_parquet(parquet_path)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples]
    print(f"Loaded {len(df)} samples from {parquet_path}")  # noqa: T201

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"nova_{args.split}.jsonl"

    written = 0
    missing_images = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            meta = row["meta"] if isinstance(row["meta"], dict) else {}
            src_image = repo_dir / row["image_path"]
            if not src_image.exists():
                missing_images += 1
                continue
            # Copy image into a local, stable directory keyed by filename.
            dst_image = images_dir / row["filename"]
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            case = {
                "image_path": str(dst_image.resolve()),
                "clinical_history": meta.get("clinical_history", ""),
                "modality": "MRI",
                "caption": row["caption_text"],
                "diagnosis": meta.get("final_diagnosis", ""),
                "boxes": _gold_boxes(row["bboxes"]),
            }
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} NOVA cases to {out_path}")  # noqa: T201
    print(f"Copied images into {images_dir}")  # noqa: T201
    if missing_images:
        print(  # noqa: T201
            f"Skipped {missing_images} cases whose image files were missing in the repo."
        )


if __name__ == "__main__":
    main()
