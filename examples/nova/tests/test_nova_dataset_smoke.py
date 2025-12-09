from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image


# Ensure example source is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SRC = REPO_ROOT / "examples" / "nova" / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.mark.asyncio
async def test_nova_dataset_smoke(monkeypatch, tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    from datasets import Dataset as HFDataset  # type: ignore
    from datasets import Features, Image as HFImage  # type: ignore

    import src.data.nova_dataset as nova_dataset
    from src.data.nova_dataset import NovaDataset

    # Prepare minimal ground-truth CSVs
    captions = tmp_path / "captions.csv"
    case_meta = tmp_path / "case_metadata.csv"
    bboxes = tmp_path / "bboxes_gold.csv"

    _write_csv(
        captions,
        ["filename", "case_id", "scan_id", "caption"],
        [["case1.png", "c1", "s1", "caption text"]],
    )
    _write_csv(
        case_meta,
        ["case_id", "clinical_history", "final_diagnosis"],
        [["c1", "history text", "diagnosis text"]],
    )
    _write_csv(
        bboxes,
        ["filename", "x", "y", "width", "height"],
        [["case1.png", "1.0", "2.0", "3.0", "4.0"]],
    )

    # Create a tiny PIL image with filename metadata for alignment
    img_path = tmp_path / "case1.png"
    pil_img = Image.new("RGB", (4, 4), color=(123, 124, 125))
    pil_img.save(img_path)
    pil_img.filename = str(img_path)

    # Build a lightweight HF dataset in-memory
    fake_ds: HFDataset = datasets.Dataset.from_dict(
        {"image": [pil_img]},
        features=Features({"image": HFImage()}),
    )

    # Monkeypatch load_dataset to avoid network access
    monkeypatch.setattr(nova_dataset, "load_dataset", lambda *_, **__: fake_ds)

    ds = NovaDataset(data_dir=str(tmp_path), ground_truth_dir=str(tmp_path), transform=None)
    sample = ds[0]

    assert sample["has_ground_truth"] is True
    meta = sample["metadata"]
    gt = sample["ground_truth"]

    assert meta["filename"] == "case1.png"
    assert gt["final_diagnosis"] == "diagnosis text"
    assert gt["caption"] == "caption text"

    locs = gt["localizations"]
    assert len(locs) == 1
    assert locs[0]["bbox"] == (1.0, 2.0, 3.0, 4.0)

