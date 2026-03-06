from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure example source is importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "nova"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _import_ground_truth():
    """Import NovaGroundTruth directly, bypassing torch-dependent __init__.py."""
    import importlib.util

    name = "src.data.nova_ground_truth"
    spec = importlib.util.spec_from_file_location(
        name,
        EXAMPLE_ROOT / "src" / "data" / "nova_ground_truth.py",
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.NovaGroundTruth


def test_ground_truth_bbox_conversion(tmp_path: Path) -> None:
    """Verify (x, y, width, height) → (x1, y1, x2, y2) conversion.

    This is the critical conversion that affects all localization metrics.
    Does NOT require torch or datasets.
    """
    cls = _import_ground_truth()

    # Write minimal CSV files
    _write_csv(
        tmp_path / "captions.csv",
        ["filename", "case_id", "scan_id", "caption"],
        [["img.png", "c1", "s1", "test caption"]],
    )
    _write_csv(
        tmp_path / "case_metadata.csv",
        ["case_id", "clinical_history", "final_diagnosis"],
        [["c1", "test history", "test diagnosis"]],
    )
    _write_csv(
        tmp_path / "bboxes_gold.csv",
        ["filename", "x", "y", "width", "height"],
        [
            # x=10, y=20, width=100, height=80 → (10, 20, 110, 100)
            ["img.png", "10.0", "20.0", "100.0", "80.0"],
        ],
    )

    gt = cls(str(tmp_path))
    sample = gt.get_ground_truth("img.png")
    assert sample is not None

    assert sample.caption == "test caption"
    assert sample.final_diagnosis == "test diagnosis"
    assert sample.clinical_history == "test history"

    assert len(sample.localizations) == 1
    bbox = sample.localizations[0].bbox
    # (x, y, w, h) = (10, 20, 100, 80) → (x1, y1, x2, y2) = (10, 20, 110, 100)
    assert bbox == (10.0, 20.0, 110.0, 100.0), f"Expected (10, 20, 110, 100), got {bbox}"


def test_ground_truth_multiple_boxes(tmp_path: Path) -> None:
    """Verify multiple bboxes for the same image are collected correctly."""
    cls = _import_ground_truth()

    _write_csv(
        tmp_path / "captions.csv",
        ["filename", "case_id", "scan_id", "caption"],
        [["img.png", "c1", "s1", "caption"]],
    )
    _write_csv(
        tmp_path / "case_metadata.csv",
        ["case_id", "clinical_history", "final_diagnosis"],
        [["c1", "history", "diagnosis"]],
    )
    _write_csv(
        tmp_path / "bboxes_gold.csv",
        ["filename", "x", "y", "width", "height"],
        [
            ["img.png", "0.0", "0.0", "50.0", "50.0"],
            ["img.png", "100.0", "100.0", "30.0", "40.0"],
        ],
    )

    gt = cls(str(tmp_path))
    # Access internal dict to avoid beartype cross-module class identity issue
    sample = gt._ground_truth["img.png"]
    assert len(sample.localizations) == 2
    assert sample.localizations[0].bbox == (0.0, 0.0, 50.0, 50.0)
    assert sample.localizations[1].bbox == (100.0, 100.0, 130.0, 140.0)


def test_nova_dataset_smoke(monkeypatch, tmp_path: Path) -> None:
    """Smoke test for NovaDataset using parquet + snapshot_download."""
    pd = pytest.importorskip("pandas")

    from src.data import nova_dataset
    from src.data.nova_dataset import NovaDataset

    # Build a minimal parquet file matching the real dataset schema
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "case1.png"
    pil_img = Image.new("RGB", (4, 4), color=(123, 124, 125))
    pil_img.save(img_path)

    df = pd.DataFrame(
        [
            {
                "filename": "case1.png",
                "case_id": "c1",
                "scan_id": "s1",
                "caption_text": "caption text",
                "image_path": "images/case1.png",
                "meta": {
                    "clinical_history": "history text",
                    "final_diagnosis": "diagnosis text",
                },
                "bboxes": [
                    {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0, "source": "gold"},
                ],
            }
        ]
    )
    df.to_parquet(data_dir / "nova-v1.parquet")

    # Monkeypatch snapshot_download to return our tmp_path
    monkeypatch.setattr(nova_dataset, "snapshot_download", lambda *_a, **_kw: str(tmp_path))

    ds = NovaDataset(data_dir=str(tmp_path), transform=None)
    sample = ds[0]

    assert sample["has_ground_truth"] is True
    meta = sample["metadata"]
    gt = sample["ground_truth"]

    assert meta["filename"] == "case1.png"
    assert gt["final_diagnosis"] == "diagnosis text"
    assert gt["caption"] == "caption text"

    locs = gt["localizations"]
    assert len(locs) == 1
    # Parquet stores (x, y, width, height) -> dataset converts to (x1, y1, x2, y2)
    assert locs[0]["bbox"] == (1.0, 2.0, 4.0, 6.0)


def test_localization_reward_iou_threshold_alignment() -> None:
    """Verify reward IoU threshold matches evaluation standard (0.5).

    A box with IoU=0.35 should get 0 reward (below threshold),
    ensuring training doesn't reward boxes that fail at eval time.
    """
    from src.rewards import compute_localization_reward

    # Perfect overlap → reward = 1.0
    pred = [[10.0, 10.0, 110.0, 110.0]]
    ref = [[10.0, 10.0, 110.0, 110.0]]
    assert compute_localization_reward(pred, ref) == 1.0

    # No overlap → reward = 0.0
    pred_no = [[0.0, 0.0, 10.0, 10.0]]
    ref_no = [[200.0, 200.0, 300.0, 300.0]]
    assert compute_localization_reward(pred_no, ref_no) == 0.0

    # Partial overlap: IoU below 0.5 threshold → NOT a true positive
    # Two 100x100 boxes offset by 60px → intersection 4000, union 16000, IoU 0.25
    pred_partial = [[0.0, 0.0, 100.0, 100.0]]
    ref_partial = [[60.0, 0.0, 160.0, 100.0]]
    reward = compute_localization_reward(pred_partial, ref_partial)
    assert reward == 0.0, f"IoU=0.25 should not match at threshold 0.5, got reward={reward}"

    # Overlap just above 0.5: should match
    # Two 100x100 boxes offset by 30px → intersection 7000, union 13000, IoU ~0.538
    pred_good = [[0.0, 0.0, 100.0, 100.0]]
    ref_good = [[30.0, 0.0, 130.0, 100.0]]
    reward_good = compute_localization_reward(pred_good, ref_good)
    assert reward_good == 1.0, f"IoU≈0.54 should match at threshold 0.5, got reward={reward_good}"


def test_caption_reward_known_pair() -> None:
    """Spot-check caption reward on a known reference pair."""
    from src.rewards import compute_caption_reward

    # Identical strings → F1 = 1.0
    assert compute_caption_reward("hello world", "hello world") == 1.0

    # No overlap → F1 = 0.0
    assert compute_caption_reward("alpha beta", "gamma delta") == 0.0

    # Partial overlap: 3 of 4 tokens shared → precision 0.75, recall 0.75, F1 0.75
    reward = compute_caption_reward("axial t2 mri brain", "axial t2 mri lesion")
    assert abs(reward - 0.75) < 1e-6, f"Expected 0.75, got {reward}"


def test_diagnosis_reward_normalization() -> None:
    """Verify diagnosis matching works with common variations."""
    from src.rewards import compute_diagnosis_reward

    # Exact match (after normalization) → score > 0
    assert compute_diagnosis_reward("Glioblastoma", "glioblastoma") > 0.0

    # No match → 0.0
    assert compute_diagnosis_reward("Meningioma", "Glioblastoma") == 0.0


def test_diagnosis_normalize_preserves_severity() -> None:
    """Severity qualifiers (mild/moderate/severe) are clinically meaningful.

    They must NOT be stripped during normalization, because e.g.
    'mild hydrocephalus' and 'severe hydrocephalus' are different conditions.
    """
    from src.rewards import _normalize_diagnosis

    # Severity must be preserved
    assert "mild" in _normalize_diagnosis("mild hydrocephalus")
    assert "moderate" in _normalize_diagnosis("moderate stenosis")
    assert "severe" in _normalize_diagnosis("severe atrophy")

    # Hedging modifiers should still be stripped
    assert "possible" not in _normalize_diagnosis("possible glioma")
    assert "probable" not in _normalize_diagnosis("probable meningioma")
    assert "likely" not in _normalize_diagnosis("likely infarct")
    assert "suspected" not in _normalize_diagnosis("suspected tumor")


def test_diagnosis_severity_affects_matching() -> None:
    """'Mild X' should NOT match 'severe X' after normalization."""
    from src.rewards import compute_diagnosis_reward

    # Same condition, different severity → should NOT be a top-1 match
    reward = compute_diagnosis_reward("mild hydrocephalus", "severe hydrocephalus")
    assert reward == 0.0, f"Different severity should not match, got reward={reward}"


def test_processor_mode_parameter() -> None:
    """Processor accepts mode parameter and stores it."""
    from src.processor import NOVAAgenticProcessor

    proc_agentic = NOVAAgenticProcessor(mode="agentic")
    assert proc_agentic._mode == "agentic"

    proc_single = NOVAAgenticProcessor(mode="single_turn")
    assert proc_single._mode == "single_turn"


def test_config_mode_field() -> None:
    """NOVAConfig supports mode field."""
    from src.config import NOVAConfig

    # Default mode is agentic
    config = NOVAConfig()
    assert config.mode == "agentic"

    # Can override
    config2 = NOVAConfig(mode="single_turn")
    assert config2.mode == "single_turn"


def test_detection_iou_range_thresholds() -> None:
    """Verify the mAP@[50:95] range produces exactly 10 thresholds from 0.5 to 0.95."""
    pytest.importorskip("torch")
    from src.evaluation.detection import _MAP_RANGE_IOU_THRESHOLDS

    assert len(_MAP_RANGE_IOU_THRESHOLDS) == 10
    assert abs(_MAP_RANGE_IOU_THRESHOLDS[0] - 0.5) < 1e-9
    assert abs(_MAP_RANGE_IOU_THRESHOLDS[-1] - 0.95) < 1e-9
    # Verify step size is 0.05
    for i in range(1, len(_MAP_RANGE_IOU_THRESHOLDS)):
        step = _MAP_RANGE_IOU_THRESHOLDS[i] - _MAP_RANGE_IOU_THRESHOLDS[i - 1]
        assert abs(step - 0.05) < 1e-9, f"Step {i}: expected 0.05, got {step}"
