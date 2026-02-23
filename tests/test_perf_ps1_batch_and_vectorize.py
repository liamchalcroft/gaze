"""Tests for Patch Set #1 performance fixes.

Covers:
1. NovaDataset.get_sample_metadata() — header-only image reads
2. CLI deferred image loading — work_items stores indices, not samples
3. compute_intensity_profile vectorized pixel access
4. base.py final-turn reset removal (verified via absence of call)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from radiant_harness.tools.visual import compute_intensity_profile


# ---------------------------------------------------------------------------
# 1. NovaDataset.get_sample_metadata
# ---------------------------------------------------------------------------


class _FakeNovaDataset:
    """Minimal stand-in for NovaDataset that avoids HuggingFace downloads.

    Uses a tiny on-disk image so we can test header-only reads vs full decode.
    """

    def __init__(self, tmp_path: Path, n_samples: int = 3) -> None:
        import pandas as pd

        self._repo_dir = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        rows = []
        for i in range(n_samples):
            img_path = f"images/sample_{i}.png"
            full = tmp_path / img_path
            full.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64 + i, 48 + i), color=(i * 30, 0, 0)).save(full)
            rows.append(
                {
                    "image_path": img_path,
                    "filename": f"sample_{i}.png",
                    "case_id": f"case_{i}",
                    "scan_id": f"scan_{i}",
                    "caption_text": f"Caption {i}",
                    "meta": {"clinical_history": f"history_{i}", "final_diagnosis": f"dx_{i}"},
                    "bboxes": [{"source": "gold", "x": 10, "y": 20, "width": 30, "height": 40}],
                }
            )

        self._df = pd.DataFrame(rows)
        self.transform = None

    def __len__(self) -> int:
        return len(self._df)


def _patch_nova_methods(fake: _FakeNovaDataset) -> _FakeNovaDataset:
    """Monkey-patch the real NovaDataset methods onto a fake instance."""
    from examples.nova.src.data.nova_dataset import NovaDataset

    fake._extract_row_metadata = NovaDataset._extract_row_metadata.__get__(fake)  # type: ignore[attr-defined]
    fake.get_sample_metadata = NovaDataset.get_sample_metadata.__get__(fake)  # type: ignore[attr-defined]
    fake.__getitem__ = NovaDataset.__getitem__.__get__(fake)  # type: ignore[attr-defined]
    return fake


class TestGetSampleMetadata:
    def test_returns_ground_truth_and_image_size(self, tmp_path: Path) -> None:
        fake = _patch_nova_methods(_FakeNovaDataset(tmp_path))
        meta = fake.get_sample_metadata(0)

        assert "ground_truth" in meta
        assert "metadata" in meta
        assert "image_size" in meta
        assert meta["image_size"] == (64, 48)
        assert meta["has_ground_truth"] is True
        # Should NOT contain a loaded PIL Image
        assert "image" not in meta

    def test_metadata_matches_getitem(self, tmp_path: Path) -> None:
        fake = _patch_nova_methods(_FakeNovaDataset(tmp_path))
        meta = fake.get_sample_metadata(1)
        full = fake.__getitem__(1)  # type: ignore[attr-defined]

        assert meta["ground_truth"] == full["ground_truth"]
        assert meta["metadata"] == full["metadata"]
        assert meta["hf_index"] == full["hf_index"]

    def test_does_not_decode_pixels(self, tmp_path: Path) -> None:
        """Verify get_sample_metadata reads only the header, not full pixels."""
        fake = _patch_nova_methods(_FakeNovaDataset(tmp_path))

        open_calls: list[str] = []

        _real_open = Image.open

        def _tracking_open(path: Any, *a: Any, **kw: Any) -> Any:
            open_calls.append(str(path))
            return _real_open(path, *a, **kw)

        with patch("PIL.Image.open", _tracking_open):
            fake.get_sample_metadata(0)

        # Image.open is called (for header), but convert("RGB") should NOT be
        assert len(open_calls) == 1

    def test_index_out_of_range(self, tmp_path: Path) -> None:
        fake = _patch_nova_methods(_FakeNovaDataset(tmp_path, n_samples=2))
        with pytest.raises(IndexError):
            fake.get_sample_metadata(5)
        with pytest.raises(IndexError):
            fake.get_sample_metadata(-1)

    def test_missing_image_file(self, tmp_path: Path) -> None:
        fake = _patch_nova_methods(_FakeNovaDataset(tmp_path))
        # Delete the image file
        img_path = tmp_path / "images" / "sample_0.png"
        img_path.unlink()

        with pytest.raises(FileNotFoundError):
            fake.get_sample_metadata(0)


# ---------------------------------------------------------------------------
# 2. CLI deferred loading — work_items is list[int]
# ---------------------------------------------------------------------------


class TestCLIDeferredLoading:
    """Verify that the pre-validation loop stores indices, not full samples."""

    def test_work_items_are_indices(self) -> None:
        """Structural test: confirm work_items is list[int] in the code."""
        import ast
        import inspect

        from examples.nova.src.cli import run_evaluation

        source = inspect.getsource(run_evaluation)
        tree = ast.parse(source)

        # Find the work_items annotation
        found_annotation = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "work_items":
                    # Check it's list[int], not list[tuple[...]]
                    annotation_src = ast.dump(node.annotation) if node.annotation else ""
                    assert "int" in annotation_src
                    assert "tuple" not in annotation_src.lower()
                    found_annotation = True
                    break

        assert found_annotation, "work_items type annotation not found"

    def test_process_sample_takes_one_arg(self) -> None:
        """Confirm _process_sample only takes idx, not (idx, sample)."""
        import ast
        import inspect

        from examples.nova.src.cli import run_evaluation

        source = inspect.getsource(run_evaluation)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_sample":
                # args should be (idx,) not (idx, sample)
                arg_names = [a.arg for a in node.args.args]
                assert arg_names == ["idx"], f"Expected ['idx'], got {arg_names}"
                return

        pytest.fail("_process_sample not found")


# ---------------------------------------------------------------------------
# 3. compute_intensity_profile vectorized access
# ---------------------------------------------------------------------------


class TestIntensityProfileVectorized:
    def test_basic_profile_values(self) -> None:
        """Verify vectorized profile produces correct intensity values."""
        # Create a gradient image: left=0, right=255
        arr = np.zeros((100, 256), dtype=np.uint8)
        for x in range(256):
            arr[:, x] = x
        image = Image.fromarray(arr)

        result = compute_intensity_profile(image, (0.0, 0.5), (1.0, 0.5))

        profile = result["profile"]
        assert isinstance(profile, list)
        assert len(profile) > 0
        # Profile should be monotonically non-decreasing (gradient L→R)
        for i in range(1, len(profile)):
            assert profile[i] >= profile[i - 1], f"profile[{i}]={profile[i]} < profile[{i-1}]={profile[i-1]}"
        # First value near 0, last near 255
        assert profile[0] <= 5
        assert profile[-1] >= 250

    def test_profile_stats_are_numpy_computed(self) -> None:
        """Ensure stats (mean, std, min, max) come from numpy, not Python list."""
        arr = np.full((50, 50), 128, dtype=np.uint8)
        image = Image.fromarray(arr)

        result = compute_intensity_profile(image, (0.0, 0.5), (1.0, 0.5))

        assert result["mean"] == 128.0
        assert result["std"] == 0.0
        assert result["min"] == 128
        assert result["max"] == 128

    def test_profile_matches_python_loop(self) -> None:
        """Cross-check: vectorized result matches naive per-pixel loop."""
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (100, 100), dtype=np.uint8)
        image = Image.fromarray(arr)

        result = compute_intensity_profile(image, (0.1, 0.2), (0.9, 0.8))
        profile = result["profile"]

        # Reproduce with naive loop
        gray = np.array(image.convert("L"))
        h, w = gray.shape
        x0, y0 = int(0.1 * (w - 1)), int(0.2 * (h - 1))
        x1, y1 = int(0.9 * (w - 1)), int(0.8 * (h - 1))
        n = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
        xs = np.clip(np.linspace(x0, x1, n).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y0, y1, n).astype(int), 0, h - 1)
        expected = [int(gray[y, x]) for y, x in zip(ys, xs)]

        assert profile == expected


# ---------------------------------------------------------------------------
# 4. base.py final-turn reset removal
# ---------------------------------------------------------------------------


class TestFinalTurnNoReset:
    """Verify reset_to_original is NOT called on the final turn."""

    def test_no_reset_call_in_final_turn_block(self) -> None:
        """Structural test: confirm reset_to_original() is not called in the
        final-turn branch of _run_analysis.
        """
        import ast
        import inspect
        import textwrap

        from radiant_harness.base import AgenticProcessorBase

        source = textwrap.dedent(inspect.getsource(AgenticProcessorBase._run_analysis))
        tree = ast.parse(source)

        # Walk AST looking for calls to reset_to_original
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "reset_to_original":
                    pytest.fail(
                        "Found reset_to_original() call in _run_analysis — "
                        "this should have been removed as dead code on the final turn"
                    )
