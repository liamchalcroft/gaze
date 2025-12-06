"""Comprehensive tests for batch processing utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
from nova_retrieval_vlm.utils.batch_processing_utils import compute_evaluation_metrics
from nova_retrieval_vlm.utils.batch_processing_utils import draw_ground_truth_vs_predicted_boxes
from nova_retrieval_vlm.utils.batch_processing_utils import normalize_localization_result
from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result
from nova_retrieval_vlm.utils.batch_processing_utils import save_prediction
from nova_retrieval_vlm.utils.batch_processing_utils import save_reference


@pytest.mark.unit
class TestBatchContext:
    """Test cases for BatchContext data class."""

    def test_batch_context_creation(self):
        """Test BatchContext can be created with required fields."""
        ctx = BatchContext(
            idx=0,
            folder=Path("/tmp/test"),
            img_path=Path("/tmp/test.png"),
            width=256,
            height=256,
        )

        assert ctx.idx == 0
        assert ctx.folder == Path("/tmp/test")
        assert ctx.img_path == Path("/tmp/test.png")
        assert ctx.width == 256
        assert ctx.height == 256

    def test_batch_context_types(self):
        """Test BatchContext has correct types."""
        ctx = BatchContext(
            idx=1,
            folder=Path("/tmp"),
            img_path=Path("/tmp/img.png"),
            width=512,
            height=384,
        )

        assert isinstance(ctx.idx, int)
        assert isinstance(ctx.folder, Path)
        assert isinstance(ctx.img_path, Path)
        assert isinstance(ctx.width, int)
        assert isinstance(ctx.height, int)


@pytest.mark.unit
class TestLocalizationSchemaConversion:
    """Test cases for localization schema conversion via normalize_localization_result."""

    def test_convert_new_to_legacy_format(self):
        """Test conversion from new localizations format to legacy boxes format."""
        result = {
            "localizations": [
                {"bounding_box": [10, 20, 30, 40], "confidence": 0.9},
                {"bounding_box": [50, 60, 70, 80], "confidence": 0.8},
            ]
        }

        normalize_localization_result(result)

        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result
        assert result["boxes"] == [[10, 20, 30, 40], [50, 60, 70, 80]]
        assert result["labels"] == ["anomaly", "anomaly"]
        assert result["scores"] == [0.9, 0.8]

    def test_convert_handles_missing_confidence(self):
        """Test conversion handles missing confidence values."""
        result = {
            "localizations": [
                {"bounding_box": [10, 20, 30, 40]},
                {"bounding_box": [50, 60, 70, 80], "confidence": 0.7},
            ]
        }

        normalize_localization_result(result)

        assert result["scores"] == [1.0, 0.7]

    def test_convert_no_operation_when_boxes_exist(self):
        """Test no conversion when boxes already exist."""
        result = {
            "localizations": [{"bounding_box": [10, 20, 30, 40]}],
            "boxes": [[1, 2, 3, 4]],
            "labels": ["existing"],
            "scores": [0.5],
        }

        normalize_localization_result(result)

        # Should not modify existing boxes
        assert result["boxes"] == [[1, 2, 3, 4]]
        assert result["labels"] == ["existing"]
        assert result["scores"] == [0.5]

    def test_convert_empty_localizations(self):
        """Test conversion with empty localizations."""
        result = {"localizations": []}

        normalize_localization_result(result)

        assert result["boxes"] == []
        assert result["labels"] == []
        assert result["scores"] == []


@pytest.mark.unit
class TestNormalizeLocalizationResult:
    """Test cases for localization result normalization."""

    def test_normalize_adds_missing_keys(self):
        """Test normalization adds missing required keys."""
        result = {}

        normalize_localization_result(result)

        assert "boxes" in result
        assert "labels" in result
        assert "scores" in result
        assert result["boxes"] == []
        assert result["labels"] == []
        assert result["scores"] == []

    def test_normalize_standardizes_array_lengths(self):
        """Test normalization standardizes array lengths."""
        result = {
            "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "labels": ["anomaly"],  # Missing one label
            "scores": [],  # Missing all scores
        }

        normalize_localization_result(result)

        assert len(result["labels"]) == 2
        assert len(result["scores"]) == 2
        assert result["labels"] == ["anomaly", "anomaly"]
        assert result["scores"] == [1.0, 1.0]

    def test_normalize_preserves_correct_lengths(self):
        """Test normalization preserves correctly sized arrays."""
        result = {
            "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "labels": ["anomaly1", "anomaly2"],
            "scores": [0.9, 0.8],
        }

        normalize_localization_result(result)

        assert result["labels"] == ["anomaly1", "anomaly2"]
        assert result["scores"] == [0.9, 0.8]


@pytest.mark.unit
class TestSavePrediction:
    """Test cases for prediction saving."""

    def test_save_prediction_creates_file(self):
        """Test save_prediction creates prediction file with correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            result = {"boxes": [[10, 20, 30, 40]], "labels": ["test"]}

            save_prediction(folder, result)

            pred_file = folder / "pred.jsonl"
            assert pred_file.exists()

            with pred_file.open() as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == result

    def test_save_prediction_overwrites_existing(self):
        """Test save_prediction overwrites existing prediction files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            pred_file = folder / "pred.jsonl"

            # Create existing file
            pred_file.write_text('{"old": "data"}\n')

            new_result = {"new": "data"}
            save_prediction(folder, new_result)

            with pred_file.open() as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == new_result


@pytest.mark.unit
class TestSaveReference:
    """Test cases for reference saving."""

    def test_save_reference_creates_file(self):
        """Test save_reference creates reference file with ground truth data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            batch_index = 42

            # Mock dataset record
            dataset_record = {
                "bbox_gold": {
                    "x": [10, 50],
                    "y": [20, 60],
                    "width": [20, 20],
                    "height": [20, 20],
                },
                "caption": "Test caption",
                "final_diagnosis": "Test diagnosis",
            }
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = dataset_record

            save_reference(folder, batch_index, mock_dataset)

            ref_file = folder / "ref.jsonl"
            assert ref_file.exists()

            with ref_file.open() as f:
                saved_data = json.loads(f.read().strip())

            expected_boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]  # x,y,x+w,y+h format
            assert saved_data["boxes"] == expected_boxes
            assert saved_data["labels"] == ["anomaly", "anomaly"]
            assert saved_data["scores"] == [1.0, 1.0]
            assert saved_data["caption"] == "Test caption"
            assert saved_data["diagnosis"] == "Test diagnosis"
            assert saved_data["ground_truth_image_idx"] == 42

    def test_save_reference_handles_missing_bbox(self):
        """Test save_reference handles missing bbox_gold data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            dataset_record = {
                "caption": "No bbox caption",
                "diagnosis": "No bbox diagnosis",
            }
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = dataset_record

            save_reference(folder, 0, mock_dataset)

            ref_file = folder / "ref.jsonl"
            with ref_file.open() as f:
                saved_data = json.loads(f.read().strip())

            assert saved_data["boxes"] == []
            assert saved_data["labels"] == []
            assert saved_data["scores"] == []

    def test_save_reference_handles_diagnosis_fallback(self):
        """Test save_reference falls back to 'diagnosis' if 'final_diagnosis' missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            dataset_record = {
                "diagnosis": "Fallback diagnosis",
                # No final_diagnosis
            }
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = dataset_record

            save_reference(folder, 0, mock_dataset)

            ref_file = folder / "ref.jsonl"
            with ref_file.open() as f:
                saved_data = json.loads(f.read().strip())

            assert saved_data["diagnosis"] == "Fallback diagnosis"


@pytest.mark.unit
class TestDrawGroundTruthVsPredictedBoxes:
    """Test cases for bounding box visualization."""

    def test_draw_boxes_creates_visualization(self):
        """Test draw boxes creates visualization without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.png"
            output_path = Path(temp_dir) / "output.png"

            # Create a simple test image
            import numpy as np
            from PIL import Image

            test_image = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
            test_image.save(image_path)

            gt_boxes = [[10, 20, 30, 40]]
            pred_boxes = [[15, 25, 35, 45]]

            # Should run without errors
            draw_ground_truth_vs_predicted_boxes(image_path, gt_boxes, pred_boxes, output_path)

            # Output file should be created
            assert output_path.exists()

    def test_draw_boxes_handles_various_box_formats(self):
        """Test drawing handles different bounding box formats."""
        # This tests the _iter_boxes_generic function indirectly
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.png"
            output_path = Path(temp_dir) / "output.png"

            # Create a test image
            import numpy as np
            from PIL import Image

            test_image = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
            test_image.save(image_path)

            # Test different box formats
            dict_boxes = [
                {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
                {"x": 50, "y": 60, "width": 20, "height": 20},
            ]

            # Should handle dictionary format boxes without error
            draw_ground_truth_vs_predicted_boxes(image_path, dict_boxes, [], output_path)

            assert output_path.exists()


@pytest.mark.unit
class TestComputeEvaluationMetrics:
    """Test cases for evaluation metrics computation."""

    @patch("nova_retrieval_vlm.utils.batch_processing_utils.evaluate")
    def test_compute_evaluation_metrics_calls_evaluate(self, mock_evaluate):
        """Test compute_evaluation_metrics calls evaluation function."""
        mock_evaluate.return_value = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            # Create mock prediction and reference files
            (folder / "pred.jsonl").write_text('{"test": "pred"}\n')
            (folder / "ref.jsonl").write_text('{"test": "ref"}\n')

            compute_evaluation_metrics(folder, "localization")

            mock_evaluate.assert_called_once_with(
                str(folder / "pred.jsonl"), str(folder / "ref.jsonl"), task="localization"
            )

    def test_compute_evaluation_metrics_missing_files(self):
        """Test compute_evaluation_metrics raises error for missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            with pytest.raises(FileNotFoundError, match="Missing prediction or reference"):
                compute_evaluation_metrics(folder, "localization")

    @patch("nova_retrieval_vlm.utils.batch_processing_utils.evaluate")
    def test_compute_evaluation_saves_metrics(self, mock_evaluate):
        """Test compute_evaluation_metrics saves metrics to JSON file."""
        test_metrics = {"accuracy": 0.92, "f1": 0.88}
        mock_evaluate.return_value = test_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            # Create required files
            (folder / "pred.jsonl").write_text('{"test": "pred"}\n')
            (folder / "ref.jsonl").write_text('{"test": "ref"}\n')

            compute_evaluation_metrics(folder, "localization")  # Use valid task

            metrics_file = folder / "metrics.json"
            assert metrics_file.exists()

            with metrics_file.open() as f:
                saved_metrics = json.load(f)
                assert saved_metrics == test_metrics


@pytest.mark.integration
class TestPostprocessBatchResult:
    """Test cases for batch result post-processing."""

    @patch("nova_retrieval_vlm.utils.batch_processing_utils.normalize_localization_result")
    @patch("nova_retrieval_vlm.utils.batch_processing_utils.save_prediction")
    @patch("nova_retrieval_vlm.utils.batch_processing_utils.save_reference")
    @patch("nova_retrieval_vlm.utils.batch_processing_utils.draw_ground_truth_vs_predicted_boxes")
    @patch("nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics")
    def test_postprocess_batch_result_full_pipeline(
        self, mock_compute, mock_draw, mock_save_ref, mock_save_pred, mock_normalize
    ):
        """Test postprocess_batch_result executes full pipeline."""
        ctx = BatchContext(
            idx=0,
            folder=Path("/tmp/test"),
            img_path=Path("/tmp/test.png"),
            width=256,
            height=256,
        )

        prediction_result = {"boxes": [[10, 20, 30, 40]]}
        task_name = "localization"

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {
            "bbox_gold": {"x": [5], "y": [15], "width": [25], "height": [25]}
        }

        prediction_list = []

        postprocess_batch_result(ctx, prediction_result, task_name, mock_dataset, prediction_list)

        # Verify all steps were called
        mock_normalize.assert_called_once_with(prediction_result)
        mock_save_pred.assert_called_once_with(ctx.folder, prediction_result)
        mock_save_ref.assert_called_once_with(ctx.folder, ctx.idx, mock_dataset)
        mock_draw.assert_called_once()
        mock_compute.assert_called_once_with(ctx.folder, task_name)

        # Verify prediction was added to list
        assert prediction_result in prediction_list

    def test_postprocess_batch_result_prediction_list_update(self):
        """Test postprocess_batch_result updates prediction list."""
        with (
            patch("nova_retrieval_vlm.utils.batch_processing_utils.normalize_localization_result"),
            patch("nova_retrieval_vlm.utils.batch_processing_utils.save_prediction"),
            patch("nova_retrieval_vlm.utils.batch_processing_utils.save_reference"),
            patch(
                "nova_retrieval_vlm.utils.batch_processing_utils.draw_ground_truth_vs_predicted_boxes"
            ),
            patch("nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics"),
        ):
            ctx = BatchContext(
                idx=0,
                folder=Path("/tmp"),
                img_path=Path("/tmp/test.png"),
                width=100,
                height=100,
            )

            result = {"test": "prediction"}
            prediction_list = []
            mock_dataset = MagicMock()

            postprocess_batch_result(ctx, result, "test", mock_dataset, prediction_list)

            assert len(prediction_list) == 1
            assert prediction_list[0] == result


@pytest.mark.edge_case
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_normalize_with_none_values(self):
        """Test normalize handles None values gracefully."""
        result = {
            "boxes": None,
            "labels": None,
            "scores": None,
        }

        # Should not crash
        normalize_localization_result(result)

        # None should be replaced with empty list
        assert result["boxes"] == []

    def test_normalize_localization_malformed_data(self):
        """Test normalize_localization_result with malformed data."""
        result = {
            "localizations": [
                {"invalid": "data"},  # Missing bounding_box
                {"bounding_box": [1, 2, 3, 4]},  # Valid
            ]
        }

        # Should not crash, should handle gracefully
        normalize_localization_result(result)

        # Should extract only valid boxes
        assert len(result["boxes"]) == 1
        assert result["boxes"][0] == [1, 2, 3, 4]

    def test_save_prediction_with_large_data(self):
        """Test save_prediction handles large prediction data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)

            # Create large result with many boxes
            large_result = {
                "boxes": [[i, i + 1, i + 2, i + 3] for i in range(1000)],
                "labels": [f"label_{i}" for i in range(1000)],
                "scores": [0.5 + i / 2000 for i in range(1000)],
                "metadata": {"large_field": "x" * 10000},
            }

            # Should handle large data without issues
            save_prediction(folder, large_result)

            pred_file = folder / "pred.jsonl"
            assert pred_file.exists()
            assert pred_file.stat().st_size > 10000  # Should be substantial size
