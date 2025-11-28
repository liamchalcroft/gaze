"""Tests for the evaluation module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.evaluation import evaluate


class TestEvaluation:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        dir_path = tempfile.mkdtemp()
        yield dir_path
        # Clean up
        for path in Path(dir_path).glob("*"):
            path.unlink()
        os.rmdir(dir_path)

    @pytest.fixture
    def sample_preds_file(self, temp_dir):
        """Create a sample predictions file."""
        file_path = Path(temp_dir) / "preds.jsonl"

        preds = [
            {
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "labels": ["anomaly", "anomaly"],
                "scores": [0.9, 0.8],
                "caption": "Sample prediction caption",
                "diagnosis": "Sample diagnosis",
            },
            {
                "boxes": [[15, 25, 35, 45]],
                "labels": ["anomaly"],
                "scores": [0.95],
                "caption": "Another prediction caption",
                "diagnosis": "Another diagnosis",
            },
        ]

        with open(file_path, "w") as f:
            for pred in preds:
                f.write(json.dumps(pred) + "\n")

        return str(file_path)

    @pytest.fixture
    def sample_refs_file(self, temp_dir):
        """Create a sample references file."""
        file_path = Path(temp_dir) / "refs.jsonl"

        refs = [
            {
                "boxes": [[12, 22, 32, 42], [52, 62, 72, 82]],
                "labels": ["anomaly", "anomaly"],
                "scores": [1.0, 1.0],
                "caption": "Sample reference caption",
                "diagnosis": "Sample reference diagnosis",
            },
            {
                "boxes": [[18, 28, 38, 48]],
                "labels": ["anomaly"],
                "scores": [1.0],
                "caption": "Another reference caption",
                "diagnosis": "Another reference diagnosis",
            },
        ]

        with open(file_path, "w") as f:
            for ref in refs:
                f.write(json.dumps(ref) + "\n")

        return str(file_path)

    @patch("nova_retrieval_vlm.evaluation.evaluate_detection")
    def test_evaluate_localization(
        self, mock_evaluate_detection, sample_preds_file, sample_refs_file
    ):
        """Test the evaluate function for localization task."""
        # Configure the mock with NOVA protocol metrics
        mock_evaluate_detection.return_value = {
            "map30": 0.85,
            "map50": 0.9,
            "map50_95": 0.8,
            "acc50": 0.75,
        }

        # Call the function
        metrics = evaluate(sample_preds_file, sample_refs_file, task="localization")

        # Verify the result includes all NOVA metrics
        assert "detection_mAP30" in metrics
        assert "detection_mAP50" in metrics
        assert "detection_mAP50_95" in metrics
        assert "detection_ACC50" in metrics
        assert metrics["detection_mAP30"] == 0.85
        assert metrics["detection_mAP50"] == 0.9
        assert metrics["detection_mAP50_95"] == 0.8
        assert metrics["detection_ACC50"] == 0.75

        # Verify the mock was called
        mock_evaluate_detection.assert_called_once()

    @patch("nova_retrieval_vlm.evaluation.caption.evaluate_caption")
    def test_evaluate_caption(self, mock_evaluate_caption, sample_preds_file, sample_refs_file):
        """Test the evaluate function for caption task."""
        # Configure the mocks
        mock_evaluate_caption.return_value = {
            "bleu": 42.5,
            "bert_f1": 0.75,
            "radgraph_f1": 0.80,
            "meteor": 0.65,
            "modality_f1": 0.85,
            "clinical_f1": 0.72,
            "binary_f1": 0.90,
        }

        # Call the function
        metrics = evaluate(sample_preds_file, sample_refs_file, task="caption")

        # Verify the result
        assert "caption_bleu" in metrics
        assert "caption_bert_f1" in metrics
        assert metrics["caption_bleu"] == 42.5
        assert metrics["caption_bert_f1"] == 0.75

        # Verify the mock was called
        mock_evaluate_caption.assert_called_once()

    @patch("nova_retrieval_vlm.evaluation.evaluate_diagnosis_nova_official")
    def test_evaluate_diagnosis(self, mock_evaluate_diagnosis, sample_preds_file, sample_refs_file):
        """Test the evaluate function for diagnosis task."""
        # Configure the mocks with NOVA protocol metrics
        mock_evaluate_diagnosis.return_value = {
            "top1": 0.85,
            "top5": 0.92,
            "coverage": 0.80,
            "entropy": 2.5,
        }

        # Call the function
        metrics = evaluate(sample_preds_file, sample_refs_file, task="diagnosis")

        # Verify the result includes all NOVA metrics
        assert "diagnosis_top1" in metrics
        assert "diagnosis_top5" in metrics
        assert "diagnosis_coverage" in metrics
        assert "diagnosis_entropy" in metrics
        assert metrics["diagnosis_top1"] == 0.85
        assert metrics["diagnosis_top5"] == 0.92
        assert metrics["diagnosis_coverage"] == 0.80
        assert metrics["diagnosis_entropy"] == 2.5

        # Verify the mock was called
        mock_evaluate_diagnosis.assert_called_once()

    def test_evaluate_unknown_task(self, sample_preds_file, sample_refs_file):
        """Test the evaluate function with an unknown task."""
        # Call the function with an unknown task
        with pytest.raises(ValueError, match="Unknown task: unknown_task"):
            evaluate(sample_preds_file, sample_refs_file, task="unknown_task")
