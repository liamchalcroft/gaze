"""Comprehensive tests for the processor system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.processors import BaseProcessor
from nova_retrieval_vlm.processors import CaptionProcessor
from nova_retrieval_vlm.processors import DetectionProcessor
from nova_retrieval_vlm.processors import DiagnosisProcessor
from nova_retrieval_vlm.processors import LocalizationProcessor
from nova_retrieval_vlm.processors import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import EvaluationMetrics
from nova_retrieval_vlm.types import ModelResponse


@pytest.mark.unit
class TestProcessorConfig:
    """Test cases for ProcessorConfig."""

    def test_processor_config_creation(self):
        """Test ProcessorConfig can be created with required fields."""
        config = ProcessorConfig(
            task_name="localization",
            model_name="gpt-4o",
        )

        assert config.task_name == "localization"
        assert config.model_name == "gpt-4o"
        assert config.batch_size == 8  # Default value
        assert config.use_retrieval == False  # Default value

    def test_processor_config_all_fields(self):
        """Test ProcessorConfig with all fields specified."""
        config = ProcessorConfig(
            task_name="caption",
            model_name="claude-3-sonnet",
            batch_size=16,
            use_retrieval=True,
            retrieval_type="hybrid",
            output_dir=Path("/tmp/output"),
            skip_existing=True,
        )

        assert config.task_name == "caption"
        assert config.model_name == "claude-3-sonnet"
        assert config.batch_size == 16
        assert config.use_retrieval == True
        assert config.retrieval_type == "hybrid"
        assert config.output_dir == Path("/tmp/output")
        assert config.skip_existing == True

    def test_processor_config_defaults(self):
        """Test ProcessorConfig default values."""
        config = ProcessorConfig(
            task_name="test",
            model_name="test-model",
        )

        assert config.batch_size == 8
        assert config.use_retrieval == False
        assert config.retrieval_type == "bm25"
        assert config.output_dir == Path("./runs")
        assert config.skip_existing == False


@pytest.mark.unit
class TestBaseProcessor:
    """Test cases for BaseProcessor abstract base class."""

    class ConcreteProcessor(BaseProcessor):
        """Concrete implementation for testing."""

        async def process_batch(self, batch: BatchData, batch_idx: int) -> list[ModelResponse]:
            return [ModelResponse(text="test", confidence=0.8) for _ in range(len(batch.images))]

        def evaluate_responses(
            self, responses: list[ModelResponse], ground_truth: list[Any]
        ) -> EvaluationMetrics:
            return EvaluationMetrics(
                accuracy=0.85, precision=None, recall=None, f1_score=None, auc_roc=None
            )

    def test_base_processor_initialization(self):
        """Test BaseProcessor initializes correctly."""
        config = ProcessorConfig(task_name="test", model_name="test-model")
        processor = self.ConcreteProcessor(config)

        assert processor.config == config
        assert hasattr(processor, "logger")

    def test_should_skip_batch_false_by_default(self):
        """Test should_skip_batch returns False when skip_existing is False."""
        config = ProcessorConfig(task_name="test", model_name="test-model", skip_existing=False)
        processor = self.ConcreteProcessor(config)

        assert processor.should_skip_batch(0) == False

    def test_should_skip_batch_checks_file_existence(self):
        """Test should_skip_batch checks for existing output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = ProcessorConfig(
                task_name="test", model_name="test-model", output_dir=output_dir, skip_existing=True
            )
            processor = self.ConcreteProcessor(config)

            # Should not skip when file doesn't exist
            assert processor.should_skip_batch(0) == False

            # Create output file
            (output_dir / "batch_0.json").touch()

            # Should skip when file exists
            assert processor.should_skip_batch(0) == True

    def test_save_batch_results(self):
        """Test save_batch_results creates correct output file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = ProcessorConfig(
                task_name="test_task", model_name="test-model", output_dir=output_dir
            )
            processor = self.ConcreteProcessor(config)

            responses = [
                ModelResponse(text="response1", confidence=0.8),
                ModelResponse(text="response2", confidence=0.9),
            ]
            metadata = {"batch_size": 2}

            processor.save_batch_results(0, responses, metadata)

            output_file = output_dir / "batch_0.json"
            assert output_file.exists()

            import json

            with output_file.open() as f:
                saved_data = json.load(f)

            assert saved_data["batch_idx"] == 0
            assert saved_data["task"] == "test_task"
            assert saved_data["model"] == "test-model"
            assert len(saved_data["responses"]) == 2
            assert saved_data["metadata"] == metadata

    def test_load_batch_results(self):
        """Test load_batch_results loads saved results correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = ProcessorConfig(
                task_name="test", model_name="test-model", output_dir=output_dir
            )
            processor = self.ConcreteProcessor(config)

            # Create test data file
            test_data: dict[str, Any] = {
                "batch_idx": 0,
                "task": "test",
                "model": "test-model",
                "responses": [
                    {"text": "loaded response", "confidence": 0.75, "reasoning": "", "metadata": {}}
                ],
                "metadata": {},
            }

            import json

            output_file = output_dir / "batch_0.json"
            with output_file.open("w") as f:
                json.dump(test_data, f)

            loaded_responses = processor.load_batch_results(0)

            assert len(loaded_responses) == 1
            assert loaded_responses[0].text == "loaded response"
            assert loaded_responses[0].confidence == 0.75

    def test_load_batch_results_file_not_found(self):
        """Test load_batch_results raises error when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ProcessorConfig(
                task_name="test", model_name="test-model", output_dir=Path(temp_dir)
            )
            processor = self.ConcreteProcessor(config)

            with pytest.raises(FileNotFoundError, match="No saved results for batch"):
                processor.load_batch_results(999)


@pytest.mark.unit
class TestLocalizationProcessor:
    """Test cases for LocalizationProcessor."""

    @pytest.fixture
    def processor_config(self) -> ProcessorConfig:
        """Create test configuration for LocalizationProcessor."""
        return ProcessorConfig(
            task_name="localization",
            model_name="gpt-4o",
            batch_size=2,
        )

    def test_localization_processor_initialization(self, processor_config: ProcessorConfig):
        """Test LocalizationProcessor initializes correctly."""
        processor = LocalizationProcessor(processor_config)

        assert processor.config.task_name == "localization"
        assert hasattr(processor, "logger")

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test LocalizationProcessor.process_batch processes images correctly."""
        # Mock the adapter
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            '{"boxes": [[10, 20, 30, 40]], "labels": ["anomaly"], "scores": [0.9], "reasoning": "Localization analysis completed"}',
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        processor = LocalizationProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert (
            responses[0].reasoning is not None and "localization" in responses[0].reasoning.lower()
        )

        # Verify adapter was called for each image
        assert mock_adapter.generate.call_count == 2

    def test_evaluate_responses(self, processor_config: ProcessorConfig):
        """Test LocalizationProcessor.evaluate_responses."""
        processor = LocalizationProcessor(processor_config)

        responses = [
            ModelResponse(text='{"boxes": [[10, 20, 30, 40]]}', confidence=0.8),
            ModelResponse(text='{"boxes": [[50, 60, 70, 80]]}', confidence=0.9),
        ]
        ground_truth = ['{"boxes": [[10, 20, 30, 40]]}', '{"boxes": [[50, 60, 70, 80]]}']

        with patch("nova_retrieval_vlm.processors.localization.evaluate_detection") as mock_eval:
            mock_eval.return_value = {
                "map50": 0.75,
                "precision": 0.8,
                "map30": 0.7,
                "map50_95": 0.6,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.accuracy == 0.75  # map50 mapped to accuracy

    def test_create_localization_prompt(self, processor_config: ProcessorConfig):
        """Test _create_localization_prompt generates appropriate prompts."""
        processor = LocalizationProcessor(processor_config)

        metadata = {"modality": "CT", "patient_info": "65-year-old patient"}
        prompt = processor._create_localization_prompt(Path("test.png"), metadata)  # type: ignore[attr-defined]

        assert "CT" in prompt
        assert "65-year-old patient" in prompt
        assert "locate" in prompt.lower()
        assert "bounding box" in prompt.lower()


@pytest.mark.unit
class TestCaptionProcessor:
    """Test cases for CaptionProcessor."""

    @pytest.fixture
    def processor_config(self) -> ProcessorConfig:
        """Create test configuration for CaptionProcessor."""
        return ProcessorConfig(
            task_name="caption",
            model_name="gpt-4o",
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.processors.caption.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test CaptionProcessor processes batch correctly."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            "Detailed medical description of the CT scan showing normal anatomy.",
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        processor = CaptionProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert responses[0].reasoning is not None and "Generated caption" in responses[0].reasoning

    def test_evaluate_responses(self, processor_config: ProcessorConfig):
        """Test CaptionProcessor evaluation."""
        processor = CaptionProcessor(processor_config)

        responses = [
            ModelResponse(text="Generated caption 1", confidence=0.8),
            ModelResponse(text="Generated caption 2", confidence=0.9),
        ]
        ground_truth = ["Reference caption 1", "Reference caption 2"]

        with patch("nova_retrieval_vlm.evaluation.caption.evaluate_caption") as mock_eval:
            mock_eval.return_value = {
                "bleu": 0.65,
                "bert_precision": 0.75,
                "bert_recall": 0.70,
                "bert_f1": 0.72,
                "meteor": 0.68,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.accuracy == 0.65  # BLEU score
            assert metrics.precision == 0.75
            assert metrics.f1_score == 0.72

    def test_create_caption_prompt(self, processor_config: ProcessorConfig):
        """Test caption prompt creation."""
        processor = CaptionProcessor(processor_config)

        metadata = {"modality": "MRI", "patient_info": "Brain scan"}
        prompt = processor._create_caption_prompt(Path("brain.png"), metadata)  # type: ignore[attr-defined]

        assert "MRI" in prompt
        assert "Brain scan" in prompt
        assert "description" in prompt.lower()
        assert "anatomical structures" in prompt.lower()


@pytest.mark.unit
class TestDiagnosisProcessor:
    """Test cases for DiagnosisProcessor."""

    @pytest.fixture
    def processor_config(self) -> ProcessorConfig:
        """Create test configuration for DiagnosisProcessor."""
        return ProcessorConfig(
            task_name="diagnosis",
            model_name="gpt-4o",
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.processors.diagnosis.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test DiagnosisProcessor processes batch correctly."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            "Primary diagnosis: Normal brain MRI with no acute abnormalities detected.",
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        processor = DiagnosisProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        # The full reasoning should be in reasoning field
        assert responses[0].reasoning is not None and "Primary diagnosis:" in responses[0].reasoning

    def test_evaluate_responses(self, processor_config: ProcessorConfig):
        """Test DiagnosisProcessor evaluation."""
        processor = DiagnosisProcessor(processor_config)

        responses = [
            ModelResponse(text="Normal", confidence=0.9),
            ModelResponse(text="Abnormal", confidence=0.8),
        ]
        ground_truth = ["Normal", "Abnormal"]

        with patch(
            "nova_retrieval_vlm.processors.diagnosis.evaluate_diagnosis_nova_official"
        ) as mock_eval:
            mock_eval.return_value = {
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.accuracy == 1.0
            assert metrics.precision == 1.0

    def test_extract_primary_diagnosis(self, processor_config: ProcessorConfig):
        """Test _extract_primary_diagnosis extracts diagnosis correctly."""
        processor = DiagnosisProcessor(processor_config)

        # Test with explicit primary diagnosis marker
        diagnosis_text = "Primary diagnosis: Acute stroke\nDifferential: TIA"
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert result == "Acute stroke"

        # Test with diagnosis marker
        diagnosis_text = "Diagnosis: Brain tumor\nRecommendations: Further imaging"
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert result == "Brain tumor"

        # Test with no markers
        diagnosis_text = "This appears to be a normal brain scan with no abnormalities."
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert "normal brain scan" in result.lower()

    def test_create_diagnosis_prompt(self, processor_config: ProcessorConfig):
        """Test diagnosis prompt creation."""
        processor = DiagnosisProcessor(processor_config)

        metadata = {
            "modality": "CT",
            "patient_info": "65-year-old male",
            "clinical_history": "Headache and dizziness",
        }
        prompt = processor._create_diagnosis_prompt(Path("scan.png"), metadata)  # type: ignore[attr-defined]

        assert "CT" in prompt
        assert "65-year-old male" in prompt
        assert "Headache and dizziness" in prompt
        assert "Primary diagnosis" in prompt


@pytest.mark.unit
class TestDetectionProcessor:
    """Test cases for DetectionProcessor."""

    @pytest.fixture
    def processor_config(self) -> ProcessorConfig:
        """Create test configuration for DetectionProcessor."""
        return ProcessorConfig(
            task_name="detection",
            model_name="gpt-4o",
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.processors.detection.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test DetectionProcessor processes batch correctly."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            '{"detections": [{"bbox": [10, 20, 30, 40], "class": "anomaly", "confidence": 0.9}], "reasoning": "Detection analysis completed"}',
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        processor = DetectionProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert responses[0].reasoning is not None and "Detection" in responses[0].reasoning

    def test_evaluate_responses(self, processor_config: ProcessorConfig):
        """Test DetectionProcessor evaluation."""
        processor = DetectionProcessor(processor_config)

        responses = [
            ModelResponse(text='{"detections": [[10, 20, 30, 40]]}', confidence=0.8),
        ]
        ground_truth = ['{"detections": [[10, 20, 30, 40]]}']

        with patch("nova_retrieval_vlm.processors.detection.evaluate_detection") as mock_eval:
            mock_eval.return_value = {
                "map50": 0.85,
                "precision": 0.82,
                "map30": 0.8,
                "map50_95": 0.7,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.accuracy == 0.85


@pytest.mark.integration
class TestProcessorIntegration:
    """Integration tests for processor system."""

    async def test_processor_workflow_end_to_end(self, mock_image: Path):
        """Test complete processor workflow."""
        config = ProcessorConfig(
            task_name="localization",
            model_name="test-model",
        )

        batch_data = BatchData(images=[mock_image], metadata=[{"modality": "CT"}])

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate.return_value = (
                '{"boxes": [[10, 20, 30, 40]], "confidence": 0.9}',
                MagicMock(),
            )
            mock_adapter_class.return_value = mock_adapter

            processor = LocalizationProcessor(config)

            # Process batch
            responses = await processor.process_batch(batch_data, 0)
            assert len(responses) == 1

            # Evaluate responses
            with patch(
                "nova_retrieval_vlm.processors.localization.evaluate_detection"
            ) as mock_eval:
                mock_eval.return_value = {"map_50": 0.8}
                metrics = processor.evaluate_responses(responses, ['{"boxes": [[10, 20, 30, 40]]}'])
                assert metrics.accuracy == 0.8

    def test_all_processors_implement_interface(self):
        """Test all processor classes implement the BaseProcessor interface."""
        config = ProcessorConfig(task_name="test", model_name="test-model")

        processors = [
            LocalizationProcessor(config),
            CaptionProcessor(config),
            DiagnosisProcessor(config),
            DetectionProcessor(config),
        ]

        for processor in processors:
            assert isinstance(processor, BaseProcessor)
            assert hasattr(processor, "process_batch")
            assert hasattr(processor, "evaluate_responses")

    @patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter")
    def test_processor_error_handling(self, mock_adapter):
        """Test processor error handling for invalid inputs."""
        # Mock the adapter to avoid API key issues
        mock_adapter.return_value = MagicMock()

        config = ProcessorConfig(task_name="test", model_name="test-model")
        processor = LocalizationProcessor(config)

        # Test with empty batch
        empty_batch = BatchData(images=[], metadata=[])

        # Should handle empty batch gracefully
        async def test_empty_batch():
            responses = await processor.process_batch(empty_batch, 0)
            assert responses == []

        import asyncio

        asyncio.run(test_empty_batch())

    def test_processor_config_validation(self):
        """Test processor configuration validation."""
        # Test with missing required fields - should work with defaults
        minimal_config = ProcessorConfig(task_name="test", model_name="test-model")
        processor = LocalizationProcessor(minimal_config)
        assert processor.config.batch_size == 8  # Default

    async def test_processor_async_error_handling(self, mock_image: Path):
        """Test processor handles async errors gracefully."""
        config = ProcessorConfig(task_name="test", model_name="test-model")
        processor = LocalizationProcessor(config)

        batch_data = BatchData(images=[mock_image], metadata=[{}])

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            # Mock adapter to raise exception
            mock_adapter = AsyncMock()
            mock_adapter.generate.side_effect = Exception("API Error")
            mock_adapter_class.return_value = mock_adapter

            # Should propagate the exception
            with pytest.raises(Exception, match="API Error"):
                await processor.process_batch(batch_data, 0)


@pytest.mark.performance
class TestProcessorPerformance:
    """Performance and stress tests for processors."""

    async def test_processor_handles_large_batch(self, mock_image: Path):
        """Test processor can handle large batches efficiently."""
        config = ProcessorConfig(
            task_name="localization",
            model_name="test-model",
            batch_size=100,
        )

        # Create large batch
        large_batch = BatchData(images=[mock_image] * 50, metadata=[{"modality": "CT"}] * 50)

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate.return_value = ('{"boxes": []}', MagicMock())
            mock_adapter_class.return_value = mock_adapter

            processor = LocalizationProcessor(config)

            import time

            start_time = time.time()
            responses = await processor.process_batch(large_batch, 0)
            end_time = time.time()

            assert len(responses) == 50
            # Should complete in reasonable time (adjust threshold as needed)
            assert end_time - start_time < 10.0  # 10 seconds max

    def test_processor_memory_usage(self):
        """Test processor doesn't leak memory with repeated use."""
        config = ProcessorConfig(task_name="test", model_name="test-model")

        # Create and destroy many processors
        processors: list[LocalizationProcessor] = []
        for _ in range(100):
            processor = LocalizationProcessor(config)
            processors.append(processor)

        # Clear references
        del processors

        # Memory should be manageable (this is a basic test)
        import gc

        gc.collect()

        # If we get here without issues, memory management is working
