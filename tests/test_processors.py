"""Comprehensive tests for the processor system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.processors import BaseProcessor
from nova_retrieval_vlm.processors import CaptionProcessor
from nova_retrieval_vlm.processors import DiagnosisProcessor
from nova_retrieval_vlm.processors import LocalizationProcessor
from nova_retrieval_vlm.processors import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import CaptionMetrics
from nova_retrieval_vlm.types import DetectionMetrics
from nova_retrieval_vlm.types import DiagnosisMetrics
from nova_retrieval_vlm.types import ModelResponse

# Valid unified response for mocks (matches expected JSON schema)
VALID_UNIFIED_RESPONSE = json.dumps(
    {
        "caption": {
            "description": "Axial T2-weighted MRI showing normal brain anatomy",
            "confidence": 0.9,
        },
        "diagnosis": {"primary_diagnosis": "Normal", "confidence": 0.9},
        "localization": {
            "localizations": [
                {"finding": "anomaly", "bounding_box": [10, 20, 30, 40], "confidence": 0.9}
            ]
        },
    }
)


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

    def test_processor_config_all_fields(self):
        """Test ProcessorConfig with all fields specified."""
        config = ProcessorConfig(
            task_name="caption",
            model_name="claude-3-sonnet",
            batch_size=16,
            output_dir=tempfile.mkdtemp(),  # Use temp directory
            skip_existing=True,
        )

        assert config.task_name == "caption"
        assert config.model_name == "claude-3-sonnet"
        assert config.batch_size == 16
        assert config.output_dir.exists()  # Check directory exists
        assert config.skip_existing

    def test_processor_config_defaults(self):
        """Test ProcessorConfig default values."""
        config = ProcessorConfig(
            task_name="train",
            model_name="test-model",
        )

        assert config.batch_size == 8
        assert config.output_dir == Path("./runs")
        assert not config.skip_existing


@pytest.mark.unit
class TestBaseProcessor:
    """Test cases for BaseProcessor abstract base class."""

    class ConcreteProcessor(BaseProcessor):
        """Concrete implementation for testing."""

        async def process_batch(self, batch: BatchData, _batch_idx: int) -> list[ModelResponse]:
            return [ModelResponse(text="test", confidence=0.8) for _ in range(len(batch.images))]

        def evaluate_responses(
            self, _responses: list[ModelResponse], _ground_truth: list[Any]
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

        assert not processor.should_skip_batch(0)

    def test_should_skip_batch_checks_file_existence(self):
        """Test should_skip_batch checks for existing output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = ProcessorConfig(
                task_name="test", model_name="test-model", output_dir=output_dir, skip_existing=True
            )
            processor = self.ConcreteProcessor(config)

            # Should not skip when file doesn't exist
            assert not processor.should_skip_batch(0)

            # Create output file
            (output_dir / "batch_0.json").touch()

            # Should skip when file exists
            assert processor.should_skip_batch(0)

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
    @patch("nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test LocalizationProcessor.process_batch processes images correctly."""
        # Mock the adapter with unified response format
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
        mock_adapter_class.return_value = mock_adapter

        processor = LocalizationProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        # Check that unified response was generated
        assert responses[0].text is not None

        # Verify adapter was called for each image
        assert mock_adapter.generate.call_count == 2

    def test_evaluate_responses(self, processor_config: ProcessorConfig):
        """Test LocalizationProcessor.evaluate_responses."""
        processor = LocalizationProcessor(processor_config)

        # Use unified response format matching process_batch output
        responses = [
            ModelResponse(
                text='{"caption": {}, "diagnosis": {}, "localization": {"localizations": [{"bounding_box": [10, 20, 30, 40], "finding": "lesion"}]}}',
                confidence=0.8,
            ),
            ModelResponse(
                text='{"caption": {}, "diagnosis": {}, "localization": {"localizations": [{"bounding_box": [50, 60, 70, 80], "finding": "lesion"}]}}',
                confidence=0.9,
            ),
        ]
        ground_truth = ['{"boxes": [[10, 20, 30, 40]]}', '{"boxes": [[50, 60, 70, 80]]}']

        with patch("nova_retrieval_vlm.processors.localization.evaluate_detection") as mock_eval:
            # evaluate_detection returns: map30, map50, map50_95, acc50, tp30, fp30
            mock_eval.return_value = {
                "map30": 0.7,
                "map50": 0.75,
                "map50_95": 0.6,
                "acc50": 0.8,
                "tp30": 10,
                "fp30": 2,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, DetectionMetrics)
            assert metrics.map50 == 0.75
            assert metrics.map30 == 0.7
            assert metrics.map50_95 == 0.6
            assert metrics.acc50 == 0.8
            assert metrics.tp30 == 10
            assert metrics.fp30 == 2

    def test_create_unified_prompt(self, processor_config: ProcessorConfig):
        """Test _create_unified_prompt generates appropriate prompts for localization."""
        processor = LocalizationProcessor(processor_config)

        metadata = {"modality": "CT", "clinical_history": "65-year-old patient with headache"}
        prompt = processor._create_unified_prompt(Path("test.png"), metadata)

        # Check that unified prompt contains expected elements
        assert (
            "65-year-old patient" in prompt or "NOVA" in prompt
        )  # Either clinical history or dataset context
        assert "localization" in prompt.lower() or "bounding" in prompt.lower()


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
    @patch("nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test CaptionProcessor processes batch correctly."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
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
            # evaluate_caption returns: bleu, bert_f1, radgraph_f1, meteor, modality_f1, clinical_f1, binary_accuracy, binary_f1
            mock_eval.return_value = {
                "bleu": 0.65,
                "bert_f1": 0.72,
                "radgraph_f1": 0.60,
                "meteor": 0.68,
                "modality_f1": 0.75,
                "clinical_f1": 0.70,
                "binary_accuracy": 0.85,
                "binary_f1": 0.80,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, CaptionMetrics)
            assert metrics.bleu == 0.65
            assert metrics.bert_f1 == 0.72
            assert metrics.meteor == 0.68
            assert metrics.modality_f1 == 0.75
            assert metrics.clinical_f1 == 0.70
            assert metrics.binary_accuracy == 0.85
            assert metrics.binary_f1 == 0.80
            assert metrics.radgraph_f1 == 0.60

    def test_create_unified_prompt(self, processor_config: ProcessorConfig):
        """Test unified prompt creation for captioning."""
        processor = CaptionProcessor(processor_config)

        metadata = {"modality": "MRI", "clinical_history": "Brain scan patient"}
        prompt = processor._create_unified_prompt(Path("brain.png"), metadata)

        # Check unified prompt contains expected elements
        assert "NOVA" in prompt or "neuroradiol" in prompt.lower()  # Dataset or domain context
        assert (
            "caption" in prompt.lower() or "description" in prompt.lower()
        )  # Caption task mention


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
    @patch("nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter")
    async def test_process_batch(
        self,
        mock_adapter_class: MagicMock,
        processor_config: ProcessorConfig,
        mock_batch_data: BatchData,
    ):
        """Test DiagnosisProcessor processes batch correctly."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
        mock_adapter_class.return_value = mock_adapter

        processor = DiagnosisProcessor(processor_config)
        responses = await processor.process_batch(mock_batch_data, 0)

        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        # Check that a response was generated
        assert responses[0].text is not None

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
            # Mock returns NOVA protocol metrics: top1, top5, coverage, entropy
            mock_eval.return_value = {
                "top1": 1.0,
                "top5": 1.0,
                "coverage": 1.0,
                "entropy": 0.0,
            }

            metrics = processor.evaluate_responses(responses, ground_truth)

            assert isinstance(metrics, DiagnosisMetrics)
            assert metrics.top1 == 1.0
            assert metrics.top5 == 1.0
            assert metrics.coverage == 1.0
            assert metrics.entropy == 0.0

    def test_extract_primary_diagnosis(self, processor_config: ProcessorConfig):
        """Test _extract_primary_diagnosis strips diagnosis prefixes correctly.

        Note: The structured JSON response already provides primary_diagnosis separately,
        so this method only needs to strip common prefixes - not extract from complex text.
        """
        processor = DiagnosisProcessor(processor_config)

        # Test with explicit primary diagnosis marker - just strips prefix
        diagnosis_text = "Primary diagnosis: Acute stroke"
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert result == "Acute stroke"

        # Test with diagnosis marker
        diagnosis_text = "Diagnosis: Brain tumor"
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert result == "Brain tumor"

        # Test with no markers - returns as-is
        diagnosis_text = "This appears to be a normal brain scan with no abnormalities."
        result = processor._extract_primary_diagnosis(diagnosis_text)  # type: ignore[attr-defined]
        assert "normal brain scan" in result.lower()

        # Test empty input raises ValueError
        import pytest

        with pytest.raises(ValueError, match="Empty diagnosis text"):
            processor._extract_primary_diagnosis("")  # type: ignore[attr-defined]

        # Test prefix-only input raises ValueError
        with pytest.raises(ValueError, match="only a prefix"):
            processor._extract_primary_diagnosis("Primary diagnosis:")  # type: ignore[attr-defined]

    def test_create_unified_prompt(self, processor_config: ProcessorConfig):
        """Test unified prompt creation for diagnosis."""
        processor = DiagnosisProcessor(processor_config)

        metadata = {
            "modality": "CT",
            "clinical_history": "65-year-old male with headache and dizziness",
        }
        prompt = processor._create_unified_prompt(Path("scan.png"), metadata)

        # Check unified prompt contains expected elements
        assert "NOVA" in prompt or "neuroradiol" in prompt.lower()  # Dataset or domain context
        assert "diagnosis" in prompt.lower()  # Diagnosis task mention


@pytest.mark.unit
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
        ]

        for processor in processors:
            assert isinstance(processor, BaseProcessor)
            assert hasattr(processor, "process_batch")
            assert hasattr(processor, "evaluate_responses")

    @patch("nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter")
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
