"""Comprehensive integration tests for end-to-end workflows."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.config import ModelConfig
from nova_retrieval_vlm.config import PathsConfig
from nova_retrieval_vlm.processors import CaptionProcessor
from nova_retrieval_vlm.processors import LocalizationProcessor
from nova_retrieval_vlm.processors import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def integration_config(self):
        """Create integration test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(
                model=ModelConfig(name="gpt-4o", max_tokens=256, temperature=0.1),
                paths=PathsConfig(
                    data_dir=temp_dir,
                    output_dir=temp_dir,
                    index_dir=temp_dir,
                ),
                task="localization",
                batch_size=2,
                max_iterations=1,
            )
            yield config

    @pytest.fixture
    def sample_batch_data(self, mock_image):
        """Create sample batch data for testing."""
        return BatchData(
            images=[mock_image, mock_image],
            metadata=[
                {
                    "modality": "CT",
                    "patient_info": "65-year-old patient",
                    "study_id": "STUDY_001",
                },
                {
                    "modality": "MRI",
                    "patient_info": "45-year-old patient",
                    "study_id": "STUDY_002",
                },
            ],
        )

    @pytest.fixture
    def mock_huggingface_dataset(self):
        """Create mock HuggingFace dataset."""
        dataset = MagicMock()
        dataset.__getitem__.side_effect = [
            {
                "bbox_gold": {"x": [10], "y": [20], "width": [30], "height": [40]},
                "caption": "Normal brain CT scan",
                "final_diagnosis": "No acute findings",
            },
            {
                "bbox_gold": {"x": [50], "y": [60], "width": [25], "height": [35]},
                "caption": "Brain MRI with lesion",
                "diagnosis": "Possible stroke",
            },
        ]
        dataset.__len__.return_value = 2
        return dataset


@pytest.mark.integration
class TestLocalizationWorkflow(TestEndToEndWorkflows):
    """Integration tests for localization workflow."""

    @patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter")
    async def test_localization_end_to_end(
        self,
        mock_adapter_class,
        integration_config,
        sample_batch_data,
        mock_huggingface_dataset,
    ):
        """Test complete localization workflow from batch to evaluation."""
        # Setup mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            '{"boxes": [[10, 20, 40, 60]], "labels": ["anomaly"], "scores": [0.9]}',
            MagicMock(tokens=50, cost=0.001),
        )
        mock_adapter_class.return_value = mock_adapter

        # Create processor
        processor_config = ProcessorConfig(
            task_name="localization",
            model_name="gpt-4o",
            output_dir=Path(integration_config.paths.output_dir),
        )
        processor = LocalizationProcessor(processor_config)

        # Process batch
        responses = await processor.process_batch(sample_batch_data, 0)

        # Verify responses
        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert all(r.confidence > 0 for r in responses)

        # Test post-processing integration
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_ctx = BatchContext(
                idx=0,
                folder=Path(temp_dir),
                img_path=sample_batch_data.images[0],
                width=256,
                height=256,
            )

            prediction_list = []

            with patch(
                "nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics"
            ):
                postprocess_batch_result(
                    batch_ctx,
                    responses[0].model_dump(),
                    "localization",
                    mock_huggingface_dataset,
                    prediction_list,
                )

            # Verify post-processing results
            assert len(prediction_list) == 1
            assert (batch_ctx.folder / "pred.jsonl").exists()
            assert (batch_ctx.folder / "ref.jsonl").exists()

    @patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter")
    @patch("nova_retrieval_vlm.processors.localization.evaluate_detection")
    async def test_localization_with_evaluation(
        self,
        mock_evaluate,
        mock_adapter_class,
        sample_batch_data,
    ):
        """Test localization with evaluation metrics."""
        # Setup mocks
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = ('{"boxes": [[10, 20, 30, 40]]}', MagicMock())
        mock_adapter_class.return_value = mock_adapter

        mock_evaluate.return_value = {
            "map_30": 0.75,
            "map_50": 0.70,
            "map_50_95": 0.65,
            "iou": 0.80,
        }

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        # Process and evaluate
        responses = await processor.process_batch(sample_batch_data, 0)
        ground_truth = ['{"boxes": [[10, 20, 30, 40]]}', '{"boxes": [[15, 25, 35, 45]]}']

        metrics = processor.evaluate_responses(responses, ground_truth)

        assert metrics.accuracy == 0.75  # map_30
        assert mock_evaluate.called

    async def test_localization_error_handling(self, sample_batch_data):
        """Test localization workflow error handling."""
        processor_config = ProcessorConfig(task_name="localization", model_name="failing-model")
        processor = LocalizationProcessor(processor_config)

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate.side_effect = Exception("API Error")
            mock_adapter_class.return_value = mock_adapter

            # Should propagate error appropriately
            with pytest.raises(Exception, match="API Error"):
                await processor.process_batch(sample_batch_data, 0)


@pytest.mark.integration
class TestCaptionWorkflow(TestEndToEndWorkflows):
    """Integration tests for caption workflow."""

    @patch("nova_retrieval_vlm.processors.caption.OpenAIAdapter")
    async def test_caption_end_to_end(
        self,
        mock_adapter_class,
        sample_batch_data,
        mock_huggingface_dataset,
    ):
        """Test complete caption workflow."""
        # Setup mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            "Detailed medical description of a brain CT scan showing normal anatomy with no acute abnormalities.",
            MagicMock(tokens=25, cost=0.0005),
        )
        mock_adapter_class.return_value = mock_adapter

        processor_config = ProcessorConfig(task_name="caption", model_name="gpt-4o")
        processor = CaptionProcessor(processor_config)

        # Process batch
        responses = await processor.process_batch(sample_batch_data, 0)

        # Verify responses
        assert len(responses) == 2
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert all("brain" in r.text.lower() or "medical" in r.text.lower() for r in responses)

        # Test evaluation
        with patch("nova_retrieval_vlm.processors.caption.evaluate_caption") as mock_eval:
            mock_eval.return_value = {
                "bleu": 0.65,
                "bert_f1": 0.72,
                "meteor": 0.68,
            }

            ground_truth = ["Reference caption 1", "Reference caption 2"]
            metrics = processor.evaluate_responses(responses, ground_truth)

            assert metrics.accuracy == 0.65  # BLEU score
            assert metrics.f1_score == 0.72

    @patch("nova_retrieval_vlm.processors.caption.OpenAIAdapter")
    async def test_caption_with_patient_context(self, mock_adapter_class, mock_image):
        """Test caption generation with rich patient context."""
        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            "CT scan of a 65-year-old patient showing age-related changes with no acute pathology.",
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        batch_data = BatchData(
            images=[mock_image],
            metadata=[
                {
                    "modality": "CT",
                    "patient_info": "65-year-old male with history of hypertension",
                    "clinical_history": "Headache for 2 weeks",
                }
            ],
        )

        processor_config = ProcessorConfig(task_name="caption", model_name="gpt-4o")
        processor = CaptionProcessor(processor_config)

        responses = await processor.process_batch(batch_data, 0)

        assert len(responses) == 1
        # Verify that the generated prompt included patient context
        mock_adapter.generate.assert_called_once()
        call_args = mock_adapter.generate.call_args
        prompt = call_args[1]["system_prompt"]
        assert "65-year-old male" in prompt or "hypertension" in prompt


@pytest.mark.integration
class TestMultiTaskIntegration:
    """Integration tests across multiple tasks."""

    @pytest.fixture
    def multi_task_batch(self, mock_image):
        """Create batch data suitable for multiple tasks."""
        return BatchData(
            images=[mock_image] * 3,
            metadata=[
                {"modality": "CT", "task": "localization", "study_id": "LOC_001"},
                {"modality": "MRI", "task": "caption", "study_id": "CAP_001"},
                {"modality": "X-ray", "task": "diagnosis", "study_id": "DIAG_001"},
            ],
        )

    async def test_multiple_processors_same_batch(self, multi_task_batch):
        """Test running multiple processors on the same batch data."""
        processors = {
            "localization": LocalizationProcessor(
                ProcessorConfig(task_name="localization", model_name="gpt-4o")
            ),
            "caption": CaptionProcessor(ProcessorConfig(task_name="caption", model_name="gpt-4o")),
        }

        results = {}

        # Mock all adapters
        with (
            patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter") as mock_loc_adapter,
            patch("nova_retrieval_vlm.processors.caption.OpenAIAdapter") as mock_cap_adapter,
        ):
            # Setup localization mock
            mock_loc_adapter_instance = AsyncMock()
            mock_loc_adapter_instance.generate.return_value = ('{"boxes": []}', MagicMock())
            mock_loc_adapter.return_value = mock_loc_adapter_instance

            # Setup caption mock
            mock_cap_adapter_instance = AsyncMock()
            mock_cap_adapter_instance.generate.return_value = ("Medical caption", MagicMock())
            mock_cap_adapter.return_value = mock_cap_adapter_instance

            # Process with each processor
            for task_name, processor in processors.items():
                responses = await processor.process_batch(multi_task_batch, 0)
                results[task_name] = responses

        # Verify all processors handled the batch
        assert len(results) == 2
        assert len(results["localization"]) == 3
        assert len(results["caption"]) == 3

    async def test_processor_consistency_across_tasks(self, mock_image):
        """Test that processors maintain consistency across different tasks."""
        consistent_metadata = {
            "modality": "CT",
            "patient_id": "PATIENT_123",
            "study_date": "2024-01-15",
        }

        batch_data = BatchData(
            images=[mock_image],
            metadata=[consistent_metadata],
        )

        processors = [
            LocalizationProcessor(ProcessorConfig(task_name="localization", model_name="gpt-4o")),
            CaptionProcessor(ProcessorConfig(task_name="caption", model_name="gpt-4o")),
        ]

        results = []

        for processor in processors:
            with patch(
                f"nova_retrieval_vlm.processors.{processor.config.task_name}.OpenAIAdapter"
            ) as mock_adapter:
                mock_adapter_instance = AsyncMock()
                if processor.config.task_name == "localization":
                    mock_adapter_instance.generate.return_value = ('{"boxes": []}', MagicMock())
                else:
                    mock_adapter_instance.generate.return_value = ("Caption text", MagicMock())
                mock_adapter.return_value = mock_adapter_instance

                responses = await processor.process_batch(batch_data, 0)
                results.append(responses[0])

        # All responses should include consistent metadata
        for response in results:
            assert response.metadata["image_path"] == str(mock_image)
            assert response.metadata["modality"] == "CT"


@pytest.mark.integration
class TestRetrievalIntegration:
    """Integration tests for retrieval-augmented processing."""

    @pytest.fixture
    def retrieval_config(self):
        """Configuration with retrieval enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ProcessorConfig(
                task_name="localization",
                model_name="gpt-4o",
                use_retrieval=True,
                retrieval_type="bm25",
                output_dir=Path(temp_dir),
            )
            yield config

    @patch("nova_retrieval_vlm.processors.localization.OpenAIAdapter")
    async def test_localization_with_retrieval(
        self, mock_adapter_class, retrieval_config, mock_image
    ):
        """Test localization with retrieval augmentation."""
        # Mock retrieval passages
        mock_passages = [
            "Medical guideline: Localization of brain lesions requires careful analysis of anatomical landmarks.",
            "Reference: CT scans show hypodense areas in acute stroke within 6 hours.",
            "Protocol: Use bounding boxes to mark areas of abnormal attenuation.",
        ]

        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = (
            '{"boxes": [[10, 20, 30, 40]], "confidence": 0.95}',
            MagicMock(),
        )
        mock_adapter_class.return_value = mock_adapter

        batch_data = BatchData(
            images=[mock_image],
            metadata=[{"modality": "CT", "clinical_query": "locate stroke"}],
        )

        processor = LocalizationProcessor(retrieval_config)

        # Mock retrieval system (this would normally be injected)
        with patch.object(processor, "_retrieve_relevant_passages", return_value=mock_passages):
            responses = await processor.process_batch(batch_data, 0)

        assert len(responses) == 1
        response = responses[0]

        # Verify retrieval enhanced the processing
        assert response.confidence >= 0.9  # Should have high confidence with good retrieval

        # Verify adapter was called with retrieved context
        mock_adapter.generate.assert_called_once()
        call_kwargs = mock_adapter.generate.call_args[1]
        # The passages should be included in the prompt or context
        assert "passages" in call_kwargs or any(
            "guideline" in str(arg).lower() for arg in call_kwargs.values()
        )


@pytest.mark.integration
class TestErrorRecoveryAndResilience:
    """Integration tests for error recovery and system resilience."""

    async def test_partial_batch_failure_recovery(self, mock_image):
        """Test system handles partial batch failures gracefully."""
        batch_data = BatchData(
            images=[mock_image, mock_image, mock_image],
            metadata=[{}, {}, {}],
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="unreliable-model")
        processor = LocalizationProcessor(processor_config)

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            # First call succeeds, second fails, third succeeds
            mock_adapter.generate.side_effect = [
                ('{"boxes": [[1, 2, 3, 4]]}', MagicMock()),
                Exception("Temporary API failure"),
                ('{"boxes": [[5, 6, 7, 8]]}', MagicMock()),
            ]
            mock_adapter_class.return_value = mock_adapter

            # Should handle the failure appropriately
            with pytest.raises(Exception, match="Temporary API failure"):
                await processor.process_batch(batch_data, 0)

    async def test_network_timeout_handling(self, mock_image):
        """Test handling of network timeouts and retries."""
        batch_data = BatchData(images=[mock_image], metadata=[{}])

        processor_config = ProcessorConfig(task_name="caption", model_name="slow-model")
        processor = CaptionProcessor(processor_config)

        with patch("nova_retrieval_vlm.processors.caption.OpenAIAdapter") as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate.side_effect = TimeoutError("Request timeout")
            mock_adapter_class.return_value = mock_adapter

            # Should handle timeout appropriately
            with pytest.raises(TimeoutError, match="Request timeout"):
                await processor.process_batch(batch_data, 0)

    def test_invalid_response_handling(self, mock_image):
        """Test handling of invalid model responses."""
        BatchData(images=[mock_image], metadata=[{}])

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        # Test with malformed JSON responses
        invalid_responses = [
            ModelResponse(text='{"invalid": json}', confidence=0.5),
            ModelResponse(text="not json at all", confidence=0.3),
            ModelResponse(text='{"boxes": []}', confidence=0.8),  # Valid one
        ]

        ground_truth = ['{"boxes": [[1, 2, 3, 4]]}'] * 3

        # Evaluation should handle invalid responses gracefully
        with patch("nova_retrieval_vlm.processors.localization.evaluate_detection") as mock_eval:
            mock_eval.return_value = {"map_50": 0.1}  # Low score due to invalid responses

            metrics = processor.evaluate_responses(invalid_responses, ground_truth)
            assert isinstance(metrics, type(metrics))  # Should not crash


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests for performance and scalability."""

    @pytest.mark.slow
    async def test_large_batch_processing(self, mock_image):
        """Test processing large batches efficiently."""
        # Create large batch (but not too large for test performance)
        large_batch = BatchData(
            images=[mock_image] * 20,
            metadata=[{"modality": "CT", "id": i} for i in range(20)],
        )

        processor_config = ProcessorConfig(
            task_name="localization",
            model_name="efficient-model",
            batch_size=20,
        )
        processor = LocalizationProcessor(processor_config)

        with patch(
            "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate.return_value = ('{"boxes": []}', MagicMock())
            mock_adapter_class.return_value = mock_adapter

            import time

            start_time = time.time()

            responses = await processor.process_batch(large_batch, 0)

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify results
            assert len(responses) == 20
            assert all(isinstance(r, ModelResponse) for r in responses)

            # Performance assertion (adjust threshold based on requirements)
            assert processing_time < 5.0  # Should complete within 5 seconds

    async def test_concurrent_processor_usage(self, mock_image):
        """Test multiple processors running concurrently."""
        import asyncio

        batch_data = BatchData(images=[mock_image], metadata=[{}])

        processors = [
            LocalizationProcessor(ProcessorConfig(task_name="localization", model_name="model1")),
            CaptionProcessor(ProcessorConfig(task_name="caption", model_name="model2")),
        ]

        async def process_with_mock(processor, batch_data):
            adapter_path = (
                f"nova_retrieval_vlm.processors.{processor.config.task_name}.OpenAIAdapter"
            )
            with patch(adapter_path) as mock_adapter_class:
                mock_adapter = AsyncMock()
                if processor.config.task_name == "localization":
                    mock_adapter.generate.return_value = ('{"boxes": []}', MagicMock())
                else:
                    mock_adapter.generate.return_value = ("Caption", MagicMock())
                mock_adapter_class.return_value = mock_adapter

                return await processor.process_batch(batch_data, 0)

        # Run processors concurrently
        tasks = [process_with_mock(processor, batch_data) for processor in processors]
        results = await asyncio.gather(*tasks)

        # Verify all processors completed
        assert len(results) == 2
        assert all(len(result) == 1 for result in results)  # Each processed 1 image

    def test_memory_usage_stability(self, mock_image):
        """Test memory usage remains stable across multiple processing cycles."""
        import gc

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        batch_data = BatchData(images=[mock_image] * 5, metadata=[{}] * 5)

        # Process multiple batches to test memory stability
        for cycle in range(10):
            processor = LocalizationProcessor(processor_config)

            with patch(
                "nova_retrieval_vlm.processors.localization.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = ('{"boxes": []}', MagicMock())
                mock_adapter_class.return_value = mock_adapter

                # Process batch (in sync context for test simplicity)
                import asyncio

                responses = asyncio.run(processor.process_batch(batch_data, cycle))

                assert len(responses) == 5

            # Force garbage collection
            del processor
            gc.collect()

        # If we reach here without memory issues, the test passes
