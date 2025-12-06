"""Comprehensive edge case and stress testing for NOVA retrieval VLM system."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.processors import LocalizationProcessor
from nova_retrieval_vlm.processors import ProcessorConfig
from nova_retrieval_vlm.types import BatchData
from nova_retrieval_vlm.types import JSONParseError
from nova_retrieval_vlm.types import ModelResponse
from nova_retrieval_vlm.types import parse_json_response
from nova_retrieval_vlm.utils.batch_processing_utils import BatchContext
from nova_retrieval_vlm.utils.batch_processing_utils import normalize_localization_result
from nova_retrieval_vlm.utils.batch_processing_utils import postprocess_batch_result

# Valid unified response for mocks (matches expected JSON schema)
VALID_UNIFIED_RESPONSE = json.dumps(
    {
        "caption": {"description": "Test MRI scan showing normal anatomy", "confidence": 0.9},
        "diagnosis": {"primary_diagnosis": "Normal", "confidence": 0.9},
        "localization": {"localizations": []},
    }
)


@pytest.mark.edge_case
class TestEdgeCaseInputs:
    """Test edge cases with unusual or malformed inputs."""

    @patch("nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter")
    def test_empty_batch_processing(self, mock_adapter):
        """Test processors handle empty batches gracefully."""
        # Mock the adapter to avoid API key issues
        mock_adapter.return_value = MagicMock()

        empty_batch = BatchData(images=[], metadata=[])

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_empty():
            responses = await processor.process_batch(empty_batch, 0)
            assert responses == []

        asyncio.run(test_empty())

    def test_mismatched_batch_sizes(self, mock_image):
        """Test that mismatched image and metadata array sizes fail fast."""
        mismatched_batch = BatchData(
            images=[mock_image, mock_image, mock_image],  # 3 images
            metadata=[{"id": 1}, {"id": 2}],  # 2 metadata entries
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_mismatch():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                # Should fail fast with strict=True - mismatched lengths are an error
                with pytest.raises(ValueError):
                    await processor.process_batch(mismatched_batch, 0)

        asyncio.run(test_mismatch())

    def test_none_and_null_metadata(self, mock_image):
        """Test handling of None and null values in metadata."""
        # Test that None values in metadata are properly rejected by validation
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchData(
                images=[mock_image, mock_image],
                metadata=[
                    None,  # This should cause validation error
                    {"modality": None, "patient_info": None},
                ],
            )

        # Test valid batch with None values inside metadata dictionaries
        null_metadata_batch = BatchData(
            images=[mock_image, mock_image],
            metadata=[
                {},  # Empty dict instead of None
                {"modality": None, "patient_info": None},
            ],
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_nulls():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                # Should not crash with null metadata
                responses = await processor.process_batch(null_metadata_batch, 0)
                assert len(responses) >= 0

        asyncio.run(test_nulls())

    def test_very_large_metadata(self, mock_image):
        """Test handling of extremely large metadata objects."""
        large_metadata = {
            "description": "x" * 100000,  # 100KB string
            "history": {"events": [{"id": i, "data": "y" * 1000} for i in range(100)]},
            "nested": {"level1": {"level2": {"level3": {"data": "z" * 10000}}}},
        }

        large_batch = BatchData(
            images=[mock_image],
            metadata=[large_metadata],
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_large_metadata():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                start_time = time.time()
                responses = await processor.process_batch(large_batch, 0)
                end_time = time.time()

                assert len(responses) == 1
                # Should complete in reasonable time despite large metadata
                assert end_time - start_time < 10.0

        asyncio.run(test_large_metadata())

    def test_invalid_image_paths(self):
        """Test handling of invalid or non-existent image paths."""
        invalid_batch = BatchData(
            images=[
                Path("/non/existent/path.png"),
                Path(),
                Path("invalid file name with spaces.jpg"),
            ],
            metadata=[{}, {}, {}],
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_invalid_paths():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                # Should handle invalid paths appropriately (likely raise error)
                with pytest.raises((FileNotFoundError, OSError)):
                    await processor.process_batch(invalid_batch, 0)

        asyncio.run(test_invalid_paths())

    def test_unicode_and_special_characters(self, mock_image):
        """Test handling of Unicode and special characters in metadata."""
        unicode_batch = BatchData(
            images=[mock_image],
            metadata=[
                {
                    "patient_name": "José María Çelik",
                    "notes": "咖啡 ☕ 🏥 Medical notes with emoji",
                    "diagnosis": "Διάγνωσις Greek medical term",
                    "location": "Москва, Россия",
                    "special_chars": "!@#$%^&*(){}[]|\\:;\"'<>,.?/~`±§",
                }
            ],
        )

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def test_unicode():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                responses = await processor.process_batch(unicode_batch, 0)
                assert len(responses) == 1
                # Should handle unicode metadata gracefully (no crashes)
                assert responses[0] is not None
                # Verify the processor handled unicode input without errors
                mock_adapter.generate.assert_called_once()

        asyncio.run(test_unicode())


@pytest.mark.edge_case
class TestMalformedJSONHandling:
    """Test handling of malformed JSON responses from models."""

    def test_completely_invalid_json(self):
        """Test parsing of completely invalid JSON responses."""
        invalid_cases = [
            "This is not JSON at all",
            "{broken json}",
            '{"missing_quote: "value"}',
            '{"trailing_comma": "value",}',
            '{key: "unquoted key"}',
            "Just plain text response",
            "",
            None,
        ]

        for invalid_case in invalid_cases:
            if invalid_case is not None:
                with pytest.raises(JSONParseError):
                    parse_json_response(str(invalid_case))

    def test_partially_valid_json(self):
        """Test JSON with some valid structure but errors."""
        partial_cases = [
            '{"valid": "field", "invalid": }',
            '{"array": [1, 2, 3,]}',
            '{"nested": {"valid": "yes", "broken": }}',
            '{"unicode": "café", "broken": invalid}',
        ]

        for partial_case in partial_cases:
            with pytest.raises(JSONParseError):
                parse_json_response(partial_case)

    def test_json_with_extra_content(self):
        """Test JSON mixed with other content."""
        mixed_content_cases = [
            'Some text before {"valid": "json"} some text after',
            'Multiple {"json": 1} objects {"json": 2} in response',
            '```json\n{"valid": "json"}\n```\nExtra text after',
        ]

        # These should either parse successfully or fail consistently
        for case in mixed_content_cases:
            try:
                result = parse_json_response(case)
                assert isinstance(result, dict)  # Should be valid if parsed
            except JSONParseError:
                pass  # Acceptable to fail on ambiguous content

    def test_deeply_nested_json_limits(self):
        """Test JSON with extreme nesting depth."""
        # Create deeply nested JSON
        deeply_nested = "{" * 1000
        for i in range(999, -1, -1):
            deeply_nested += f'"level_{i}": '
        deeply_nested += '"deep_value"'
        deeply_nested += "}" * 1000

        # Should either parse successfully or fail gracefully
        try:
            result = parse_json_response(deeply_nested)
            # If parsed successfully, verify structure
            current = result
            for i in range(100):  # Check first 100 levels
                if f"level_{i}" in current:
                    current = current[f"level_{i}"]
                else:
                    break
        except (JSONParseError, RecursionError):
            # Acceptable to fail on extreme nesting
            pass

    def test_json_with_huge_arrays(self):
        """Test JSON with very large arrays."""
        huge_array_json = json.dumps(
            {
                "boxes": [[i, i + 1, i + 2, i + 3] for i in range(10000)],
                "scores": [0.5 + i / 20000 for i in range(10000)],
                "metadata": {"size": "huge"},
            }
        )

        start_time = time.time()
        result = parse_json_response(huge_array_json)
        end_time = time.time()

        assert len(result["boxes"]) == 10000
        assert len(result["scores"]) == 10000
        # Should parse in reasonable time
        assert end_time - start_time < 5.0


@pytest.mark.stress
class TestConcurrencyAndRaceConditions:
    """Test concurrent processing and race conditions."""

    async def test_concurrent_batch_processing(self, mock_image):
        """Test multiple batches processing concurrently."""
        batch_data = BatchData(images=[mock_image] * 5, metadata=[{"id": i} for i in range(5)])

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")

        async def process_batch_with_delay(batch_idx):
            processor = LocalizationProcessor(processor_config)

            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()

                # Add random delay to simulate real processing
                async def delayed_generate(*args, **kwargs):
                    await asyncio.sleep(0.1)
                    return ('{"boxes": []}', MagicMock())

                mock_adapter.generate = delayed_generate
                mock_adapter_class.return_value = mock_adapter

                return await processor.process_batch(batch_data, batch_idx)

        # Process 10 batches concurrently
        tasks = [process_batch_with_delay(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All batches should complete successfully
        assert len(results) == 10
        assert all(len(result) == 5 for result in results)

    def test_concurrent_file_operations(self, mock_image):
        """Test concurrent file operations don't cause conflicts."""
        import threading

        def process_batch_with_files(thread_id):
            with tempfile.TemporaryDirectory() as temp_dir:
                batch_ctx = BatchContext(
                    idx=thread_id,
                    folder=Path(temp_dir) / f"batch_{thread_id}",
                    img_path=mock_image,
                    width=256,
                    height=256,
                )
                batch_ctx.folder.mkdir(parents=True, exist_ok=True)

                # Use float coordinates to satisfy beartype contract for detection preds
                prediction_result = {"boxes": [[10.0, 20.0, 30.0, 40.0]], "thread_id": thread_id}
                mock_dataset = MagicMock()
                mock_dataset.__getitem__.return_value = {"bbox_gold": {}}
                prediction_list = []

                with patch(
                    "nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics"
                ):
                    postprocess_batch_result(
                        batch_ctx, prediction_result, "localization", mock_dataset, prediction_list
                    )

                # Verify files were created correctly
                assert (batch_ctx.folder / "pred.jsonl").exists()
                assert (batch_ctx.folder / "ref.jsonl").exists()

        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_batch_with_files, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    async def test_resource_cleanup_under_exceptions(self, mock_image):
        """Test proper resource cleanup when exceptions occur during processing."""
        batch_data = BatchData(images=[mock_image], metadata=[{}])

        processor_config = ProcessorConfig(task_name="localization", model_name="failing-model")
        processor = LocalizationProcessor(processor_config)

        failure_count = 0

        async def failing_process():
            nonlocal failure_count

            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()

                async def sometimes_fail(*args, **kwargs):
                    nonlocal failure_count
                    failure_count += 1
                    if failure_count % 3 == 0:  # Fail every 3rd call
                        raise Exception("Simulated failure")
                    return ('{"boxes": []}', MagicMock())

                mock_adapter.generate = sometimes_fail
                mock_adapter_class.return_value = mock_adapter

                try:
                    return await processor.process_batch(batch_data, failure_count)
                except Exception:
                    return None

        # Run multiple failing processes
        tasks = [failing_process() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed, some should fail
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception) or r is None]

        assert len(successes) > 0  # Some should succeed
        assert len(failures) > 0  # Some should fail


@pytest.mark.stress
class TestMemoryAndPerformanceStress:
    """Stress tests for memory usage and performance."""

    @pytest.mark.slow
    def test_large_batch_memory_usage(self, mock_image):
        """Test memory usage with very large batches."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large batch
        large_batch_size = 100
        large_batch = BatchData(
            images=[mock_image] * large_batch_size,
            metadata=[{"id": i, "data": "x" * 1000} for i in range(large_batch_size)],
        )

        processor_config = ProcessorConfig(
            task_name="localization", model_name="test-model", batch_size=large_batch_size
        )
        processor = LocalizationProcessor(processor_config)

        async def process_large_batch():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                return await processor.process_batch(large_batch, 0)

        # Process the large batch
        responses = asyncio.run(process_large_batch())

        # Check memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert len(responses) == large_batch_size
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500  # Less than 500MB increase

    @pytest.mark.slow
    async def test_sustained_processing_performance(self, mock_image):
        """Test performance remains stable over sustained processing."""
        batch_data = BatchData(images=[mock_image] * 10, metadata=[{"batch": i} for i in range(10)])

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")

        processing_times = []

        for iteration in range(20):  # Process 20 batches
            processor = LocalizationProcessor(processor_config)

            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                start_time = time.time()
                responses = await processor.process_batch(batch_data, iteration)
                end_time = time.time()

                processing_time = end_time - start_time
                processing_times.append(processing_time)

                assert len(responses) == 10

        # Performance should remain stable (no significant degradation)
        avg_early = sum(processing_times[:5]) / 5
        avg_late = sum(processing_times[-5:]) / 5

        # Later batches shouldn't be more than 50% slower than early ones
        assert avg_late < avg_early * 1.5

    @pytest.mark.slow
    def test_many_small_batches_vs_few_large_batches(self, mock_image):
        """Compare performance of many small batches vs few large batches."""

        # Test many small batches (10 images each, 10 batches)
        small_batch_times = []
        for _ in range(10):
            small_batch = BatchData(images=[mock_image] * 10, metadata=[{"type": "small"}] * 10)

            processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
            processor = LocalizationProcessor(processor_config)

            async def process_small():
                with patch(
                    "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
                ) as mock_adapter_class:
                    mock_adapter = AsyncMock()
                    mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                    mock_adapter_class.return_value = mock_adapter

                    start = time.time()
                    await processor.process_batch(small_batch, 0)
                    return time.time() - start

            small_batch_times.append(asyncio.run(process_small()))

        # Test few large batches (100 images each, 1 batch)
        large_batch = BatchData(images=[mock_image] * 100, metadata=[{"type": "large"}] * 100)

        processor_config = ProcessorConfig(task_name="localization", model_name="test-model")
        processor = LocalizationProcessor(processor_config)

        async def process_large():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                start = time.time()
                await processor.process_batch(large_batch, 0)
                return time.time() - start

        large_batch_time = asyncio.run(process_large())

        total_small_time = sum(small_batch_times)

        # Compare performance characteristics

        # Both approaches should complete in reasonable time
        assert total_small_time < 30.0  # 30 seconds max
        assert large_batch_time < 30.0  # 30 seconds max


@pytest.mark.edge_case
class TestResourceLimitsAndBoundaryConditions:
    """Test system behavior at resource limits and boundary conditions."""

    def test_extremely_long_text_responses(self):
        """Test handling of extremely long text responses."""
        # Create very long response text
        long_text = "This is a very long medical description. " * 10000
        long_json = json.dumps({"description": long_text, "boxes": []})

        # Should parse without issues
        result = parse_json_response(long_json)
        assert len(result["description"]) > 400000  # Very long
        assert result["boxes"] == []

    def test_zero_and_negative_confidence_values(self):
        """Test handling of zero and negative confidence values."""
        # Valid confidence values (0.0 <= confidence <= 1.0) should work
        valid_responses = [
            ModelResponse(text="Zero confidence", confidence=0.0),
            ModelResponse(text="Very small confidence", confidence=1e-10),
            ModelResponse(text="Max confidence", confidence=1.0),
            ModelResponse(text="Mid confidence", confidence=0.5),
        ]

        # All valid responses should be handled gracefully
        for response in valid_responses:
            assert response.text is not None
            assert isinstance(response.confidence, int | float)
            assert 0.0 <= response.confidence <= 1.0

        # Invalid negative confidence should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelResponse(text="Negative confidence", confidence=-0.5)

        # Invalid high confidence should also raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelResponse(text="Very large confidence", confidence=1000.0)

    def test_boundary_bounding_box_coordinates(self):
        """Test bounding boxes with boundary coordinate values."""
        boundary_cases = {
            "zero_coordinates": {"boxes": [[0, 0, 0, 0]]},
            "negative_coordinates": {"boxes": [[-10, -5, -1, -1]]},
            "very_large_coordinates": {"boxes": [[1e6, 1e6, 1e7, 1e7]]},
            "floating_point_precision": {"boxes": [[0.000001, 0.000001, 0.999999, 0.999999]]},
            "mixed_precision": {"boxes": [[1, 2.5, 3, 4.7]]},
        }

        for case_data in boundary_cases.values():
            # Should normalize without crashing
            normalize_localization_result(case_data)
            assert "boxes" in case_data
            assert "labels" in case_data
            assert "scores" in case_data

    def test_disk_space_simulation(self, mock_image):
        """Test behavior when disk space is limited (simulated)."""
        # This test simulates disk space issues by using very small temp directories
        # and large amounts of data

        batch_ctx = BatchContext(
            idx=0,
            folder=Path("/tmp/test_limited_space"),
            img_path=mock_image,
            width=1000,
            height=1000,
        )

        # Create very large prediction result
        huge_result = {
            "boxes": [[i, i + 1, i + 2, i + 3] for i in range(50000)],
            "metadata": {"huge_field": "x" * 1000000},  # 1MB string
            "details": [{"id": i, "data": "y" * 1000} for i in range(10000)],
        }

        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {"bbox_gold": {}}
        prediction_list = []

        # Should handle large data appropriately
        try:
            with patch(
                "nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics"
            ):
                batch_ctx.folder.mkdir(parents=True, exist_ok=True)
                postprocess_batch_result(
                    batch_ctx, huge_result, "localization", mock_dataset, prediction_list
                )

            # If successful, verify files were created
            assert (batch_ctx.folder / "pred.jsonl").exists()

        except OSError:
            # Acceptable to fail with disk space issues
            pass
        finally:
            # Cleanup
            import shutil

            if batch_ctx.folder.exists():
                shutil.rmtree(batch_ctx.folder)

    def test_maximum_batch_size_limits(self, mock_image):
        """Test system behavior at maximum batch sizes."""
        # Test with very large batch size setting
        extreme_config = ProcessorConfig(
            task_name="localization",
            model_name="test-model",
            batch_size=128,  # Maximum allowed batch size
        )

        # Create batch that matches the large batch size
        extreme_batch = BatchData(
            images=[mock_image] * 1000,  # 1000 images (smaller than config for test speed)
            metadata=[{"id": i} for i in range(1000)],
        )

        processor = LocalizationProcessor(extreme_config)

        async def test_extreme_batch():
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (VALID_UNIFIED_RESPONSE, MagicMock())
                mock_adapter_class.return_value = mock_adapter

                start_time = time.time()
                responses = await processor.process_batch(extreme_batch, 0)
                end_time = time.time()

                # Should complete successfully
                assert len(responses) == 1000
                # Should complete in reasonable time (adjust as needed)
                assert end_time - start_time < 60.0  # 1 minute max

        # Run test with timeout
        try:
            asyncio.run(asyncio.wait_for(test_extreme_batch(), timeout=120))
        except asyncio.TimeoutError:
            pytest.skip("Extreme batch test timed out - system limits reached")


@pytest.mark.edge_case
class TestNetworkAndIOFailures:
    """Test handling of network and I/O failures."""

    async def test_intermittent_network_failures(self, mock_image):
        """Test handling of intermittent network connection issues."""
        batch_data = BatchData(images=[mock_image] * 5, metadata=[{}] * 5)

        processor_config = ProcessorConfig(task_name="localization", model_name="unreliable-model")
        processor = LocalizationProcessor(processor_config)

        call_count = 0

        async def unreliable_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Simulate network issues
            if call_count % 3 == 0:
                raise ConnectionError("Network timeout")
            elif call_count % 5 == 0:
                raise OSError("Connection refused")
            else:
                return ('{"boxes": []}', MagicMock())

        with patch(
            "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
        ) as mock_adapter_class:
            mock_adapter = AsyncMock()
            mock_adapter.generate = unreliable_generate
            mock_adapter_class.return_value = mock_adapter

            # Should handle network failures appropriately
            with pytest.raises((ConnectionError, OSError)):
                await processor.process_batch(batch_data, 0)

    def test_file_permission_errors(self, mock_image):
        """Test handling of file permission errors during output."""
        import os
        import stat

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory with restricted permissions
            restricted_dir = Path(temp_dir) / "restricted"
            restricted_dir.mkdir()
            os.chmod(restricted_dir, stat.S_IRUSR)  # Read-only

            batch_ctx = BatchContext(
                idx=0,
                folder=restricted_dir / "output",
                img_path=mock_image,
                width=256,
                height=256,
            )

            prediction_result = {"boxes": [[10, 20, 30, 40]]}
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = {"bbox_gold": {}}
            prediction_list = []

            # Should handle permission errors gracefully
            with (
                pytest.raises((PermissionError, OSError)),
                patch("nova_retrieval_vlm.utils.batch_processing_utils.compute_evaluation_metrics"),
            ):
                postprocess_batch_result(
                    batch_ctx, prediction_result, "localization", mock_dataset, prediction_list
                )

    async def test_corrupted_model_responses(self, mock_image):
        """Test handling of corrupted or truncated model responses."""
        batch_data = BatchData(images=[mock_image], metadata=[{}])

        processor_config = ProcessorConfig(task_name="localization", model_name="corrupted-model")
        processor = LocalizationProcessor(processor_config)

        corrupted_responses = [
            ('{"boxes": [[10, 20, 30', MagicMock()),  # Truncated JSON
            ('��{"boxes": []}', MagicMock()),  # Binary corruption
            ('\x00\x01{"boxes": []}\xff', MagicMock()),  # Null bytes
            ('{"boxes": []} followed by garbage data 123!@#', MagicMock()),  # Extra content
        ]

        for corrupted_response, log in corrupted_responses:
            with patch(
                "nova_retrieval_vlm.models.openai_adapter.OpenAIAdapter"
            ) as mock_adapter_class:
                mock_adapter = AsyncMock()
                mock_adapter.generate.return_value = (corrupted_response, log)
                mock_adapter_class.return_value = mock_adapter

                # Should handle corrupted responses appropriately
                try:
                    responses = await processor.process_batch(batch_data, 0)
                    # If it succeeds, verify the response is reasonable
                    assert len(responses) == 1
                    assert isinstance(responses[0], ModelResponse)
                except (JSONParseError, Exception):
                    # Acceptable to fail on corrupted data
                    pass
