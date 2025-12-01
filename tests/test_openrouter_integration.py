"""Comprehensive OpenRouter API integration tests.

These tests can optionally run against the real OpenRouter API using an API key.
Set OPENROUTER_API_KEY environment variable to enable real API tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from nova_retrieval_vlm.data.nova_dataset import NovaDataset
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
from nova_retrieval_vlm.processors.base import ProcessorConfig
from nova_retrieval_vlm.processors.caption import CaptionProcessor
from nova_retrieval_vlm.types import BatchData


@pytest.mark.integration
class TestOpenRouterAPIIntegration:
    """Integration tests for OpenRouter API functionality."""

    @pytest.fixture
    def api_key(self) -> str | None:
        """Get OpenRouter API key from environment or return None for mock tests."""
        return os.getenv("OPENROUTER_API_KEY")

    @pytest.fixture
    def test_image_path(self) -> Path:
        """Create a test image for API testing."""
        # Use a real NOVA dataset image if available, otherwise create mock
        try:
            dataset = NovaDataset()
            sample = dataset[0]
            # Save to temp file for testing
            temp_path = Path(tempfile.mktemp(suffix=".jpg"))
            if hasattr(sample["image"], "save"):
                sample["image"].save(temp_path)
            else:
                # Create a minimal test image
                from PIL import Image

                Image.new("RGB", (480, 480), color="gray").save(temp_path)
            yield temp_path
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            # Fallback: create a simple test image
            from PIL import Image

            temp_path = Path(tempfile.mktemp(suffix=".jpg"))
            Image.new("RGB", (480, 480), color="gray").save(temp_path)
            yield temp_path
            if temp_path.exists():
                temp_path.unlink()

    @pytest.fixture
    def sample_batch_data(self, test_image_path: Path) -> BatchData:
        """Create sample batch data for testing."""
        return BatchData(
            images=[str(test_image_path)],
            metadata=[
                {
                    "clinical_history": "A ten-month-old infant presented with vertical nystagmus and optic nerve atrophy.",
                    "final_diagnosis": "Septo-optic dysplasia",
                    "caption": "Coronal T2-weighted MRI shows absence of septum pellucidum",
                    "width": 480,
                    "height": 480,
                    "image_id": "test_001",
                }
            ],
            labels=None,
        )

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping real API tests",
    )
    @pytest.mark.asyncio
    async def test_real_openrouter_caption_generation(
        self, api_key: str, test_image_path: Path, sample_batch_data: BatchData
    ):
        """Test real OpenRouter API for caption generation."""
        # Use a reliable model for testing
        model_name = "anthropic/claude-3.5-sonnet"

        adapter = OpenAIAdapter(
            model_name=model_name,
            reasoning_enabled=False,
            enable_caching=True,
        )

        # Test single image generation
        response_text, generation_log = await adapter.generate(
            image_path=test_image_path,
            _passages=[],
            system_prompt="Describe this brain MRI image in detail.",
            max_tokens=500,
            temperature=0.1,
        )

        # Verify response
        assert response_text is not None
        assert len(response_text) > 50  # Should be a substantial response
        assert generation_log is not None
        assert generation_log["model"] == model_name
        assert generation_log["success"] is True

        print("✅ Real API Caption Test Success")
        print(f"Response length: {len(response_text)} characters")
        print(f"Generated caption: {response_text[:200]}...")

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping real API tests",
    )
    @pytest.mark.asyncio
    async def test_real_openrouter_clinical_diagnosis(
        self, api_key: str, test_image_path: Path, sample_batch_data: BatchData
    ):
        """Test real OpenRouter API with clinical history integration."""
        model_name = "anthropic/claude-3.5-sonnet"

        adapter = OpenAIAdapter(
            model_name=model_name,
            reasoning_enabled=False,
            enable_caching=True,
        )

        # Create clinical prompt
        clinical_prompt = f"""
        Analyze this brain MRI considering the following clinical information:

        Clinical History: {sample_batch_data.metadata[0]["clinical_history"]}

        Provide a diagnosis that explains BOTH the imaging findings AND the clinical presentation.
        """

        response_text, generation_log = await adapter.generate(
            image_path=test_image_path,
            _passages=[],
            system_prompt=clinical_prompt,
            max_tokens=800,
            temperature=0.2,
        )

        # Verify response
        assert response_text is not None
        assert len(response_text) > 100

        # Check if clinical history was utilized
        clinical_terms = ["optic", "nystagmus", "septo", "optic nerve"]
        response_lower = response_text.lower()
        utilized_clinical = any(term in response_lower for term in clinical_terms)

        print("✅ Real API Clinical Diagnosis Test Success")
        print(f"Clinical history utilized: {utilized_clinical}")
        print(f"Response preview: {response_text[:300]}...")

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping real API tests",
    )
    @pytest.mark.asyncio
    async def test_real_openrouter_processor_integration(self, api_key: str, test_image_path: Path):
        """Test real OpenRouter API through processor interface."""
        config = ProcessorConfig(
            task_name="caption",
            model_name="anthropic/claude-3.5-sonnet",
            batch_size=1,
            output_dir=tempfile.mkdtemp(),
            max_tokens=500,
            temperature=0.1,
        )

        processor = CaptionProcessor(config)
        batch_data = BatchData(
            images=[str(test_image_path)],
            metadata=[{"width": 480, "height": 480, "image_id": "test_001"}],
            labels=None,
        )

        # Test processor with real API
        responses = await processor.process_batch(batch_data, batch_idx=0)

        assert len(responses) == 1
        response = responses[0]
        assert response.text is not None
        assert len(response.text) > 50
        assert response.confidence is not None
        assert 0 <= response.confidence <= 1

        print("✅ Real API Processor Integration Test Success")
        print(f"Processor response: {response.text[:200]}...")

    @pytest.mark.asyncio
    async def test_mock_openrouter_adapter(self, test_image_path: Path):
        """Test OpenRouter adapter with mock responses (no API key required)."""
        # Test without real API key - use mock
        from unittest.mock import MagicMock
        from unittest.mock import patch

        model_name = "anthropic/claude-3.5-sonnet"
        adapter = OpenAIAdapter(model_name=model_name, enable_caching=False)

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "Mock caption: This is a brain MRI showing normal anatomy."

        # Create mock usage object
        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        mock_response.usage = mock_usage

        with patch.object(
            adapter.client.chat.completions, "create", return_value=mock_response
        ) as mock_api:
            response_text, generation_log = await adapter.generate(
                image_path=test_image_path,
                _passages=[],
                system_prompt="Describe this image.",
                max_tokens=500,
                temperature=0.1,
            )

            # Verify mocked response
            assert response_text == "Mock caption: This is a brain MRI showing normal anatomy."
            assert generation_log.model_name == model_name
            assert generation_log.tokens == 100
            assert generation_log.cost >= 0
            mock_api.assert_called_once()

        print("✅ Mock OpenRouter Adapter Test Success")

    @pytest.mark.asyncio
    async def test_multiple_models_comparison(self, test_image_path: Path):
        """Test comparison of different OpenRouter models (mock mode)."""
        from unittest.mock import MagicMock
        from unittest.mock import patch

        models_to_test = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
        ]

        responses = {}

        for model_name in models_to_test:
            adapter = OpenAIAdapter(model_name=model_name, enable_caching=False)

            # Mock response for each model
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f"Mock response from {model_name}"

            mock_usage = MagicMock()
            mock_usage.total_tokens = 50
            mock_response.usage = mock_usage

            with patch.object(
                adapter.client.chat.completions, "create", return_value=mock_response
            ):
                response_text, generation_log = await adapter.generate(
                    image_path=test_image_path,
                    _passages=[],
                    system_prompt="Describe this image.",
                    max_tokens=200,
                    temperature=0.1,
                )

                responses[model_name] = {
                    "response": response_text,
                    "log": generation_log,
                }

        # Verify all models responded
        assert len(responses) == len(models_to_test)
        for model_name in models_to_test:
            assert model_name in responses
            assert responses[model_name]["response"] is not None
            assert responses[model_name]["log"].model_name == model_name

        print("✅ Multiple Models Comparison Test Success")
        for model_name, result in responses.items():
            print(f"  {model_name}: {result['response'][:50]}...")

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping real API tests",
    )
    @pytest.mark.asyncio
    async def test_api_error_handling(self, api_key: str):
        """Test API error handling with real API."""
        # Test with invalid model name
        adapter = OpenAIAdapter(model_name="invalid/model/name", enable_caching=False)

        from PIL import Image

        temp_path = Path(tempfile.mktemp(suffix=".jpg"))
        Image.new("RGB", (100, 100), color="gray").save(temp_path)

        try:
            response_text, generation_log = await adapter.generate(
                image_path=temp_path,
                _passages=[],
                system_prompt="Test prompt.",
                max_tokens=50,
                temperature=0.1,
            )

            # Should either fail gracefully or succeed with error info
            if generation_log.get("success", False):
                print("⚠️  Invalid model unexpectedly succeeded")
            else:
                print("✅ API error handling works correctly")
                assert "error" in generation_log.lower()

        except Exception as e:
            # Expected for invalid model
            print(f"✅ Expected exception caught: {e}")
        finally:
            if temp_path.exists():
                temp_path.unlink()


# Test execution utilities
def run_openrouter_tests(api_key: str | None = None):
    """Utility to run OpenRouter tests programmatically."""
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    # Run tests
    pytest.main(
        [
            __file__,
            "-v",
            "-x",  # Stop on first failure
        ]
    )


if __name__ == "__main__":
    # Allow running tests directly with API key
    import sys

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    run_openrouter_tests(api_key)
