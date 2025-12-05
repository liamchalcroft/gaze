import json
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nova_retrieval_vlm.models.base import GenerationLog
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    @pytest.fixture
    def mock_openai_client(self):
        with patch("nova_retrieval_vlm.models.openai_adapter.OpenAI") as mock_openai:
            # Configure the mock
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps(
                                {
                                    "boxes": [[10, 20, 30, 40]],
                                    "labels": ["anomaly"],
                                    "scores": [0.95],
                                    "caption": "Test caption",
                                    "diagnosis": "Test diagnosis",
                                }
                            )
                        )
                    )
                ],
                usage=MagicMock(total_tokens=100),
            )
            mock_openai.return_value = mock_client
            yield mock_openai

    @pytest.fixture
    def adapter(self, mock_openai_client):
        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test-api-key"
        adapter = OpenAIAdapter(model_name="test-model")
        yield adapter
        # Clean up
        del os.environ["OPENAI_API_KEY"]

    def test_init(self, mock_openai_client):
        """Test initialization of the adapter."""
        os.environ["OPENAI_API_KEY"] = "test-api-key"
        adapter = OpenAIAdapter(model_name="test-model")

        assert adapter.model_name == "test-model"
        assert adapter.api_key == "test-api-key"
        assert adapter.max_retries == 3  # Default value
        assert adapter.timeout == 60  # Default value

        # Test custom values
        adapter = OpenAIAdapter(
            model_name="custom-model", api_key="custom-api-key", max_retries=5, timeout=30
        )

        assert adapter.model_name == "custom-model"
        assert adapter.api_key == "custom-api-key"
        assert adapter.max_retries == 5
        assert adapter.timeout == 30

        # Clean up
        del os.environ["OPENAI_API_KEY"]

    def test_init_no_api_key(self):
        """Test initialization without API key raises an error."""
        # Ensure environment variable is not set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        with pytest.raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY not set"):
            OpenAIAdapter(model_name="test-model")

    @pytest.mark.asyncio
    async def test_generate(self, adapter, mock_image):
        """Test the generate method."""
        passages = ["Test passage 1", "Test passage 2"]
        system_prompt = "You are a test assistant."

        result, log = await adapter.generate(mock_image, passages, system_prompt)

        # Verify the result
        result_json = json.loads(result)
        assert "boxes" in result_json
        assert "labels" in result_json
        assert "scores" in result_json
        assert "caption" in result_json
        assert "diagnosis" in result_json

        # Verify the log
        assert isinstance(log, GenerationLog)
        assert log.tokens == 100
        # Cost estimation removed - now always 0 (use OpenRouter dashboard for costs)
        assert log.cost == 0.0

        # Verify the call to the OpenAI client
        adapter.client.chat.completions.create.assert_called_once()
        call_args = adapter.client.chat.completions.create.call_args[1]
        assert call_args["model"] == "test-model"
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == system_prompt
        assert call_args["messages"][1]["role"] == "user"
        assert isinstance(call_args["messages"][1]["content"], list)

    @pytest.mark.asyncio
    async def test_generate_text(self, adapter):
        """Test the generate_text method."""
        prompt_text = "Generate a test response."
        system_prompt = "You are a test assistant."

        result, log = await adapter.generate_text(prompt_text, system_prompt)

        # Verify the result is a string
        assert isinstance(result, str)

        # Verify the log
        assert isinstance(log, GenerationLog)
        assert log.tokens == 100
        # Cost estimation removed - now always 0 (use OpenRouter dashboard for costs)
        assert log.cost == 0.0

        # Verify the call to the OpenAI client
        adapter.client.chat.completions.create.assert_called_once()
        call_args = adapter.client.chat.completions.create.call_args[1]
        assert call_args["model"] == "test-model"
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == system_prompt
        assert call_args["messages"][1]["role"] == "user"
        assert call_args["messages"][1]["content"] == prompt_text
