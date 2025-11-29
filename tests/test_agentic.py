"""Tests for the agentic processing module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from PIL import Image

from nova_retrieval_vlm.agentic.tools import ToolRegistry
from nova_retrieval_vlm.agentic.tools import ToolResult
from nova_retrieval_vlm.agentic.tools import VisualTool


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="gray")
            img.save(f.name)
            yield Path(f.name)
            # Cleanup
            Path(f.name).unlink(missing_ok=True)

    def test_registry_initialization(self):
        """Test that registry initializes with default tools."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas()

        # Should have default tools registered
        assert len(schemas) >= 5
        tool_names = {s["function"]["name"] for s in schemas}
        assert "zoom" in tool_names
        assert "crop" in tool_names
        assert "adjust_contrast" in tool_names
        assert "threshold" in tool_names
        assert "reset" in tool_names

    def test_set_image(self, temp_image):
        """Test setting source image."""
        registry = ToolRegistry()
        registry.set_image(temp_image)

        assert registry.current_image is not None
        assert registry.current_image.size == (100, 100)

    def test_execute_zoom(self, temp_image):
        """Test zoom tool execution."""
        registry = ToolRegistry(temp_image)
        result = registry.execute("zoom", factor=2.0)

        assert result.success
        assert result.tool_name == "zoom"
        assert "2.0" in result.description
        assert result.image_base64 is not None
        assert registry.current_image.size == (200, 200)

    def test_execute_crop(self, temp_image):
        """Test crop tool execution."""
        registry = ToolRegistry(temp_image)
        result = registry.execute("crop", box=[0.25, 0.25, 0.75, 0.75])

        assert result.success
        assert result.tool_name == "crop"
        assert result.image_base64 is not None
        # Cropped to 50% of each dimension
        assert registry.current_image.size[0] <= 100
        assert registry.current_image.size[1] <= 100

    def test_execute_contrast(self, temp_image):
        """Test contrast adjustment tool."""
        registry = ToolRegistry(temp_image)
        result = registry.execute("adjust_contrast", factor=1.5)

        assert result.success
        assert result.tool_name == "adjust_contrast"
        assert "1.5" in result.description

    def test_execute_threshold(self, temp_image):
        """Test intensity threshold tool."""
        registry = ToolRegistry(temp_image)
        result = registry.execute("threshold", lower=50, upper=200)

        assert result.success
        assert result.tool_name == "threshold"
        assert "[50, 200]" in result.description

    def test_execute_reset(self, temp_image):
        """Test reset tool."""
        registry = ToolRegistry(temp_image)

        # Modify image first
        registry.execute("zoom", factor=2.0)
        assert registry.current_image.size == (200, 200)

        # Reset
        result = registry.execute("reset")
        assert result.success
        assert registry.current_image.size == (100, 100)

    def test_execute_unknown_tool(self, temp_image):
        """Test executing unknown tool returns error."""
        registry = ToolRegistry(temp_image)
        result = registry.execute("unknown_tool")

        assert not result.success
        assert "unknown_tool" in result.error

    def test_execute_without_image(self):
        """Test executing tool without image returns error."""
        registry = ToolRegistry()
        result = registry.execute("zoom", factor=2.0)

        assert not result.success
        assert "No image" in result.description or "No image" in result.error

    def test_tool_history(self, temp_image):
        """Test that tool history is tracked."""
        registry = ToolRegistry(temp_image)

        registry.execute("zoom", factor=1.5)
        registry.execute("adjust_contrast", factor=1.2)

        history = registry.history
        assert len(history) == 2
        assert history[0].tool_name == "zoom"
        assert history[1].tool_name == "adjust_contrast"

    def test_get_tool_schemas_format(self):
        """Test that tool schemas are in OpenAI format."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas()

        for schema in schemas:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_register_custom_tool(self):
        """Test registering a custom tool."""
        registry = ToolRegistry()

        def custom_execute(**kwargs):
            return ToolResult(
                success=True,
                tool_name="custom",
                description="Custom tool executed",
            )

        custom_tool = VisualTool(
            name="custom",
            description="A custom tool",
            parameters={"value": {"type": "number"}},
            execute=custom_execute,
        )

        registry.register(custom_tool)
        schemas = registry.get_tool_schemas()
        tool_names = {s["function"]["name"] for s in schemas}
        assert "custom" in tool_names


class TestAgenticConfig:
    """Tests for agentic configuration."""

    def test_agentic_config_defaults(self):
        """Test default agentic configuration values."""
        from nova_retrieval_vlm.config import AgenticConfig

        config = AgenticConfig()

        assert config.enabled is False
        assert config.use_tools is True
        assert config.max_turns == 10
        assert config.confidence_threshold == 0.7
        assert config.reasoning_enabled is False

    def test_agentic_config_in_main_config(self):
        """Test agentic config is part of main config."""
        from nova_retrieval_vlm.config import Config

        config = Config()
        assert hasattr(config, "agentic")
        assert config.agentic.enabled is False

    def test_agentic_config_validation(self):
        """Test agentic config validation."""
        from nova_retrieval_vlm.config import AgenticConfig

        # Valid config
        config = AgenticConfig(max_turns=5, confidence_threshold=0.8)
        assert config.max_turns == 5

        # Invalid max_turns
        with pytest.raises(ValueError):
            AgenticConfig(max_turns=0)

        # Invalid confidence threshold
        with pytest.raises(ValueError):
            AgenticConfig(confidence_threshold=1.5)


class TestAgenticProcessor:
    """Tests for the AgenticProcessor class."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        from nova_retrieval_vlm.agentic import AgenticProcessor

        processor = AgenticProcessor(
            model_name="openai/gpt-4o",
            use_tools=True,
            max_turns=3,
        )

        assert processor.model_name == "openai/gpt-4o"
        assert processor.use_tools is True
        assert processor.max_turns == 3

    @pytest.mark.asyncio
    async def test_processor_analyze_mocked(self, temp_image):
        """Test processor analyze with mocked model."""
        from nova_retrieval_vlm.agentic import AgenticProcessor

        # Mock the OpenAIAdapter at class level before lazy initialization
        mock_response = '{"boxes": [[10, 10, 50, 50]], "labels": ["lesion"], "reasoning": "test"}'
        mock_adapter = MagicMock()
        # generate_chat returns (text, tool_calls, log)
        mock_log = MagicMock()
        mock_log.tokens = 100
        mock_adapter.generate_chat = AsyncMock(return_value=(mock_response, None, mock_log))

        with patch(
            "nova_retrieval_vlm.agentic.processor.OpenAIAdapter",
            return_value=mock_adapter,
        ):
            processor = AgenticProcessor(
                model_name="openai/gpt-4o",
                use_tools=False,
            )

            result = await processor.analyze(
                image_path=temp_image,
                task="localization",
                metadata={"modality": "MRI"},
            )

        assert result.final_response is not None
        assert len(result.turns) >= 1
        assert result.total_tokens >= 0

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="gray")
            img.save(f.name)
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)


# RetrievalManager tests removed - replaced with search_web tool


class TestAgenticLocalizationProcessor:
    """Tests for the AgenticLocalizationProcessor class."""

    @pytest.fixture
    def processor_config(self, tmp_path):
        """Create a processor config for testing."""
        from nova_retrieval_vlm.processors import ProcessorConfig

        return ProcessorConfig(
            task_name="localization",
            model_name="openai/gpt-4o",
            batch_size=1,
            use_retrieval=False,
            retrieval_type="bm25",
            output_dir=tmp_path,
            skip_existing=False,
        )

    def test_processor_initialization(self, processor_config):
        """Test agentic localization processor initialization."""
        from nova_retrieval_vlm.agentic import AgenticLocalizationProcessor

        processor = AgenticLocalizationProcessor(
            config=processor_config,
            use_tools=True,
            max_turns=3,
        )

        assert processor.use_tools is True
        assert processor.max_turns == 3

    def test_processor_has_evaluate_responses(self, processor_config):
        """Test that processor has evaluate_responses method."""
        from nova_retrieval_vlm.agentic import AgenticLocalizationProcessor

        processor = AgenticLocalizationProcessor(config=processor_config)

        assert hasattr(processor, "evaluate_responses")
        assert callable(processor.evaluate_responses)


class TestCLIAgenticIntegration:
    """Tests for CLI integration with agentic processors."""

    def test_create_processor_standard(self):
        """Test creating standard processor."""
        from nova_retrieval_vlm.cli import create_processor
        from nova_retrieval_vlm.config import Config
        from nova_retrieval_vlm.processors import LocalizationProcessor

        config = Config(agentic={"enabled": False})
        processor = create_processor(config)

        assert isinstance(processor, LocalizationProcessor)

    def test_create_processor_agentic(self):
        """Test creating agentic processor when enabled."""
        from nova_retrieval_vlm.agentic import AgenticLocalizationProcessor
        from nova_retrieval_vlm.cli import create_processor
        from nova_retrieval_vlm.config import Config

        config = Config(
            agentic={
                "enabled": True,
                "use_visual_reasoning": True,
                "use_tools": True,
                "max_turns": 3,
            }
        )
        processor = create_processor(config)

        assert isinstance(processor, AgenticLocalizationProcessor)


class TestAgenticWebSearchIntegration:
    """Tests for web search integration in agentic tools."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="white")
            img.save(f.name)
            yield Path(f.name)
            # Cleanup
            Path(f.name).unlink(missing_ok=True)

    def test_tool_registry_includes_search_web(self):
        """Test that search_web tool is included in tool registry."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas()

        search_web_schema = next(
            (schema for schema in schemas if schema["function"]["name"] == "search_web"), None
        )

        assert search_web_schema is not None, "search_web tool not found in registry"

        # Verify tool schema structure
        function_def = search_web_schema["function"]
        assert function_def["name"] == "search_web"
        assert "query" in function_def["parameters"]["required"]
        assert function_def["parameters"]["properties"]["query"]["type"] == "string"

    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature_sync")
    def test_search_web_tool_execution(self, mock_search, temp_image):
        """Test search_web tool execution with mocking."""
        # Mock successful search result
        from nova_retrieval_vlm.retrieval.web_search import SearchResult

        mock_search.return_value = [
            SearchResult(
                title="Test Medical Paper",
                url="https://pubmed.ncbi.nlm.nih.gov/12345",
                content="Test medical content with reliability scoring",
                snippet="Test medical content snippet",
                source="pubmed",
                reliability_score=0.9,
                medical_relevance=0.85,
                extracted_entities=["glioblastoma", "MRI", "diagnosis"],
            )
        ]

        registry = ToolRegistry()
        registry.set_image(temp_image)

        # Test search_web tool execution
        result = registry.execute(
            "search_web", query="glioblastoma MRI findings", search_type="pubmed"
        )

        assert result.success is True
        assert "Found 1" in result.description
        assert result.metadata["results_count"] == 1
        assert result.metadata["query"] == "glioblastoma MRI findings"
        assert result.metadata["search_type"] == "pubmed"

        # Verify the search was called with correct parameters
        mock_search.assert_called_once_with(
            query="glioblastoma MRI findings", max_results=5, search_type="pubmed"
        )

    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature_sync")
    def test_search_web_error_handling(self, mock_search, temp_image):
        """Test search_web tool error handling."""
        mock_search.side_effect = Exception("Search failed")

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = registry.execute("search_web", query="test query")

        assert result.success is False
        assert "search failed" in result.description.lower()
        assert "error" in result.metadata

    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature_sync")
    def test_search_web_empty_results(self, mock_search, temp_image):
        """Test search_web tool with empty results."""
        mock_search.return_value = []

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = registry.execute("search_web", query="obscure condition")

        assert result.success is False  # Implementation returns False for no results
        assert (
            "no results" in result.description.lower()
            or "no pubmed results" in result.description.lower()
        )
        assert result.metadata["results_count"] == 0

    @patch("nova_retrieval_vlm.retrieval.web_search.PubMedSearchEngine")
    def test_search_result_creation(self, mock_engine):
        """Test SearchResult dataclass creation and validation."""
        from nova_retrieval_vlm.retrieval.web_search import SearchResult

        result = SearchResult(
            title="Test Paper",
            url="https://example.com/paper",
            content="This is a test medical paper content.",
            snippet="Test snippet",
            source="pubmed",
            reliability_score=0.85,
            medical_relevance=0.9,
            extracted_entities=["glioblastoma", "MRI", "diagnosis"],
        )

        assert result.title == "Test Paper"
        assert result.url == "https://example.com/paper"
        assert result.snippet == "Test snippet"
        assert result.source == "pubmed"
        assert result.reliability_score == 0.85
        assert result.medical_relevance == 0.9
        assert "glioblastoma" in result.extracted_entities
