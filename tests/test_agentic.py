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

    @pytest.mark.asyncio
    async def test_execute_zoom(self, temp_image):
        """Test zoom tool execution."""
        registry = ToolRegistry(temp_image)
        result = await registry.execute("zoom", factor=2.0)

        assert result.success
        assert result.tool_name == "zoom"
        assert "2.0" in result.description
        assert result.image_base64 is not None
        assert registry.current_image.size == (200, 200)

    @pytest.mark.asyncio
    async def test_execute_crop(self, temp_image):
        """Test crop tool execution."""
        registry = ToolRegistry(temp_image)
        result = await registry.execute("crop", box=[0.25, 0.25, 0.75, 0.75])

        assert result.success
        assert result.tool_name == "crop"
        assert result.image_base64 is not None
        # Cropped to 50% of each dimension
        assert registry.current_image.size[0] <= 100
        assert registry.current_image.size[1] <= 100

    @pytest.mark.asyncio
    async def test_execute_contrast(self, temp_image):
        """Test contrast adjustment tool."""
        registry = ToolRegistry(temp_image)
        result = await registry.execute("adjust_contrast", factor=1.5)

        assert result.success
        assert result.tool_name == "adjust_contrast"
        assert "1.5" in result.description

    @pytest.mark.asyncio
    async def test_execute_threshold(self, temp_image):
        """Test intensity threshold tool."""
        registry = ToolRegistry(temp_image)
        result = await registry.execute("threshold", lower=50, upper=200)

        assert result.success
        assert result.tool_name == "threshold"
        assert "[50, 200]" in result.description

    @pytest.mark.asyncio
    async def test_execute_flip_horizontal(self, temp_image):
        """Test flip horizontal tool."""
        registry = ToolRegistry(temp_image)
        registry.set_image(temp_image)  # Explicitly load image
        original_size = registry.current_image.size

        result = await registry.execute("flip_horizontal")

        assert result.success
        assert result.tool_name == "flip_horizontal"
        assert "horizontal" in result.description.lower()
        assert result.image_base64 is not None
        # Size should remain the same after flip
        assert registry.current_image.size == original_size

    @pytest.mark.asyncio
    async def test_execute_flip_vertical(self, temp_image):
        """Test flip vertical tool."""
        registry = ToolRegistry(temp_image)
        registry.set_image(temp_image)  # Explicitly load image
        original_size = registry.current_image.size

        result = await registry.execute("flip_vertical")

        assert result.success
        assert result.tool_name == "flip_vertical"
        assert "vertical" in result.description.lower()
        assert result.image_base64 is not None
        # Size should remain the same after flip
        assert registry.current_image.size == original_size

    @pytest.mark.asyncio
    async def test_execute_rotate_clockwise(self, temp_image):
        """Test rotate tool clockwise."""
        registry = ToolRegistry(temp_image)
        registry.set_image(temp_image)  # Explicitly load image
        original_size = registry.current_image.size

        result = await registry.execute("rotate", clockwise=True)

        assert result.success
        assert result.tool_name == "rotate"
        assert "clockwise" in result.description.lower()
        assert result.image_base64 is not None
        # For square images, size should be swapped (W, H) -> (H, W)
        assert registry.current_image.size == (original_size[1], original_size[0])

    @pytest.mark.asyncio
    async def test_execute_rotate_counterclockwise(self, temp_image):
        """Test rotate tool counter-clockwise."""
        registry = ToolRegistry(temp_image)
        registry.set_image(temp_image)  # Explicitly load image
        original_size = registry.current_image.size

        result = await registry.execute("rotate", clockwise=False)

        assert result.success
        assert result.tool_name == "rotate"
        assert "counter-clockwise" in result.description.lower()
        assert result.image_base64 is not None
        # For square images, size should be swapped
        assert registry.current_image.size == (original_size[1], original_size[0])

    @pytest.mark.asyncio
    async def test_execute_reset(self, temp_image):
        """Test reset tool."""
        registry = ToolRegistry(temp_image)

        # Modify image first
        await registry.execute("zoom", factor=2.0)
        assert registry.current_image.size == (200, 200)

        # Reset
        result = await registry.execute("reset")
        assert result.success
        assert registry.current_image.size == (100, 100)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, temp_image):
        """Test executing unknown tool raises UnknownToolError."""
        from nova_retrieval_vlm.agentic.tools import UnknownToolError

        registry = ToolRegistry(temp_image)
        with pytest.raises(UnknownToolError) as exc_info:
            await registry.execute("unknown_tool")

        assert exc_info.value.tool_name == "unknown_tool"
        assert "unknown_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_without_image(self):
        """Test executing tool without image raises ToolExecutionError."""
        from nova_retrieval_vlm.agentic.tools import ToolExecutionError

        registry = ToolRegistry()
        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("zoom", factor=2.0)

        assert "requires an image" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_history(self, temp_image):
        """Test that tool history is tracked."""
        registry = ToolRegistry(temp_image)

        await registry.execute("zoom", factor=1.5)
        await registry.execute("adjust_contrast", factor=1.2)

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

        async def custom_execute(**kwargs):
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

    def test_disabled_tools(self):
        """Test that disabled_tools prevents tool registration."""
        registry = ToolRegistry(disabled_tools=["zoom", "crop", "search_web"])
        schemas = registry.get_tool_schemas()
        tool_names = {s["function"]["name"] for s in schemas}

        # Disabled tools should not be registered
        assert "zoom" not in tool_names
        assert "crop" not in tool_names
        assert "search_web" not in tool_names

        # Other tools should still be registered
        assert "adjust_contrast" in tool_names
        assert "threshold" in tool_names
        assert "reset" in tool_names

    def test_disabled_tools_empty_list(self):
        """Test that empty disabled_tools list registers all tools."""
        registry_all = ToolRegistry(disabled_tools=[])
        registry_none = ToolRegistry()

        schemas_all = registry_all.get_tool_schemas()
        schemas_none = registry_none.get_tool_schemas()

        # Should have same tools
        assert len(schemas_all) == len(schemas_none)


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
        from nova_retrieval_vlm.config import AgenticConfig
        from nova_retrieval_vlm.config import Config
        from nova_retrieval_vlm.processors import LocalizationProcessor

        config = Config(agentic=AgenticConfig(enabled=False))
        processor = create_processor(config)

        assert isinstance(processor, LocalizationProcessor)

    def test_create_processor_agentic(self):
        """Test creating agentic processor when enabled."""
        from nova_retrieval_vlm.agentic import AgenticLocalizationProcessor
        from nova_retrieval_vlm.cli import create_processor
        from nova_retrieval_vlm.config import AgenticConfig
        from nova_retrieval_vlm.config import Config

        config = Config(
            agentic=AgenticConfig(
                enabled=True,
                reasoning_enabled=True,
                use_tools=True,
                max_turns=3,
            )
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

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature")
    async def test_search_web_tool_execution(self, mock_search, temp_image):
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
        result = await registry.execute(
            "search_web", query="glioblastoma MRI findings", search_type="general"
        )

        assert result.success is True
        assert "Found 1" in result.description
        assert result.metadata["results_count"] == 1
        assert result.metadata["query"] == "glioblastoma MRI findings"

        # Verify the search was called with correct parameters
        mock_search.assert_called_once_with(
            query="glioblastoma MRI findings", max_results=5, search_type="general"
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature")
    async def test_search_web_error_handling(self, mock_search, temp_image):
        """Test search_web tool error handling."""
        from nova_retrieval_vlm.retrieval.web_search import SearchError

        mock_search.side_effect = SearchError("test", "Search failed")

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = await registry.execute("search_web", query="test query")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_literature")
    async def test_search_web_empty_results(self, mock_search, temp_image):
        """Test search_web tool with empty results."""
        mock_search.return_value = []

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = await registry.execute("search_web", query="obscure condition")

        # Empty results should be success=True (no results found is valid)
        assert result.success is True
        assert (
            "no" in result.description.lower()
            or "0" in result.description
        )
        assert result.metadata["results_count"] == 0

    @patch("nova_retrieval_vlm.retrieval.web_search.PubMedSearchEngine")
    def test_search_result_creation(self, _mock_engine):
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


class TestAgenticImageSearchIntegration:
    """Tests for image search integration in agentic tools."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="white")
            img.save(f.name)
            yield Path(f.name)
            # Cleanup
            Path(f.name).unlink(missing_ok=True)

    def test_tool_registry_includes_search_images(self):
        """Test that search_images tool is included in tool registry."""
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas()

        search_images_schema = next(
            (schema for schema in schemas if schema["function"]["name"] == "search_images"),
            None,
        )

        assert search_images_schema is not None, "search_images tool not found in registry"

        # Verify tool schema structure
        function_def = search_images_schema["function"]
        assert function_def["name"] == "search_images"
        assert "query" in function_def["parameters"]["required"]
        assert function_def["parameters"]["properties"]["query"]["type"] == "string"
        # Check optional parameters
        assert "modality" in function_def["parameters"]["properties"]
        assert "body_part" in function_def["parameters"]["properties"]

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_images")
    async def test_search_images_tool_execution(self, mock_search, temp_image):
        """Test search_images tool execution with mocking."""
        from nova_retrieval_vlm.retrieval.image_search import ImageSearchResult

        mock_search.return_value = [
            ImageSearchResult(
                title="Brain MRI Glioblastoma",
                image_url="https://openi.nlm.nih.gov/images/12345.jpg",
                thumbnail_url="https://openi.nlm.nih.gov/thumbs/12345.jpg",
                source_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/",
                source="openi",
                modality="MRI",
                body_part="brain",
                caption="T1-weighted MRI showing enhancing mass",
                reliability_score=0.9,
            )
        ]

        registry = ToolRegistry()
        registry.set_image(temp_image)

        # Test search_images tool execution
        result = await registry.execute("search_images", query="glioblastoma MRI")

        assert result.success is True
        assert "Found 1" in result.description
        assert result.metadata["results_count"] == 1
        assert result.metadata["query"] == "glioblastoma MRI"

        # Verify the search was called with correct parameters
        mock_search.assert_called_once_with(
            query="glioblastoma MRI", max_results=5, modality=None, body_part=None
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_images")
    async def test_search_images_with_filters(self, mock_search, temp_image):
        """Test search_images tool with modality and body_part filters."""
        from nova_retrieval_vlm.retrieval.image_search import ImageSearchResult

        mock_search.return_value = [
            ImageSearchResult(
                title="Brain CT Scan",
                image_url="https://openi.nlm.nih.gov/images/67890.jpg",
                thumbnail_url=None,
                source_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC67890/",
                source="openi",
                modality="CT",
                body_part="brain",
                reliability_score=0.85,
            )
        ]

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = await registry.execute(
            "search_images", query="hemorrhage", modality="CT", body_part="brain"
        )

        assert result.success is True
        mock_search.assert_called_once_with(
            query="hemorrhage", max_results=5, modality="CT", body_part="brain"
        )

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_images")
    async def test_search_images_error_handling(self, mock_search, temp_image):
        """Test search_images tool error handling."""
        from nova_retrieval_vlm.retrieval.image_search import ImageSearchError

        mock_search.side_effect = ImageSearchError("Open-i", "API unavailable")

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = await registry.execute("search_images", query="test query")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    @patch("nova_retrieval_vlm.agentic.tools.search_medical_images")
    async def test_search_images_empty_results(self, mock_search, temp_image):
        """Test search_images tool with empty results."""
        mock_search.return_value = []

        registry = ToolRegistry()
        registry.set_image(temp_image)

        result = await registry.execute("search_images", query="extremely rare condition")

        # Empty results should be success=True
        assert result.success is True
        assert "no" in result.description.lower() or "0" in result.description
        assert result.metadata["results_count"] == 0

    def test_image_search_result_creation(self):
        """Test ImageSearchResult dataclass creation and validation."""
        from nova_retrieval_vlm.retrieval.image_search import ImageSearchResult

        result = ImageSearchResult(
            title="Test Brain MRI",
            image_url="https://example.com/image.jpg",
            thumbnail_url="https://example.com/thumb.jpg",
            source_url="https://example.com/article",
            source="openi",
            modality="MRI",
            body_part="brain",
            diagnosis="glioblastoma",
            caption="T1-weighted MRI showing enhancing mass in right frontal lobe",
            reliability_score=0.9,
        )

        assert result.title == "Test Brain MRI"
        assert result.image_url == "https://example.com/image.jpg"
        assert result.source == "openi"
        assert result.modality == "MRI"
        assert result.body_part == "brain"
        assert result.reliability_score == 0.9

        # Test to_dict method
        result_dict = result.to_dict()
        assert result_dict["title"] == "Test Brain MRI"
        assert result_dict["modality"] == "MRI"
        assert "reliability" in result_dict
