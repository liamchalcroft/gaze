"""Tests for the agentic processing module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from aiohttp import ContentTypeError
from PIL import Image

from radiant_harness import Tool
from radiant_harness import ToolExecutionError
from radiant_harness import ToolRegistry
from radiant_harness import ToolResult
from radiant_harness import UnknownToolError
from radiant_harness import create_search_tools
from radiant_harness import create_visual_tools
from radiant_harness.base import AgenticProcessingError
from radiant_harness.base import AgenticProcessorBase
from radiant_harness.types import ToolCall


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
        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
        registry.set_image(temp_image)

        assert registry.current_image is not None
        assert registry.current_image.size == (100, 100)

    @pytest.mark.asyncio
    async def test_execute_zoom(self, temp_image):
        """Test zoom tool execution."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
        result = await registry.execute("zoom", factor=2.0)

        assert result.success
        assert result.tool_name == "zoom"
        assert "2.0" in result.description
        assert result.image_base64 is not None
        assert registry.current_image.size == (200, 200)

    @pytest.mark.asyncio
    async def test_execute_crop(self, temp_image):
        """Test crop tool execution."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
        result = await registry.execute("adjust_contrast", factor=1.5)

        assert result.success
        assert result.tool_name == "adjust_contrast"
        assert "1.5" in result.description

    @pytest.mark.asyncio
    async def test_execute_threshold(self, temp_image):
        """Test intensity threshold tool."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
        result = await registry.execute("threshold", lower=50, upper=200)

        assert result.success
        assert result.tool_name == "threshold"
        assert "[50, 200]" in result.description

    def test_tool_schema_includes_validation_keywords(self):
        """Tool schemas should propagate validation keywords to the model."""
        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(tools=tools)
        schemas = registry.get_tool_schemas()

        zoom = next(s for s in schemas if s["function"]["name"] == "zoom")
        zoom_factor = zoom["function"]["parameters"]["properties"]["factor"]
        assert zoom_factor["minimum"] == 0.5
        assert zoom_factor["maximum"] == 4.0

        crop = next(s for s in schemas if s["function"]["name"] == "crop")
        crop_box = crop["function"]["parameters"]["properties"]["box"]
        assert crop_box["minItems"] == 4
        assert crop_box["maxItems"] == 4

        search_web = next(s for s in schemas if s["function"]["name"] == "search_web")
        search_type = search_web["function"]["parameters"]["properties"]["search_type"]
        assert set(search_type["enum"]) == {
            "diagnosis",
            "research",
            "guidelines",
            "anatomy",
            "general",
        }

    @pytest.mark.asyncio
    async def test_execute_flip_horizontal(self, temp_image):
        """Test flip horizontal tool."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

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
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)
        with pytest.raises(UnknownToolError) as exc_info:
            await registry.execute("unknown_tool")

        assert exc_info.value.tool_name == "unknown_tool"
        assert "unknown_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_without_image(self):
        """Test executing tool without image raises ToolExecutionError."""
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("zoom", factor=2.0)

        assert "requires an image" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_history(self, temp_image):
        """Test that tool history is tracked."""
        tools = create_visual_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        await registry.execute("zoom", factor=1.5)
        await registry.execute("adjust_contrast", factor=1.2)

        history = registry.history
        assert len(history) == 2
        assert history[0].tool_name == "zoom"
        assert history[1].tool_name == "adjust_contrast"

    @pytest.mark.asyncio
    async def test_agentic_processor_fails_fast_on_unknown_tool(self, temp_image):
        """Agentic processor should surface unknown tool errors instead of masking."""

        class MinimalProcessor(AgenticProcessorBase):
            def get_system_prompt(self, images, metadata):
                return "system"

            def get_user_message(self, images, metadata):
                return "user"

            def get_response_schema(self):
                return {"type": "object"}

            def validate_response(self, response):
                return True

        registry = ToolRegistry(image_path=temp_image, tools=[])
        processor = MinimalProcessor(use_tools=True)
        tool_call = ToolCall(id="1", name="missing_tool", arguments={})

        with pytest.raises(AgenticProcessingError):
            await processor._execute_tools([tool_call], registry, turn_idx=0)

    def test_get_tool_schemas_format(self):
        """Test that tool schemas are in OpenAI format."""
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)
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
        tools = create_visual_tools()
        registry = ToolRegistry(tools=tools)

        async def custom_execute(registry, **kwargs):
            return ToolResult(
                tool_name="custom",
                description="Custom tool executed",
            )

        custom_tool = Tool(
            name="custom",
            description="A custom tool",
            parameters={"value": {"type": "number"}},
            execute=custom_execute,
            requires_image=False,
        )

        registry.register(custom_tool)
        schemas = registry.get_tool_schemas()
        tool_names = {s["function"]["name"] for s in schemas}
        assert "custom" in tool_names

    @pytest.mark.asyncio
    async def test_execute_normalizes_value_errors(self):
        """ValueError from a tool is wrapped as ToolExecutionError."""

        async def failing_execute(_registry, **_kwargs):
            raise ValueError("bad input")

        failing_tool = Tool(
            name="failing",
            description="Fails with ValueError",
            parameters={},
            execute=failing_execute,
            requires_image=False,
        )

        registry = ToolRegistry(tools=[failing_tool])

        with pytest.raises(ToolExecutionError):
            await registry.execute("failing")


class _DummyProcessor(AgenticProcessorBase):
    """Minimal concrete processor for validation tests."""

    def get_system_prompt(self, images, metadata):
        return "sys"

    def get_user_message(self, images, metadata):
        return "user"

    def get_response_schema(self):
        return None

    def validate_response(self, response):
        return True


@pytest.fixture
def dummy_processor():
    return _DummyProcessor(model_name="test-model", use_tools=False, max_turns=2)


class TestAgenticValidation:
    """Validation and error-handling tests for agent loop."""

    @pytest.mark.asyncio
    async def test_execute_tools_accepts_dict_arguments(self, dummy_processor):
        tool_registry = MagicMock()
        tool_registry.execute = AsyncMock(return_value=ToolResult(tool_name="noop", description="ok"))
        tool_call = ToolCall(id="1", name="noop", arguments={"a": 1})

        results = await dummy_processor._execute_tools([tool_call], tool_registry, turn_idx=0)
        assert len(results) == 1
        tool_registry.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_tools_rejects_non_object_json(self, dummy_processor):
        tool_registry = MagicMock()
        tool_call = ToolCall(id="1", name="noop", arguments="[]")

        with pytest.raises(AgenticProcessingError):
            await dummy_processor._execute_tools([tool_call], tool_registry, turn_idx=0)

    @pytest.mark.asyncio
    async def test_run_analysis_rejects_non_bool_continue(self, dummy_processor):
        mock_adapter = MagicMock()
        mock_log = MagicMock()
        mock_log.tokens = 1
        mock_adapter.generate_chat = AsyncMock(
            return_value=('{"continue": "yes"}', None, mock_log)
        )
        dummy_processor._model_adapter = mock_adapter

        with pytest.raises(AgenticProcessingError):
            await dummy_processor._run_analysis(images=[], metadata={}, tool_registry=None)

    @pytest.mark.asyncio
    async def test_run_analysis_rejects_non_dict_response(self, dummy_processor):
        mock_adapter = MagicMock()
        mock_log = MagicMock()
        mock_log.tokens = 1
        mock_adapter.generate_chat = AsyncMock(return_value=('["not_a_dict"]', None, mock_log))
        dummy_processor._model_adapter = mock_adapter

        with pytest.raises(AgenticProcessingError):
            await dummy_processor._run_analysis(images=[], metadata={}, tool_registry=None)

    def test_disabled_tools(self):
        """Test that disabled_tools prevents tool registration."""
        disabled = {"zoom", "crop", "search_web"}
        tools = create_visual_tools(disabled) + create_search_tools(disabled)
        registry = ToolRegistry(tools=tools)
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
        """Test that empty disabled_tools set registers all tools."""
        tools_all = create_visual_tools(set()) + create_search_tools(set())
        tools_none = create_visual_tools() + create_search_tools()
        registry_all = ToolRegistry(tools=tools_all)
        registry_none = ToolRegistry(tools=tools_none)

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


class TestNOVAAgenticProcessor:
    """Tests for the NOVAAgenticProcessor class."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        from nova_retrieval_vlm.nova import NOVAAgenticProcessor

        processor = NOVAAgenticProcessor(
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
        from nova_retrieval_vlm.nova import NOVAAgenticProcessor

        # Mock response with required NOVA fields
        mock_response = '''{"caption": {"description": "test"}, "diagnosis": {"primary_diagnosis": "test"}, "localization": {"localizations": []}, "continue": false}'''
        mock_adapter = MagicMock()
        mock_log = MagicMock()
        mock_log.tokens = 100
        mock_adapter.generate_chat = AsyncMock(return_value=(mock_response, None, mock_log))

        with patch(
            "radiant_harness.base.OpenAIAdapter",
            return_value=mock_adapter,
        ):
            processor = NOVAAgenticProcessor(
                model_name="openai/gpt-4o",
                use_tools=False,
            )

            result = await processor.analyze(
                images=temp_image,
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


class TestCLIAgenticIntegration:
    """Tests for CLI integration with processors."""

    def test_create_processor_standard(self):
        """Test creating standard processor."""
        from nova_retrieval_vlm.cli import create_processor
        from nova_retrieval_vlm.config import AgenticConfig
        from nova_retrieval_vlm.config import Config
        from nova_retrieval_vlm.processors import LocalizationProcessor

        config = Config(agentic=AgenticConfig(enabled=False))
        processor = create_processor(config)

        assert isinstance(processor, LocalizationProcessor)

    def test_create_processor_always_returns_localization(self):
        """Test create_processor always returns LocalizationProcessor.

        For agentic mode, NOVAAgenticProcessor is used directly in run_task,
        not via create_processor.
        """
        from nova_retrieval_vlm.cli import create_processor
        from nova_retrieval_vlm.config import AgenticConfig
        from nova_retrieval_vlm.config import Config
        from nova_retrieval_vlm.processors import LocalizationProcessor

        config = Config(
            agentic=AgenticConfig(
                enabled=True,
                reasoning_enabled=True,
                use_tools=True,
                max_turns=3,
            )
        )
        processor = create_processor(config)

        # create_processor always returns LocalizationProcessor
        # Agentic mode uses NOVAAgenticProcessor directly in run_task
        assert isinstance(processor, LocalizationProcessor)


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
        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(tools=tools)
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
    @patch("radiant_harness.tools.search.search_medical_literature")
    async def test_search_web_tool_execution(self, mock_search, temp_image):
        """Test search_web tool execution with mocking."""
        # Mock successful search result
        from radiant_harness.retrieval.web_search import SearchResult

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

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

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
    @patch("radiant_harness.tools.search.search_medical_literature")
    async def test_search_web_error_handling(self, mock_search, temp_image):
        """Test search_web tool error handling."""
        from radiant_harness.retrieval.web_search import SearchError

        mock_search.side_effect = SearchError("test", "Search failed")

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        result = await registry.execute("search_web", query="test query")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    @patch("radiant_harness.tools.search.search_medical_literature")
    async def test_search_web_empty_results(self, mock_search, temp_image):
        """Test search_web tool with empty results."""
        mock_search.return_value = []

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        result = await registry.execute("search_web", query="obscure condition")

        # Empty results should be success=True (no results found is valid)
        assert result.success is True
        assert (
            "no" in result.description.lower()
            or "0" in result.description
        )
        assert result.metadata["results_count"] == 0

    @pytest.mark.asyncio
    async def test_search_only_mode_exposes_tools(self):
        """Tools must be available when only web search is enabled."""

        class DummyAdapter:
            def __init__(self):
                self.last_tools = None

            async def generate_chat(
                self,
                messages,
                max_tokens,
                temperature,
                tools,
                response_format,
            ):
                self.last_tools = tools
                return ('{"continue": false}', None, SimpleNamespace(tokens=1))

        class DummyProcessor(AgenticProcessorBase):
            def _ensure_initialized(self):
                self._model_adapter = DummyAdapter()

            def get_system_prompt(self, images, metadata):
                return "system"

            def get_user_message(self, images, metadata):
                return "user"

            def get_response_schema(self):
                return None

            def validate_response(self, response):
                return True

        processor = DummyProcessor(use_tools=False, use_web_search=True, max_turns=2)
        await processor.analyze(images=None, metadata={})

        tools = processor._model_adapter.last_tools  # type: ignore[attr-defined]
        assert tools is not None
        tool_names = {schema["function"]["name"] for schema in tools}
        assert "search_web" in tool_names

    @patch("radiant_harness.retrieval.web_search.PubMedSearchEngine")
    def test_search_result_creation(self, _mock_engine):
        """Test SearchResult dataclass creation and validation."""
        from radiant_harness.retrieval.web_search import SearchResult

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
        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(tools=tools)
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
    @patch("radiant_harness.tools.search.search_medical_images")
    async def test_search_images_tool_execution(self, mock_search, temp_image):
        """Test search_images tool execution with mocking."""
        from radiant_harness.retrieval.image_search import ImageSearchResult

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

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

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
    @patch("radiant_harness.tools.search.search_medical_images")
    async def test_search_images_with_filters(self, mock_search, temp_image):
        """Test search_images tool with modality and body_part filters."""
        from radiant_harness.retrieval.image_search import ImageSearchResult

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

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        result = await registry.execute(
            "search_images", query="hemorrhage", modality="CT", body_part="brain"
        )

        assert result.success is True
        mock_search.assert_called_once_with(
            query="hemorrhage", max_results=5, modality="CT", body_part="brain"
        )

    @pytest.mark.asyncio
    @patch("radiant_harness.tools.search.search_medical_images")
    async def test_search_images_error_handling(self, mock_search, temp_image):
        """Test search_images tool error handling."""
        from radiant_harness.retrieval.image_search import ImageSearchError

        mock_search.side_effect = ImageSearchError("Open-i", "API unavailable")

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        result = await registry.execute("search_images", query="test query")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    @patch("radiant_harness.tools.search.search_medical_images")
    async def test_search_images_empty_results(self, mock_search, temp_image):
        """Test search_images tool with empty results."""
        mock_search.return_value = []

        tools = create_visual_tools() + create_search_tools()
        registry = ToolRegistry(image_path=temp_image, tools=tools)

        result = await registry.execute("search_images", query="extremely rare condition")

        # Empty results should be success=True
        assert result.success is True
        assert "no" in result.description.lower() or "0" in result.description
        assert result.metadata["results_count"] == 0

    def test_image_search_result_creation(self):
        """Test ImageSearchResult dataclass creation and validation."""
        from radiant_harness.retrieval.image_search import ImageSearchResult

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


class TestHarnessRegressions:
    """Regression tests for recent harness fixes."""

    @pytest.mark.asyncio
    async def test_web_search_cache_key_respects_enhancement(self):
        """Cache keys must differ for enhanced vs raw queries."""
        from radiant_harness.retrieval.web_search import SearchEngine
        from radiant_harness.retrieval.web_search import SearchResult
        from radiant_harness.retrieval.web_search import WebSearchManager

        class DummyEngine(SearchEngine):
            def __init__(self):
                super().__init__("dummy", timeout=1, max_retries=1)

            async def _search_impl(self, query, max_results):
                return [
                    SearchResult(
                        title="Dummy Reliable Result",
                        url="https://example.com/dummy",
                        content="Medical content",
                        snippet="Medical content",
                        source="dummy",
                        reliability_score=0.8,
                        medical_relevance=0.5,
                    )
                ]

        class TestableWebSearchManager(WebSearchManager):
            def __init__(self):
                # Skip parent init to avoid external dependencies
                self.max_results_per_engine = 2
                self.max_total_results = 5
                self.cache_duration = 300
                self.rate_limit_delay = 0
                self._cache = {}
                self.engines = [DummyEngine()]

        manager = TestableWebSearchManager()

        await manager.search("glioma", search_type="general", enhance_query=True)
        await manager.search("glioma", search_type="general", enhance_query=False)

        keys = list(manager._cache.keys())
        assert any("enh=True" in key for key in keys)
        assert any("enh=False" in key for key in keys)
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_openi_non_json_is_hard_failure(self):
        """Non-JSON Open-i responses should raise ImageSearchError."""
        from unittest.mock import Mock

        from radiant_harness.retrieval.image_search import ImageSearchError
        from radiant_harness.retrieval.image_search import OpenISearchEngine

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                raise ContentTypeError(request_info=Mock(), history=(), message="not json")

            async def text(self):
                return "plain text response"

        class FakeSession:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, *args, **kwargs):
                return self._response

        engine = OpenISearchEngine()
        fake_response = FakeResponse()
        engine._get_session = AsyncMock(return_value=FakeSession(fake_response))  # type: ignore[attr-defined]

        with pytest.raises(ImageSearchError):
            await engine._search_impl("query", max_results=1)
