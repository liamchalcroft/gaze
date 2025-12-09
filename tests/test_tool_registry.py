from pathlib import Path

import pytest
from PIL import Image

from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.exceptions import UnknownToolError
from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import encode_image
from radiant_harness.types import ToolResult


async def _echo_tool(registry: ToolRegistry, value: int) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="echo", description=f"echo {value}", metadata={"value": value})


async def _needs_image(registry: ToolRegistry) -> ToolResult:
    # Use the image manager properly
    image_manager = registry.get_image_manager()
    image_manager.transform_image(lambda img: img.copy())
    image = image_manager.current_image
    if image is None:
        raise ToolExecutionError("missing image")
    encoded = encode_image(image)
    return ToolResult(
        tool_name="needs_image",
        description="used active image",
        image_base64=encoded.data,
        image_mime_type=encoded.mime_type,
    )


def _create_temp_image(tmp_path: Path) -> Path:
    path = tmp_path / "test.png"
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(path)
    return path


def test_schema_requires_explicit_type() -> None:
    bad_tool = Tool(
        name="no_type",
        description="missing type should fail",
        parameters={"bad": {}},
        execute=_echo_tool,
        requires_image=False,
    )

    registry = ToolRegistry(tools=[bad_tool])
    with pytest.raises(ValueError):
        registry.get_tool_schemas()


@pytest.mark.asyncio
async def test_unknown_tool_raises() -> None:
    registry = ToolRegistry(tools=[])
    with pytest.raises(UnknownToolError):
        await registry.execute("nope")


@pytest.mark.asyncio
async def test_requires_image_without_path(tmp_path: Path) -> None:
    tool = Tool(
        name="needs_image",
        description="requires image load",
        parameters={},
        execute=_needs_image,
        requires_image=True,
    )

    registry = ToolRegistry(tools=[tool])
    with pytest.raises(ToolExecutionError):
        await registry.execute("needs_image")

    image_path = _create_temp_image(tmp_path)
    registry.get_image_manager().set_image(image_path)
    result = await registry.execute("needs_image")
    assert result.success
    assert result.image_base64


@pytest.mark.asyncio
async def test_execution_records_history() -> None:
    tool = Tool(
        name="echo",
        description="simple echo",
        parameters={"value": {"type": "integer", "description": "number to echo"}},
        execute=_echo_tool,
        requires_image=False,
    )

    registry = ToolRegistry(tools=[tool])
    result = await registry.execute("echo", value=2)
    assert result.metadata["value"] == 2
    assert registry.history[-1].tool_name == "echo"


@pytest.mark.asyncio
async def test_history_limit_prevents_memory_leak() -> None:
    """Test that history limit prevents unbounded memory growth."""
    tool = Tool(
        name="echo",
        description="simple echo",
        parameters={"value": {"type": "integer", "description": "number to echo"}},
        execute=_echo_tool,
        requires_image=False,
    )

    registry = ToolRegistry(max_history=3, tools=[tool])

    # Execute tool more times than history limit
    for i in range(5):
        await registry.execute("echo", value=i)

    # History should not exceed limit
    assert len(registry.history) <= 3
    assert len(registry._tool_history) <= 3


def test_backward_compatibility_removed() -> None:
    """Test that backward compatibility properties have been removed."""

    registry = ToolRegistry(tools=[])

    # These properties should NOT exist anymore
    assert not hasattr(registry, "current_image")
    assert not hasattr(registry, "image_path")
    assert not hasattr(registry, "tools")
    assert not hasattr(registry, "transform_image")

    # But manager accessors should exist
    assert hasattr(registry, "get_image_manager")
    assert hasattr(registry, "get_documenter")
