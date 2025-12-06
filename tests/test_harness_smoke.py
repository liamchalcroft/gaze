import pytest

from radiant_harness.tools import Tool
from radiant_harness.tools import ToolRegistry
from radiant_harness.tools import encode_image
from radiant_harness.types import ToolResult


async def _echo_tool(registry: ToolRegistry, value: int) -> ToolResult:  # noqa: ARG001
    return ToolResult(tool_name="echo", description=f"echo {value}", metadata={"value": value})


def test_schema_requires_explicit_type() -> None:
    bad_tool = Tool(
        name="no_type",
        description="missing type should fail",
        parameters={"bad": {}},
        execute=_echo_tool,
        requires_image=False,
    )
    registry = ToolRegistry(image_path=None, tools=[bad_tool])
    with pytest.raises(ValueError):
        registry.get_tool_schemas()


@pytest.mark.asyncio
async def test_schema_and_execution_round_trip() -> None:
    good_tool = Tool(
        name="echo",
        description="simple echo",
        parameters={"value": {"type": "integer", "description": "number to echo"}},
        execute=_echo_tool,
        requires_image=False,
    )
    registry = ToolRegistry(image_path=None, tools=[good_tool])

    schemas = registry.get_tool_schemas()
    assert schemas[0]["function"]["parameters"]["properties"]["value"]["type"] == "integer"

    result = await registry.execute("echo", value=7)
    assert result.success
    assert result.metadata["value"] == 7


def test_encode_image_preserves_grayscale_mode() -> None:
    from PIL import Image

    img = Image.new("L", (4, 4), color=128)
    encoded = encode_image(img)
    assert encoded.mime_type == "image/png"
    assert encoded.data  # non-empty base64

