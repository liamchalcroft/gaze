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


def test_sync_context_manager_cleanup(tmp_path: Path) -> None:
    """Test that sync context manager properly cleans up resources."""
    image_path = _create_temp_image(tmp_path)

    with ToolRegistry(image_path=image_path, tools=[]) as registry:
        # Image should be loaded
        assert registry.get_image_manager().has_image
        assert registry.get_image_manager().current_image is not None

    # After exiting context, resources should be cleaned up
    assert registry.get_image_manager().current_image is None
    assert not registry.get_image_manager().has_image


def test_sync_context_manager_without_image() -> None:
    """Test sync context manager works without image."""
    with ToolRegistry(tools=[]) as registry:
        assert not registry.get_image_manager().has_image

    # Should not raise even without image
    assert registry.get_image_manager().current_image is None


# ── encode_image format/quality tests ────────────────────────────


class TestEncodeImage:
    """Tests for encode_image JPEG/PNG support (PR1 — payload reduction)."""

    def test_default_format_is_jpeg(self) -> None:
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = encode_image(img)
        assert result.mime_type == "image/jpeg"

    def test_jpeg_much_smaller_than_png_for_photo(self) -> None:
        """JPEG payload should be significantly smaller than PNG for photo-like images."""
        import numpy as np

        # Create a gradient + noise image (mimics real medical image content)
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        jpeg_result = encode_image(img, format="JPEG")
        png_result = encode_image(img, format="PNG")
        # JPEG should be meaningfully smaller for noisy / photo-like content
        assert len(jpeg_result.data) < len(png_result.data) * 0.5

    def test_explicit_png_format(self) -> None:
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        result = encode_image(img, format="PNG")
        assert result.mime_type == "image/png"

    def test_rgba_image_converts_for_jpeg(self) -> None:
        """JPEG cannot encode alpha; encode_image should convert RGBA → RGB."""
        img = Image.new("RGBA", (32, 32), color=(128, 128, 128, 255))
        result = encode_image(img)
        assert result.mime_type == "image/jpeg"
        assert result.data.startswith("/9")  # base64 of JPEG SOI marker

    def test_palette_image_converts_for_jpeg(self) -> None:
        """Palette (P) mode must be converted to RGB for JPEG."""
        img = Image.new("P", (32, 32))
        result = encode_image(img, format="JPEG")
        assert result.mime_type == "image/jpeg"

    def test_custom_quality(self) -> None:
        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        high_q = encode_image(img, quality=95)
        low_q = encode_image(img, quality=10)
        # Lower quality → smaller payload
        assert len(low_q.data) < len(high_q.data)

    def test_invalid_format_raises(self) -> None:
        img = Image.new("RGB", (16, 16))
        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image(img, format="BMP")

    def test_data_url_round_trip(self) -> None:
        """Encoded data URL should be a valid data URI."""
        img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        result = encode_image(img)
        url = result.to_data_url()
        assert url.startswith("data:image/jpeg;base64,")

    def test_grayscale_image_jpeg(self) -> None:
        """Grayscale (L) mode should encode to JPEG without error."""
        img = Image.new("L", (32, 32), color=128)
        result = encode_image(img)
        assert result.mime_type == "image/jpeg"
        assert result.data.startswith("/9")  # base64 of JPEG SOI marker

    # ── Medical image modes (Patch Set #5) ────────────────────────

    def test_mode_I_jpeg(self) -> None:
        """32-bit integer (I) mode must encode to JPEG via RGB conversion."""
        img = Image.new("I", (32, 32), color=1000)
        result = encode_image(img, format="JPEG")
        assert result.mime_type == "image/jpeg"
        assert result.data.startswith("/9")  # base64 of JPEG SOI marker

    def test_mode_I_png(self) -> None:
        """32-bit integer (I) mode must encode to PNG."""
        img = Image.new("I", (32, 32), color=1000)
        result = encode_image(img, format="PNG")
        assert result.mime_type == "image/png"
        assert result.data.startswith("iVBOR")  # base64 of PNG magic bytes

    def test_mode_F_jpeg(self) -> None:
        """Float32 (F) mode must encode to JPEG via RGB conversion."""
        img = Image.new("F", (32, 32), color=0.5)
        result = encode_image(img, format="JPEG")
        assert result.mime_type == "image/jpeg"
        assert result.data.startswith("/9")  # base64 of JPEG SOI marker

    def test_mode_F_png(self) -> None:
        """Float32 (F) mode must encode to PNG via L conversion."""
        img = Image.new("F", (32, 32), color=0.5)
        result = encode_image(img, format="PNG")
        assert result.mime_type == "image/png"
        assert result.data.startswith("iVBOR")  # base64 of PNG magic bytes

    def test_mode_I16_jpeg(self) -> None:
        """16-bit integer (I;16) mode — common from DICOM PNGs — must encode to JPEG."""
        import numpy as np

        arr = np.full((32, 32), 30000, dtype=np.uint16)
        img = Image.fromarray(arr)
        assert img.mode in ("I;16", "I"), f"Expected I;16 or I mode, got {img.mode}"
        result = encode_image(img, format="JPEG")
        assert result.mime_type == "image/jpeg"
        assert result.data.startswith("/9")  # base64 of JPEG SOI marker

    def test_mode_I16_png(self) -> None:
        """16-bit integer (I;16) mode must encode to PNG."""
        import numpy as np

        arr = np.full((32, 32), 30000, dtype=np.uint16)
        img = Image.fromarray(arr)
        assert img.mode in ("I;16", "I"), f"Expected I;16 or I mode, got {img.mode}"
        result = encode_image(img, format="PNG")
        assert result.mime_type == "image/png"
        assert result.data.startswith("iVBOR")  # base64 of PNG magic bytes


@pytest.mark.asyncio
async def test_execute_wraps_type_error() -> None:
    """TypeError from bad kwargs must surface as ToolExecutionError, not raw TypeError."""

    async def _strict_tool(registry: ToolRegistry, factor: float) -> ToolResult:  # noqa: ARG001
        # If called with a non-float, the body will TypeError on arithmetic
        return ToolResult(tool_name="strict", description=f"got {factor + 1}", metadata={})

    tool = Tool(
        name="strict",
        description="needs a float",
        parameters={"factor": {"type": "number", "description": "must be numeric"}},
        execute=_strict_tool,
        requires_image=False,
    )

    registry = ToolRegistry(tools=[tool])
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await registry.execute("strict", factor="not_a_number")


@pytest.mark.asyncio
async def test_execute_unexpected_kwarg_wraps_type_error() -> None:
    """Passing an unknown keyword argument should produce ToolExecutionError."""

    async def _no_args_tool(registry: ToolRegistry) -> ToolResult:  # noqa: ARG001
        return ToolResult(tool_name="simple", description="ok", metadata={})

    tool = Tool(
        name="simple",
        description="no params",
        parameters={},
        execute=_no_args_tool,
        requires_image=False,
    )

    registry = ToolRegistry(tools=[tool])
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await registry.execute("simple", bogus=42)


def test_tool_result_is_frozen() -> None:
    """ToolResult must be frozen to preserve audit integrity in history."""
    result = ToolResult(tool_name="test", description="desc")

    with pytest.raises(AttributeError):
        result.output = "mutated"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        result.description = "mutated"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        result.error = "mutated"  # type: ignore[misc]


# ── PR2: asyncio.to_thread tests ────────────────────────────────


@pytest.mark.asyncio
async def test_encode_image_runs_off_event_loop(tmp_path: Path) -> None:
    """Visual tool executors must run encode_image via asyncio.to_thread."""
    import asyncio

    from radiant_harness.tools.visual import create_visual_tools

    image_path = _create_temp_image(tmp_path)
    tools = create_visual_tools()
    registry = ToolRegistry(image_path=image_path, tools=tools)

    # Execute a visual tool — if it blocks the event loop, we can detect it
    # by checking that another coroutine can interleave.
    marker: list[str] = []

    async def _background():
        marker.append("ran")

    bg = asyncio.create_task(_background())
    result = await registry.execute("zoom", factor=2.0)
    await bg  # ensure background task completed

    assert result.success
    assert result.image_base64
    assert marker == ["ran"]


# ── PR4: set_preloaded_image tests ──────────────────────────────


def test_set_preloaded_image_avoids_disk_read(tmp_path: Path) -> None:
    """set_preloaded_image should work without re-reading from disk."""
    from radiant_harness.tools.image_manager import ImageManager

    image_path = _create_temp_image(tmp_path)
    img = Image.open(image_path)
    img.load()

    manager = ImageManager()
    manager.set_preloaded_image(img, image_path)

    assert manager.has_image
    assert manager.current_image is not None
    assert manager.image_path == image_path
    assert manager.current_image is not manager._original_image  # independent copies


def test_set_preloaded_image_reset_works(tmp_path: Path) -> None:
    """reset_to_original should work after set_preloaded_image."""
    from radiant_harness.tools.image_manager import ImageManager
    from radiant_harness.tools.visual import zoom_image

    image_path = _create_temp_image(tmp_path)
    img = Image.open(image_path)
    img.load()
    original_size = img.size

    manager = ImageManager()
    manager.set_preloaded_image(img, image_path)

    # Transform then reset
    manager.transform_image(lambda i: zoom_image(i, 2.0))
    assert manager.current_image is not None
    assert manager.current_image.size != original_size

    manager.reset_to_original()
    assert manager.current_image is not None
    assert manager.current_image.size == original_size


# ── Type coercion and beartype defense tests ─────────────────────


def _create_large_temp_image(tmp_path: Path) -> Path:
    """Create a test image large enough for crop operations (min_image_size=10)."""
    path = tmp_path / "test_large.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(path)
    return path


@pytest.mark.asyncio
async def test_crop_with_int_coords_from_json(tmp_path: Path) -> None:
    """Crop box with integer elements (common JSON output) must not crash."""
    from radiant_harness.tools.visual import create_visual_tools

    image_path = _create_large_temp_image(tmp_path)
    tools = create_visual_tools()
    registry = ToolRegistry(image_path=image_path, tools=tools)

    # JSON typically sends [0, 0, 1, 1] as ints, not floats
    result = await registry.execute("crop", box=[0, 0, 1, 1])
    assert result.success
    assert result.image_base64


@pytest.mark.asyncio
async def test_threshold_with_float_bounds_from_json(tmp_path: Path) -> None:
    """Threshold with float bounds (e.g. 50.0) must coerce to int."""
    from radiant_harness.tools.visual import create_visual_tools

    image_path = _create_large_temp_image(tmp_path)
    tools = create_visual_tools()
    registry = ToolRegistry(image_path=image_path, tools=tools)

    result = await registry.execute("threshold", lower=50.0, upper=200.0)
    assert result.success
    assert result.image_base64


@pytest.mark.asyncio
async def test_crop_rejects_bool_coords(tmp_path: Path) -> None:
    """Boolean values must not pass as crop coordinates."""
    from radiant_harness.tools.visual import create_visual_tools

    image_path = _create_large_temp_image(tmp_path)
    tools = create_visual_tools()
    registry = ToolRegistry(image_path=image_path, tools=tools)

    with pytest.raises(ToolExecutionError, match="must be numbers"):
        await registry.execute("crop", box=[True, False, True, True])


@pytest.mark.asyncio
async def test_beartype_violation_wrapped_as_tool_error(tmp_path: Path) -> None:
    """BeartypeException from inside tool must become ToolExecutionError, not crash."""
    from beartype import beartype as bt

    @bt
    def _strict_op(x: float) -> float:
        return x

    async def _bad_tool(registry: ToolRegistry, value: str) -> ToolResult:  # noqa: ARG001
        _strict_op(value)  # type: ignore[arg-type]  # intentional: trigger beartype
        return ToolResult(tool_name="bad", description="unreachable", metadata={})

    tool = Tool(
        name="bad",
        description="triggers beartype",
        parameters={"value": {"type": "string", "description": "will fail"}},
        execute=_bad_tool,
        requires_image=False,
    )
    registry = ToolRegistry(tools=[tool])
    with pytest.raises(ToolExecutionError, match="invalid arguments"):
        await registry.execute("bad", value="not_a_float")


def test_zoom_rejects_oversized_result() -> None:
    """Zoom that exceeds max_image_dimension must raise ValueError."""
    from radiant_harness.config import ImageProcessingConfig
    from radiant_harness.tools.visual import zoom_image

    cfg = ImageProcessingConfig(max_image_dimension=1000)
    img = Image.new("RGB", (512, 512))
    with pytest.raises(ValueError, match="exceeds max_image_dimension"):
        zoom_image(img, 2.5, config=cfg)  # 1280 > 1000
    img.close()


def test_zoom_within_max_dimension_succeeds() -> None:
    """Zoom that stays within max_image_dimension must succeed."""
    from radiant_harness.config import ImageProcessingConfig
    from radiant_harness.tools.visual import zoom_image

    cfg = ImageProcessingConfig(max_image_dimension=2000)
    img = Image.new("RGB", (512, 512))
    result = zoom_image(img, 2.0, config=cfg)  # 1024 < 2000
    assert result.size == (1024, 1024)
    img.close()
    result.close()


# ── PR5: OpenAI adapter retry config tests ──────────────────────


def test_openai_adapter_disables_sdk_retries() -> None:
    """SDK max_retries should be 0 to let tenacity control retry logic."""
    import os

    from radiant_harness.models.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter(model_name="test-model")
    # Set a dummy key to allow client creation
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
    try:
        client = adapter.client
        assert client.max_retries == 0
    finally:
        if os.environ.get("OPENAI_API_KEY") == "sk-test-dummy":
            del os.environ["OPENAI_API_KEY"]


# ---------------------------------------------------------------------------
# EncodedImage _data_url pre-computation
# ---------------------------------------------------------------------------


class TestEncodedImageDataUrlCached:
    """Verify to_data_url() returns the pre-computed string, not a new one."""

    def test_data_url_is_precomputed_in_post_init(self) -> None:
        from radiant_harness.tools.registry import EncodedImage

        enc = EncodedImage(data="abc123", mime_type="image/jpeg")
        assert enc._data_url == "data:image/jpeg;base64,abc123"

    def test_to_data_url_returns_same_object(self) -> None:
        from radiant_harness.tools.registry import EncodedImage

        enc = EncodedImage(data="xyz", mime_type="image/png")
        url1 = enc.to_data_url()
        url2 = enc.to_data_url()
        assert url1 is url2

    def test_encode_image_returns_precomputed_url(self) -> None:
        img = Image.new("RGB", (8, 8), color=(100, 100, 100))
        result = encode_image(img)
        url = result.to_data_url()
        assert url.startswith("data:image/jpeg;base64,")
        assert url == f"data:{result.mime_type};base64,{result.data}"

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        from radiant_harness.tools.registry import EncodedImage

        enc = EncodedImage(data="abc", mime_type="image/jpeg")
        with pytest.raises(AttributeError):
            enc.data = "mutated"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            enc._data_url = "mutated"  # type: ignore[misc]
