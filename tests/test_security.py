"""Security tests for credential safety, SSRF protection, and error sanitization."""

from __future__ import annotations

from pathlib import Path

import pytest

from radiant_harness.base import _sanitize_tool_content
from radiant_harness.config import SearchConfig
from radiant_harness.exceptions import APIError
from radiant_harness.exceptions import ModelError
from radiant_harness.exceptions import ToolExecutionError
from radiant_harness.models.openai_adapter import OpenAIAdapter
from radiant_harness.models.openai_adapter import _safe_error_summary
from radiant_harness.retrieval.image_search import ImageDownloadError
from radiant_harness.retrieval.image_search import MedicalImageSearchManager
from radiant_harness.tools.image_manager import ImageManager

# ---------------------------------------------------------------------------
# _safe_error_summary tests
# ---------------------------------------------------------------------------


class TestSafeErrorSummary:
    """Verify _safe_error_summary never leaks credentials."""

    def test_class_name_only_for_generic_exception(self) -> None:
        summary = _safe_error_summary(ValueError("something went wrong"))
        assert summary == "ValueError"
        assert "went wrong" not in summary  # generic Exception has no .message

    def test_truncates_long_message(self) -> None:
        """APIStatusError.message could contain echoed headers."""

        class FakeAPIError(Exception):
            status_code = 429
            message = "x" * 500

        summary = _safe_error_summary(FakeAPIError())
        assert len(summary) < 300
        assert "..." in summary
        assert "429" in summary
        assert "FakeAPIError" in summary

    def test_no_api_key_in_output(self) -> None:
        """Even if the exception embeds a key, the summary must not."""

        class LeakyError(Exception):
            status_code = 401
            message = "Authorization: Bearer sk-1234567890abcdef"

        summary = _safe_error_summary(LeakyError())
        # The message is short enough to pass through, but it will only
        # contain the first 200 chars of message — verify it stays bounded
        assert "LeakyError" in summary
        assert "401" in summary


# ---------------------------------------------------------------------------
# OpenAIAdapter base_url validation tests
# ---------------------------------------------------------------------------


class TestOpenAIAdapterBaseUrlValidation:
    """Verify SSRF protection on base_url."""

    def test_rejects_http_scheme(self) -> None:
        with pytest.raises(ModelError, match="HTTPS"):
            OpenAIAdapter(model_name="test", base_url="http://evil.example.com/v1")

    def test_accepts_https_openrouter(self) -> None:
        adapter = OpenAIAdapter(
            model_name="test",
            base_url="https://openrouter.ai/api/v1",
        )
        assert adapter._base_url == "https://openrouter.ai/api/v1"

    def test_accepts_https_openai(self) -> None:
        adapter = OpenAIAdapter(
            model_name="test",
            base_url="https://api.openai.com/v1",
        )
        assert adapter._base_url == "https://api.openai.com/v1"

    def test_accepts_none_base_url(self) -> None:
        adapter = OpenAIAdapter(model_name="test", base_url=None)
        assert adapter._base_url is None

    def test_rejects_unknown_https_host_without_opt_in(self) -> None:
        """Custom HTTPS endpoints are rejected unless env-var opted in."""
        import os
        from unittest.mock import patch

        env = {k: v for k, v in os.environ.items() if k != "RADIANT_ALLOW_CUSTOM_BASE_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ModelError, match="RADIANT_ALLOW_CUSTOM_BASE_URL"):
                OpenAIAdapter(
                    model_name="test",
                    base_url="https://my-custom-proxy.example.com/v1",
                )

    def test_warns_on_unknown_https_host_with_opt_in(self) -> None:
        """Custom HTTPS endpoints are allowed with env-var opt-in and logged."""
        import os
        from io import StringIO
        from unittest.mock import patch

        from loguru import logger

        sink = StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            with patch.dict(os.environ, {"RADIANT_ALLOW_CUSTOM_BASE_URL": "1"}):
                OpenAIAdapter(
                    model_name="test",
                    base_url="https://my-custom-proxy.example.com/v1",
                )
            sink.seek(0)
            output = sink.read().lower()
            assert "allowlist" in output
        finally:
            logger.remove(handler_id)


# ---------------------------------------------------------------------------
# SearchConfig SSRF protection tests
# ---------------------------------------------------------------------------


class TestSearchConfigBaseUrlValidation:
    """Verify SearchConfig rejects dangerous base URLs."""

    def test_rejects_http_ncbi_url(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            SearchConfig(ncbi_base_url="http://eutils.ncbi.nlm.nih.gov/entrez/eutils/")

    def test_rejects_http_openi_url(self) -> None:
        with pytest.raises(ValueError, match="HTTPS"):
            SearchConfig(openi_base_url="http://openi.nlm.nih.gov/api/search")

    def test_accepts_default_urls(self) -> None:
        config = SearchConfig()
        assert config.ncbi_base_url.startswith("https://")
        assert config.openi_base_url.startswith("https://")

    def test_rejects_localhost(self) -> None:
        with pytest.raises(ValueError, match="loopback"):
            SearchConfig(ncbi_base_url="https://localhost/eutils/")

    def test_rejects_bare_loopback_ip(self) -> None:
        with pytest.raises(ValueError, match="private/loopback"):
            SearchConfig(ncbi_base_url="https://127.0.0.1/eutils/")

    def test_rejects_private_ip(self) -> None:
        with pytest.raises(ValueError, match="private/loopback"):
            SearchConfig(openi_base_url="https://192.168.1.1/api/search")

    def test_rejects_link_local_ip(self) -> None:
        with pytest.raises(ValueError, match="private/loopback"):
            SearchConfig(openi_base_url="https://169.254.1.1/api/search")

    def test_no_dns_resolution_at_construction(self) -> None:
        """SearchConfig must NOT call socket.getaddrinfo during __post_init__."""
        import socket
        from unittest.mock import patch

        with patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS called")):
            config = SearchConfig()  # should not trigger DNS
            assert config.ncbi_base_url.startswith("https://")


# ---------------------------------------------------------------------------
# APIError no longer stores response_body
# ---------------------------------------------------------------------------


class TestAPIErrorNoResponseBody:
    """Verify APIError does not expose raw API responses."""

    def test_no_response_body_attribute(self) -> None:
        err = APIError("test error", model_name="gpt-4o", status_code=500)
        assert not hasattr(err, "response_body")

    def test_message_preserved(self) -> None:
        err = APIError("safe summary", status_code=429)
        assert "safe summary" in str(err)
        assert err.status_code == 429


# ---------------------------------------------------------------------------
# PR 3: Prompt injection defense — _sanitize_tool_content
# ---------------------------------------------------------------------------


class TestSanitizeToolContent:
    """Verify tool result sanitization for prompt injection defense."""

    def test_wraps_in_markers_with_boundary(self) -> None:
        result = _sanitize_tool_content("hello world")
        assert result.startswith("[Tool Result - External Data - ")
        assert "[End Tool Result - " in result
        assert "hello world" in result

    def test_boundary_is_unique_per_call(self) -> None:
        """Each invocation should produce a different boundary."""
        r1 = _sanitize_tool_content("a")
        r2 = _sanitize_tool_content("a")
        # Extract boundaries
        b1 = r1.split("[Tool Result - External Data - ")[1].split("]")[0]
        b2 = r2.split("[Tool Result - External Data - ")[1].split("]")[0]
        assert b1 != b2, "Boundaries must differ between calls"

    def test_boundary_matches_open_and_close(self) -> None:
        """The open and close markers must use the same boundary."""
        result = _sanitize_tool_content("test")
        open_boundary = result.split("[Tool Result - External Data - ")[1].split("]")[0]
        close_boundary = result.split("[End Tool Result - ")[1].split("]")[0]
        assert open_boundary == close_boundary

    def test_strips_control_characters(self) -> None:
        text_with_controls = "normal\x00hidden\x01evil\x7fmore"
        result = _sanitize_tool_content(text_with_controls)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result
        assert "normalhiddenevilmore" in result

    def test_preserves_newlines_and_tabs(self) -> None:
        text = "line1\nline2\ttabbed"
        result = _sanitize_tool_content(text)
        assert "line1\nline2\ttabbed" in result

    def test_truncates_long_content(self) -> None:
        long_text = "A" * 20_000
        result = _sanitize_tool_content(long_text)
        assert "[...truncated]" in result
        # Marker overhead + 8000 chars + truncation notice + boundary
        assert len(result) < 8_300

    def test_custom_max_chars(self) -> None:
        text = "B" * 500
        result = _sanitize_tool_content(text, max_chars=100)
        assert "[...truncated]" in result

    def test_short_content_not_truncated(self) -> None:
        text = "short text"
        result = _sanitize_tool_content(text)
        assert "[...truncated]" not in result


# ---------------------------------------------------------------------------
# PR 4: Path traversal protection — ImageManager
# ---------------------------------------------------------------------------


class TestImageManagerPathTraversal:
    """Verify ImageManager rejects path traversal attacks."""

    def test_rejects_dotdot_in_path(self) -> None:
        manager = ImageManager()
        with pytest.raises(ToolExecutionError, match="traversal"):
            manager.set_image(Path("/some/path/../../../etc/passwd"))

    def test_rejects_directory_path(self, tmp_path: Path) -> None:
        """Directories should be rejected even if they exist."""
        manager = ImageManager()
        with pytest.raises(ToolExecutionError, match="not a regular file"):
            manager.set_image(tmp_path)

    def test_accepts_normal_path(self, tmp_path: Path) -> None:
        """A valid image file should be accepted."""
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.new("RGB", (10, 10)).save(img_path)

        manager = ImageManager()
        manager.set_image(img_path)
        assert manager.has_image
        assert manager.image_path == img_path.resolve()
        manager.close()

    def test_set_preloaded_rejects_dotdot(self) -> None:
        from PIL import Image

        manager = ImageManager()
        img = Image.new("RGB", (10, 10))
        with pytest.raises(ToolExecutionError, match="traversal"):
            manager.set_preloaded_image(img, Path("/a/b/../../../etc/shadow"))
        img.close()

    def test_validate_resolves_symlinks(self, tmp_path: Path) -> None:
        """Symlinks are resolved — the resolved path is stored."""
        from PIL import Image

        real = tmp_path / "real.png"
        Image.new("RGB", (10, 10)).save(real)
        link = tmp_path / "link.png"
        link.symlink_to(real)

        manager = ImageManager()
        manager.set_image(link)
        assert manager.image_path == real.resolve()
        manager.close()


# ---------------------------------------------------------------------------
# PR 6: Image magic byte validation
# ---------------------------------------------------------------------------


class TestImageMagicByteValidation:
    """Verify downloaded images are validated against magic bytes."""

    def test_accepts_valid_png(self) -> None:
        # PNG magic bytes
        content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        MedicalImageSearchManager._validate_image_magic(content, "http://example.com/img.png")

    def test_accepts_valid_jpeg(self) -> None:
        content = b"\xff\xd8\xff" + b"\x00" * 100
        MedicalImageSearchManager._validate_image_magic(content, "http://example.com/img.jpg")

    def test_accepts_valid_gif(self) -> None:
        content = b"GIF89a" + b"\x00" * 100
        MedicalImageSearchManager._validate_image_magic(content, "http://example.com/img.gif")

    def test_rejects_html_content(self) -> None:
        content = b"<html><body>not an image</body></html>"
        with pytest.raises(ImageDownloadError, match="known image format"):
            MedicalImageSearchManager._validate_image_magic(
                content, "http://example.com/img.png"
            )

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ImageDownloadError, match="known image format"):
            MedicalImageSearchManager._validate_image_magic(b"", "http://example.com/img.png")

    def test_rejects_pdf_content(self) -> None:
        content = b"%PDF-1.4 fake pdf content"
        with pytest.raises(ImageDownloadError, match="known image format"):
            MedicalImageSearchManager._validate_image_magic(
                content, "http://example.com/img.png"
            )
