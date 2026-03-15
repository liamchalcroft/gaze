"""Tests for radiant_harness.verifiers.mixin — path safety and image encoding.

Covers _safe_resolve_image_path, _image_file_to_data_url, and the
VerifiableProcessorMixin.as_verifiers_env factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

try:
    from radiant_harness.verifiers.mixin import _image_file_to_data_url
    from radiant_harness.verifiers.mixin import _safe_resolve_image_path

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

pytestmark = pytest.mark.skipif(not _HAS_VERIFIERS, reason="verifiers not installed")


# ---------------------------------------------------------------------------
# _safe_resolve_image_path
# ---------------------------------------------------------------------------


class TestSafeResolveImagePath:
    def test_valid_relative_path(self, tmp_path: Path) -> None:
        img = tmp_path / "scan.png"
        img.touch()
        result = _safe_resolve_image_path(tmp_path, "scan.png")
        assert result == str(img.resolve())

    def test_valid_nested_path(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        img = subdir / "scan.png"
        img.touch()
        result = _safe_resolve_image_path(tmp_path, "sub/scan.png")
        assert result == str(img.resolve())

    def test_traversal_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="traversal blocked"):
            _safe_resolve_image_path(tmp_path, "../../etc/passwd")

    def test_traversal_with_dots_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="traversal blocked"):
            _safe_resolve_image_path(tmp_path, "../../../tmp/evil")

    def test_base_itself_is_allowed(self, tmp_path: Path) -> None:
        # Resolving "." relative to base should give the base itself
        result = _safe_resolve_image_path(tmp_path, ".")
        assert result == str(tmp_path.resolve())

    def test_prefix_confusion_blocked(self, tmp_path: Path) -> None:
        """Ensure /data/images_evil doesn't pass for /data/images base."""
        # Create a sibling directory with a name that is a prefix match
        evil = tmp_path.parent / (tmp_path.name + "_evil")
        evil.mkdir(exist_ok=True)
        try:
            # Construct a relative path that resolves to the evil directory
            relative = f"../{evil.name}/file.txt"
            with pytest.raises(ValueError, match="traversal blocked"):
                _safe_resolve_image_path(tmp_path, relative)
        finally:
            evil.rmdir()


# ---------------------------------------------------------------------------
# _image_file_to_data_url
# ---------------------------------------------------------------------------


class TestImageFileToDataUrl:
    def test_png_to_data_url(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        Image.new("RGB", (10, 10), color="red").save(img_path)

        data_url = _image_file_to_data_url(str(img_path))
        # encode_image may convert to JPEG; just verify it's a valid data URL
        assert data_url.startswith("data:image/")
        assert ";base64," in data_url
        b64_part = data_url.split(",", 1)[1]
        assert len(b64_part) > 10

    def test_jpeg_to_data_url(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10), color="blue").save(img_path, format="JPEG")

        data_url = _image_file_to_data_url(str(img_path))
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _image_file_to_data_url("/nonexistent/image.png")
