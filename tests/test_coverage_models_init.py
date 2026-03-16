"""Coverage tests for models/__init__.py lazy __getattr__ (lines 42-48)."""

from __future__ import annotations

import pytest


class TestLazyImports:
    def test_huggingface_adapter_lazy_import(self) -> None:
        """Accessing HuggingFaceAdapter triggers lazy import from huggingface_adapter."""
        from radiant_harness import models

        adapter_cls = models.HuggingFaceAdapter
        assert adapter_cls.__name__ == "HuggingFaceAdapter"
        assert hasattr(adapter_cls, "generate_chat")

    def test_huggingface_vlm_adapter_lazy_import(self) -> None:
        """Accessing HuggingFaceVLMAdapter triggers lazy import."""
        from radiant_harness import models

        adapter_cls = models.HuggingFaceVLMAdapter
        assert adapter_cls.__name__ == "HuggingFaceVLMAdapter"
        assert hasattr(adapter_cls, "generate_chat")

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """Accessing a non-existent attribute raises AttributeError."""
        from radiant_harness import models

        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            _ = models.NonExistent

    def test_lazy_imports_return_same_class(self) -> None:
        """Multiple accesses return the same class object."""
        from radiant_harness import models

        cls1 = models.HuggingFaceAdapter
        cls2 = models.HuggingFaceAdapter
        assert cls1 is cls2
