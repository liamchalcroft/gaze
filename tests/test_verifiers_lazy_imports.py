"""Tests for lazy imports in radiant_harness.verifiers subpackage.

Verifies that reward functions can be imported without pulling in the heavy
``verifiers`` and ``datasets`` optional dependencies, and that the lazy
``__getattr__`` correctly defers imports of adapter/base/mixin symbols.
"""

from __future__ import annotations

import pytest


class TestRewardImportsAreLightweight:
    """Reward symbols must be importable without verifiers/datasets."""

    def test_reward_classes_importable(self) -> None:
        from radiant_harness.verifiers import BaseRewardFunction
        from radiant_harness.verifiers import CombinedReward
        from radiant_harness.verifiers import ExactMatchReward
        from radiant_harness.verifiers import IoUReward
        from radiant_harness.verifiers import TokenF1Reward
        from radiant_harness.verifiers import extract_completion_text

        for sym in (
            BaseRewardFunction,
            CombinedReward,
            ExactMatchReward,
            IoUReward,
            TokenF1Reward,
            extract_completion_text,
        ):
            assert sym is not None

    def test_rewards_available_in_all(self) -> None:
        import radiant_harness.verifiers as pkg

        for name in (
            "BaseRewardFunction",
            "ExactMatchReward",
            "TokenF1Reward",
            "IoUReward",
            "CombinedReward",
            "extract_completion_text",
        ):
            assert name in pkg.__all__


class TestLazyGetattr:
    """The __getattr__ on the verifiers subpackage lazily resolves heavy symbols."""

    def test_unknown_attr_raises_attribute_error(self) -> None:
        import radiant_harness.verifiers as pkg

        with pytest.raises(AttributeError, match="no attribute"):
            _ = pkg.NoSuchSymbol  # type: ignore[attr-defined]

    def test_heavy_symbols_listed_in_all(self) -> None:
        import radiant_harness.verifiers as pkg

        for name in ("RadiantHarnessAdapter", "BaseMultiTurnEnv", "VerifiableProcessorMixin"):
            assert name in pkg.__all__

    def test_getattr_resolves_adapter(self) -> None:
        """RadiantHarnessAdapter is lazily imported via __getattr__."""
        import radiant_harness.verifiers as pkg

        # This will trigger `from .adapter import RadiantHarnessAdapter` via __getattr__
        # which in turn does `import verifiers` — only works if verifiers is installed.
        try:
            cls = pkg.RadiantHarnessAdapter
            assert cls.__name__ == "RadiantHarnessAdapter"
        except ImportError:
            pytest.skip("verifiers package not installed")

    def test_getattr_resolves_base(self) -> None:
        """BaseMultiTurnEnv is lazily imported via __getattr__."""
        import radiant_harness.verifiers as pkg

        try:
            cls = pkg.BaseMultiTurnEnv
            assert cls.__name__ == "BaseMultiTurnEnv"
        except ImportError:
            pytest.skip("verifiers package not installed")

    def test_getattr_resolves_mixin(self) -> None:
        """VerifiableProcessorMixin is lazily imported via __getattr__."""
        import radiant_harness.verifiers as pkg

        try:
            cls = pkg.VerifiableProcessorMixin
            assert cls.__name__ == "VerifiableProcessorMixin"
        except ImportError:
            pytest.skip("verifiers package not installed")
