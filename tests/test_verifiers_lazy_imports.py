"""Tests for lazy imports in gaze.verifiers subpackage.

Verifies that reward functions can be imported without pulling in the heavy
``verifiers`` and ``datasets`` optional dependencies, and that the lazy
``__getattr__`` correctly defers imports of adapter/base/mixin symbols.
"""

from __future__ import annotations

import pytest


class TestRewardImportsAreLightweight:
    """Reward symbols must be importable without verifiers/datasets."""

    def test_rewards_import_with_verifiers_and_datasets_unavailable(self) -> None:
        """The documented lazy-import property, checked with both packages blocked.

        The other tests here run in an environment where ``verifiers`` happens to
        be installed, so they cannot detect an accidental eager import. This one
        blocks both modules in a subprocess and imports the reward symbols.
        """
        import subprocess
        import sys
        import textwrap

        program = textwrap.dedent(
            """
            import sys

            class _Blocker:
                def find_module(self, name, path=None):
                    return self.find_spec(name, path)

                def find_spec(self, name, path=None, target=None):
                    root = name.split(".")[0]
                    if root in {"verifiers", "datasets"}:
                        raise ImportError(f"{root} is blocked for this test")
                    return None

            sys.meta_path.insert(0, _Blocker())
            from gaze.verifiers import BaseRewardFunction, IoUReward, TokenF1Reward

            assert IoUReward is not None and TokenF1Reward is not None
            assert BaseRewardFunction is not None
            assert "verifiers" not in sys.modules
            assert "datasets" not in sys.modules
            print("ok")
            """
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no external input
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (
            f"reward imports pulled in an optional dependency:\n{result.stderr}"
        )
        assert "ok" in result.stdout

    def test_reward_classes_importable(self) -> None:
        from gaze.verifiers import BaseRewardFunction
        from gaze.verifiers import CombinedReward
        from gaze.verifiers import ExactMatchReward
        from gaze.verifiers import IoUReward
        from gaze.verifiers import TokenF1Reward
        from gaze.verifiers import extract_completion_text

        for sym in (
            BaseRewardFunction,
            CombinedReward,
            ExactMatchReward,
            IoUReward,
            TokenF1Reward,
            extract_completion_text,
        ):
            assert callable(sym), f"{sym.__name__} should be callable"

    def test_rewards_available_in_all(self) -> None:
        import gaze.verifiers as pkg

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
        import gaze.verifiers as pkg

        with pytest.raises(AttributeError, match="no attribute"):
            _ = pkg.NoSuchSymbol  # type: ignore[attr-defined]

    def test_heavy_symbols_listed_in_all(self) -> None:
        import gaze.verifiers as pkg

        for name in ("GazeAdapter", "BaseMultiTurnEnv", "VerifiableProcessorMixin"):
            assert name in pkg.__all__

    def test_getattr_resolves_adapter(self) -> None:
        """GazeAdapter is lazily imported via __getattr__."""
        import gaze.verifiers as pkg

        # This will trigger `from .adapter import GazeAdapter` via __getattr__
        # which in turn does `import verifiers` — only works if verifiers is installed.
        try:
            cls = pkg.GazeAdapter
            assert cls.__name__ == "GazeAdapter"
        except ImportError:
            pytest.skip("verifiers package not installed")

    def test_getattr_resolves_base(self) -> None:
        """BaseMultiTurnEnv is lazily imported via __getattr__."""
        import gaze.verifiers as pkg

        try:
            cls = pkg.BaseMultiTurnEnv
            assert cls.__name__ == "BaseMultiTurnEnv"
        except ImportError:
            pytest.skip("verifiers package not installed")

    def test_getattr_resolves_mixin(self) -> None:
        """VerifiableProcessorMixin is lazily imported via __getattr__."""
        import gaze.verifiers as pkg

        try:
            cls = pkg.VerifiableProcessorMixin
            assert cls.__name__ == "VerifiableProcessorMixin"
        except ImportError:
            pytest.skip("verifiers package not installed")
