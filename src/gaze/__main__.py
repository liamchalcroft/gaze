"""Command-line entry point for GAZE.

GAZE is primarily a library. This CLI reports version and environment
information; analyses are run by importing the library or via the example
benchmarks in the source repository.
"""

from __future__ import annotations

import argparse
import platform
import sys

from beartype import beartype

_REPO_URL = "https://github.com/liamchalcroft/gaze"
_DOCS_URL = "https://liamchalcroft.github.io/gaze/"


@beartype
def _build_info() -> str:
    """Build the human-readable info banner shown by ``gaze`` / ``gaze info``."""
    from gaze import __version__
    from gaze.tools import create_search_tools
    from gaze.tools import create_visual_tools

    n_visual = len(create_visual_tools(set()))
    n_search = len(create_search_tools(set()))

    lines = [
        f"GAZE {__version__}",
        f"Python {platform.python_version()} ({platform.system()} {platform.machine()})",
        "",
        "Grounded agentic framework for medical vision-language models.",
        "",
        "Adapters:  OpenAIAdapter, LMStudioAdapter, HuggingFaceAdapter",
        f"Tools:     {n_visual} visual, {n_search} search",
        "",
        f"Docs:      {_DOCS_URL}",
        f"Source:    {_REPO_URL}",
        f"Examples:  {_REPO_URL}/tree/main/examples",
        "",
        "GAZE is a library. Import it in your code:",
        "    from gaze import AgenticProcessorBase, analyze",
    ]
    return "\n".join(lines)


@beartype
def main(argv: list[str] | None = None) -> int:
    """Run the GAZE CLI. Returns a process exit code."""
    from gaze import __version__

    parser = argparse.ArgumentParser(
        prog="gaze",
        description="GAZE: grounded agentic framework for medical vision-language models.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"GAZE {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="Show version, environment, and documentation links.")

    parser.parse_args(argv)

    # Both the bare invocation and the explicit ``info`` command print the banner.
    print(_build_info())
    return 0


if __name__ == "__main__":
    sys.exit(main())
