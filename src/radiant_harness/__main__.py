"""CLI entry point displaying library info and usage."""

from __future__ import annotations

import sys


def main() -> int:
    """Display library information and usage instructions."""
    from radiant_harness import __version__

    print(f"Radiant Harness v{__version__}")
    print()
    print("A framework for building agentic VLM systems for medical image analysis.")
    print()
    print("This package is a library - import it in your code:")
    print()
    print("    from radiant_harness import AgenticProcessorBase, ToolRegistry")
    print()
    print("To run the NOVA benchmark example:")
    print()
    print("    cd examples/nova")
    print("    python -m src.cli --task localization --model openai/gpt-4o")
    print()
    print("For more information, see the README.md or CLAUDE.md files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
