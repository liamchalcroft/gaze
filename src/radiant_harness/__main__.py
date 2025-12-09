"""NOVA VLM - Radiology Vision Language Model Evaluation Framework.

This is a research framework for benchmarking vision-language models
on the NOVA brain-MRI dataset using agentic multi-turn reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add examples to path for NOVA CLI
sys.path.insert(0, str(Path(__file__).parent / "examples"))


def main() -> None:
    """Main entry point for the NOVA VLM framework."""
    parser = argparse.ArgumentParser(
        description="NOVA VLM - Radiology Vision Language Model Evaluation Framework",
        epilog="Example: python -m nova_retrieval_vlm task=localization model=openai/gpt-4o",
    )
    parser.add_argument(
        "hydra_args",
        nargs="*",
        help="Hydra configuration arguments (e.g., task=localization model=openai/gpt-4o)",
    )

    args = parser.parse_args()

    # Import and run NOVA CLI
    from nova.src.cli import main as nova_main

    # Convert hydra args to sys.argv format
    sys.argv = ["nova"] + args.hydra_args

    try:
        nova_main()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
