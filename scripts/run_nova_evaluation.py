#!/usr/bin/env python3
"""NOVA Dataset Evaluation Runner

Simple script to run NOVA dataset evaluation with specified configuration.
Performs unified multi-task analysis (captioning + diagnosis + localization) in one pass.

Usage:
    python scripts/run_nova_evaluation.py --config config/baseline.yaml
    python scripts/run_nova_evaluation.py --config config/agentic.yaml --output-dir ./runs/experiment_1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger


@beartype
def run_evaluation(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run NOVA evaluation with specified configuration.

    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results

    Returns:
        Dictionary with evaluation results
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build CLI command
    cmd = [
        "uv", "run", "python", "-m", "nova_retrieval_vlm.cli",
        "--config-path", str(config_path.parent),
        "--config-name", config_path.stem,
        f"paths.output_dir={output_dir}",
    ]

    logger.info(f"🚀 Running NOVA evaluation:")
    logger.info(f"   Config: {config_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Command: {' '.join(cmd)}")

    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=3600 * 12,  # 12 hour timeout
        )

        if result.returncode == 0:
            logger.info("✅ Evaluation completed successfully!")
            return {
                "status": "success",
                "config": str(config_path),
                "output_dir": str(output_dir),
                "stdout": result.stdout,
            }
        else:
            logger.error(f"❌ Evaluation failed:")
            logger.error(f"STDERR: {result.stderr}")
            return {
                "status": "failed",
                "config": str(config_path),
                "error": result.stderr,
                "stdout": result.stdout,
            }

    except subprocess.TimeoutExpired:
        logger.error("❌ Evaluation timed out after 12 hours")
        return {
            "status": "timeout",
            "config": str(config_path),
            "error": "Evaluation timed out after 12 hours",
        }
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return {
            "status": "error",
            "config": str(config_path),
            "error": str(e),
        }


@beartype
def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run NOVA dataset evaluation with configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Configuration file path (YAML)",
    )

    parser.add_argument(
        "--output-dir",
        default="./runs/nova_evaluation",
        help="Output directory for results",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    logger.info("🔬 NOVA Dataset Evaluation Runner")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output_dir}")

    try:
        result = run_evaluation(args.config, args.output_dir)

        if result["status"] == "success":
            logger.info("🎉 Evaluation completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Evaluation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()