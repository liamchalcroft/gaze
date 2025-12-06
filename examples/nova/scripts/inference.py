#!/usr/bin/env python3
"""NOVA Inference Script

Generates per-subject predictions from NOVA dataset using specified configuration.
Outputs structured results with captions, diagnoses, localizations, and metadata.

Usage:
    python scripts/inference.py --config config/baseline.yaml --output-dir ./results/baseline
    python scripts/inference.py --config config/agentic.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from beartype import beartype
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@beartype
def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


@beartype
def generate_output_dir(config_path: str | Path, config: dict[str, Any]) -> Path:
    """Generate output directory name based on config filename and model."""
    config_name = Path(config_path).stem
    model_name = config.get("model", {}).get("name", "unknown_model")
    model_name = model_name.replace("/", "_").replace(":", "_").replace("-", "_")

    output_dir = Path("./results") / f"{config_name}_{model_name}"
    return output_dir


@beartype
def save_per_subject_results(
    subject_idx: int, subject_data: dict[str, Any], output_dir: Path
) -> None:
    """Save results for a single subject as structured JSON."""
    # Create subject directory
    subject_dir = output_dir / f"subject_{subject_idx:04d}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Save full response data
    results_file = subject_dir / "predictions.json"
    with open(results_file, "w") as f:
        json.dump(subject_data, f, indent=2)

    # Save summary for quick access
    summary = {
        "subject_id": subject_idx,
        "caption": subject_data.get("caption", {}),
        "diagnosis": subject_data.get("diagnosis", {}),
        "localization": subject_data.get("localization", {}),
        "metadata": subject_data.get("metadata", {}),
        "confidence": subject_data.get("confidence", 0.0),
        "processing_info": {
            "model": subject_data.get("model"),
            "task": subject_data.get("task"),
            "has_caption": bool(subject_data.get("caption")),
            "has_diagnosis": bool(subject_data.get("diagnosis")),
            "has_localization": bool(subject_data.get("localization", {}).get("localizations")),
        },
    }

    summary_file = subject_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved results for subject {subject_idx:04d} to {subject_dir}")


@beartype
def run_inference(config_path: str | Path, output_dir: str | Path | None = None) -> None:
    """Run inference on NOVA dataset and save per-subject results."""
    config_path = Path(config_path)
    config = load_config(config_path)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_output_dir(config_path, config)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 NOVA Inference Runner")
    logger.info(f"Config: {config_path}")
    logger.info(f"Model: {config.get('model', {}).get('name', 'unknown')}")
    logger.info(f"Output: {output_dir}")

    # Build Hydra overrides
    overrides = [
        f"paths.output_dir={output_dir}",
        f"paths.data_dir={config.get('paths', {}).get('data_dir', './data/nova')}",
        f"model.name={config.get('model', {}).get('name')}",
        f"model.max_tokens={config.get('model', {}).get('max_tokens', 4096)}",
        f"model.temperature={config.get('model', {}).get('temperature', 0.7)}",
        f"batch_size={config.get('batch_size', 4)}",
        f"max_iterations={config.get('max_iterations', 50)}",
        f"request_delay={config.get('request_delay', 3.0)}",
        f"skip_existing={str(config.get('skip_existing', False)).lower()}",
    ]

    # Add visualization overrides
    if "visualization" in config:
        viz_config = config["visualization"]
        for key, value in viz_config.items():
            if key == "out_dir" and value is None:
                continue  # Skip null out_dir
            overrides.append(f"visualization.{key}={value}")

    # Add model-specific overrides
    for key, value in config.get("model", {}).items():
        if key not in ["name", "max_tokens", "temperature"]:
            overrides.append(f"model.{key}={value}")

    # Add agentic overrides
    if "agentic" in config:
        for key, value in config["agentic"].items():
            if isinstance(value, bool):
                overrides.append(f"agentic.{key}={str(value).lower()}")
            else:
                overrides.append(f"agentic.{key}={value}")

    # Import and run the CLI
    try:
        from src.cli import main as cli_main

        logger.info(f"Running with overrides: {overrides}")

        # Set up sys.argv for CLI
        original_argv = sys.argv.copy()
        sys.argv = ["inference.py"] + overrides

        # Run the CLI inference
        cli_main()

        # Restore original argv
        sys.argv = original_argv

        logger.info("✅ Inference completed successfully!")
        logger.info(f"📁 Per-subject results saved to: {output_dir / 'per_subject'}")

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise


@beartype
def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run NOVA inference and generate per-subject predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Configuration file path (YAML)",
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for results (auto-generated if not specified)",
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

    try:
        run_inference(args.config, args.output_dir)
    except KeyboardInterrupt:
        logger.info("🛑 Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
