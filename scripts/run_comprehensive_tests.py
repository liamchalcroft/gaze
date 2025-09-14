#!/usr/bin/env python3
"""Comprehensive test runner for NOVA retrieval VLM system.

This script provides different test execution modes:
- Unit tests only
- Integration tests only
- Full test suite
- Performance/stress tests
- Coverage reporting
- CI-friendly output
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for NOVA retrieval VLM system"
    )

    parser.add_argument(
        "--mode",
        choices=["unit", "integration", "full", "stress", "quick", "ci"],
        default="full",
        help="Test execution mode (default: full)",
    )

    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")

    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel (where possible)"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")

    parser.add_argument("--html-report", action="store_true", help="Generate HTML coverage report")

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["uv", "run", "pytest"]

    # Add coverage if requested
    if args.coverage:
        base_cmd.extend(["--cov=nova_retrieval_vlm", "--cov-report=term-missing"])
        if args.html_report:
            base_cmd.extend(["--cov-report=html"])

    # Add parallel execution if requested
    if args.parallel:
        base_cmd.extend(["-n", "auto"])

    # Add verbosity
    if args.verbose:
        base_cmd.append("-vv")

    # Add fail-fast
    if args.fail_fast:
        base_cmd.append("-x")

    print("🚀 NOVA Retrieval VLM Comprehensive Test Suite")
    print(f"Mode: {args.mode}")
    print(f"Coverage: {'Enabled' if args.coverage else 'Disabled'}")
    print(f"Parallel: {'Enabled' if args.parallel else 'Disabled'}")

    success = True

    if args.mode == "unit":
        # Run only unit tests
        cmd = base_cmd + ["-m", "unit", "tests/"]
        success = run_command(cmd, "Unit Tests")

    elif args.mode == "integration":
        # Run only integration tests
        cmd = base_cmd + ["-m", "integration", "tests/test_integration.py"]
        success = run_command(cmd, "Integration Tests")

    elif args.mode == "stress":
        # Run stress and performance tests
        cmd = base_cmd + ["-m", "stress or performance", "tests/test_edge_cases_and_stress.py"]
        success = run_command(cmd, "Stress and Performance Tests")

    elif args.mode == "quick":
        # Run quick subset of tests
        cmd = base_cmd + ["-m", "not slow and not stress", "tests/"]
        success = run_command(cmd, "Quick Test Suite")

    elif args.mode == "ci":
        # CI-friendly test run
        cmd = base_cmd + [
            "--tb=short",
            "--strict-markers",
            "-m",
            "not slow and not stress",
            "tests/",
        ]
        success = run_command(cmd, "CI Test Suite")

    elif args.mode == "full":
        # Run complete test suite in stages
        test_stages = [
            {
                "name": "Unit Tests",
                "cmd": base_cmd + ["tests/test_types.py", "tests/test_batch_processing_utils.py"],
                "required": True,
            },
            {
                "name": "Processor Tests",
                "cmd": base_cmd + ["tests/test_processors.py"],
                "required": True,
            },
            {
                "name": "Integration Tests",
                "cmd": base_cmd + ["tests/test_integration.py"],
                "required": True,
            },
            {
                "name": "Edge Case Tests",
                "cmd": base_cmd
                + ["-m", "not slow and not stress", "tests/test_edge_cases_and_stress.py"],
                "required": False,
            },
            {
                "name": "Legacy Tests (Fixed Imports)",
                "cmd": base_cmd + ["tests/test_retrievers.py", "tests/test_adapters.py"],
                "required": False,
            },
        ]

        for stage in test_stages:
            stage_success = run_command(stage["cmd"], stage["name"])
            if not stage_success:
                if stage["required"]:
                    success = False
                    print(f"❌ Required test stage '{stage['name']}' failed!")
                    if args.fail_fast:
                        break
                else:
                    print(f"⚠️  Optional test stage '{stage['name']}' failed, continuing...")

    # Summary
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Test suite completed successfully")
    else:
        print("💥 SOME TESTS FAILED!")
        print("❌ Check the output above for details")
    print(f"{'=' * 60}")

    # Coverage report location
    if args.coverage and args.html_report:
        html_report_path = Path("htmlcov/index.html")
        if html_report_path.exists():
            print(f"\n📊 HTML coverage report available at: {html_report_path.absolute()}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
