#!/usr/bin/env python3
"""Helper utility for configuring and running parallel benchmarks."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_system_info():
    """Get system information relevant for parallel execution."""
    cpu_count = os.cpu_count()

    # Try to get memory info
    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        memory_gb = "Unknown (install psutil for memory info)"
        available_gb = "Unknown"

    print("🖥️  System Information:")
    print(f"   CPU cores: {cpu_count}")
    print(
        f"   Total memory: {memory_gb:.1f} GB"
        if isinstance(memory_gb, float)
        else f"   Total memory: {memory_gb}"
    )
    print(
        f"   Available memory: {available_gb:.1f} GB"
        if isinstance(available_gb, float)
        else f"   Available memory: {available_gb}"
    )
    print()


def estimate_runtime(approach="full", parallel_level="approaches"):
    """Estimate runtime improvements from parallelization."""
    print(f"⏱️  Runtime Estimates for {approach} benchmark:")
    print()

    # Base estimates (these are rough estimates - adjust based on your experience)
    base_times = {
        "single_model_task": 300,  # 5 minutes per model-task combination
        "num_models": 3,
        "num_tasks": 3,
        "num_approaches": 5,
    }

    total_combinations = base_times["num_models"] * base_times["num_tasks"]
    if approach == "full":
        total_combinations *= base_times["num_approaches"]

    sequential_time = total_combinations * base_times["single_model_task"]

    if parallel_level == "approaches" and approach == "full":
        # Parallel approaches
        parallel_time = sequential_time / base_times["num_approaches"]
        speedup = base_times["num_approaches"]
    elif parallel_level == "models":
        # Parallel models (conservative estimate)
        parallel_workers = min(3, base_times["num_models"])
        parallel_time = sequential_time / parallel_workers
        speedup = parallel_workers
    elif parallel_level == "both":
        # Both levels
        approach_speedup = base_times["num_approaches"] if approach == "full" else 1
        model_speedup = min(3, base_times["num_models"])
        speedup = approach_speedup * model_speedup
        parallel_time = sequential_time / speedup
    else:
        parallel_time = sequential_time
        speedup = 1

    print(f"   Sequential: {sequential_time / 3600:.1f} hours")
    print(f"   Parallel:   {parallel_time / 3600:.1f} hours")
    print(f"   Speedup:    {speedup:.1f}x faster")
    print(f"   Time saved: {(sequential_time - parallel_time) / 3600:.1f} hours")
    print()


def recommend_settings():
    """Recommend optimal parallel settings based on system."""
    cpu_count = os.cpu_count() or 1

    print("🎯 Recommended Settings:")
    print()

    if cpu_count >= 8:
        print("   High-performance system detected!")
        print("   ✅ Use full parallel benchmarks")
        print("   ✅ Max parallel approaches: 6 (all)")
        print("   ✅ Max parallel models: 3-4")
        print("   ✅ Batch size: 4-8")
    elif cpu_count >= 4:
        print("   Medium-performance system detected")
        print("   ✅ Use parallel benchmarks with limits")
        print("   ✅ Max parallel approaches: 3")
        print("   ✅ Max parallel models: 2")
        print("   ✅ Batch size: 2-4")
    else:
        print("   Lower-performance system detected")
        print("   ⚠️  Consider sequential execution or limited parallelism")
        print("   ✅ Max parallel approaches: 2")
        print("   ✅ Max parallel models: 1")
        print("   ✅ Batch size: 1-2")

    print()


def generate_commands():
    """Generate example commands for different parallelization levels."""
    print("📝 Example Commands:")
    print()

    print("1. Full parallel benchmark (all approaches in parallel):")
    print("   ./scripts/run_full_benchmarks_parallel.sh \\")
    print("     --data-dir ~/data/nova \\")
    print("     --output-dir runs/parallel_full \\")
    print("     --batch-size 4 \\")
    print("     --max-parallel 6")
    print()

    print("2. Single approach with parallel models:")
    print("   ./scripts/run_comprehensive_benchmark_parallel.sh \\")
    print("     --data-dir ~/data/nova \\")
    print("     --output-dir runs/parallel_comprehensive \\")
    print("     --batch-size 4 \\")
    print("     --max-parallel-models 3")
    print()

    print("3. Original sequential (for comparison):")
    print("   ./scripts/run_full_benchmarks.sh \\")
    print("     --data-dir ~/data/nova \\")
    print("     --output-dir runs/sequential_full \\")
    print("     --batch-size 4")
    print()


def run_benchmark(args):
    """Run the specified benchmark with parallel options."""
    script_dir = Path(__file__).parent

    if args.benchmark == "full":
        script_path = script_dir / "run_full_benchmarks_parallel.sh"
        cmd = [
            str(script_path),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--batch-size",
            str(args.batch_size),
            "--max-parallel",
            str(args.max_parallel),
        ]
    elif args.benchmark == "comprehensive":
        script_path = script_dir / "run_comprehensive_benchmark_parallel.sh"
        cmd = [
            str(script_path),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
            "--batch-size",
            str(args.batch_size),
            "--max-parallel-models",
            str(args.max_parallel_models),
        ]
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        return

    if args.max_iters > 0:
        cmd.extend(["--max-iters", str(args.max_iters)])

    print(f"🚀 Running command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Helper for parallel benchmark execution")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show system info and recommendations")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run parallel benchmarks")
    run_parser.add_argument(
        "benchmark", choices=["full", "comprehensive"], help="Which benchmark to run"
    )
    run_parser.add_argument("--data-dir", default="~/data/nova", help="Path to NOVA dataset")
    run_parser.add_argument(
        "--output-dir", default="runs/parallel_benchmark", help="Output directory"
    )
    run_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    run_parser.add_argument(
        "--max-parallel", type=int, default=6, help="Max parallel approaches (for full benchmark)"
    )
    run_parser.add_argument(
        "--max-parallel-models",
        type=int,
        default=3,
        help="Max parallel models (for single approach)",
    )
    run_parser.add_argument("--max-iters", type=int, default=-1, help="Max iterations (-1 for all)")

    args = parser.parse_args()

    if args.command == "info":
        get_system_info()
        recommend_settings()
        estimate_runtime("full", "approaches")
        estimate_runtime("comprehensive", "models")
        generate_commands()
    elif args.command == "run":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
