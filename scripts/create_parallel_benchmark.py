#!/usr/bin/env python3
"""Utility to generate parallel versions of existing benchmark scripts."""

import argparse
import re
from pathlib import Path


def read_existing_script(script_path: Path) -> str:
    """Read the content of an existing benchmark script."""
    try:
        return script_path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f"Script not found: {script_path}")


def create_parallel_version(script_content: str, approach_name: str) -> str:
    """Transform a sequential script to a parallel version."""

    # Template for parallel script
    parallel_template = f"""#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
source "${{SCRIPT_DIR}}/model_list.sh"

DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/{approach_name}_benchmark_parallel"
BATCH_SIZE=${{BATCH_SIZE:-4}}
MAX_ITERS=-1
MAX_PARALLEL_MODELS=3  # Number of models to run in parallel

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    --max-parallel-models) MAX_PARALLEL_MODELS="$2"; shift 2;;
    -h|--help) echo "Usage: $(basename "$0") [options]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

TASKS=(localization caption diagnosis)

mkdir -p "${{OUTPUT_DIR}}"

echo "==== PARALLEL {approach_name.upper()} BENCHMARK ===="
echo "DATA_DIR   = ${{DATA_DIR}}"
echo "OUTPUT_DIR = ${{OUTPUT_DIR}}"
echo "MODELS     = ${{MODELS_DEFAULT[*]}}"
echo "TASKS      = ${{TASKS[*]}}"
echo "APPROACH   = {approach_name}"
echo "MAX_PARALLEL = ${{MAX_PARALLEL_MODELS}} models"
echo "============================================"

# Function to run a single model-task combination
run_model_task() {{
    local MODEL="$1"
    local TASK="$2"

    RUN_DIR="${{OUTPUT_DIR}}/${{TASK}}/$(echo "${{MODEL}}" | tr '/:' '_')"
    echo "▶ {approach_name} (parallel) | ${{TASK}} | ${{MODEL}} → ${{RUN_DIR}}"

    # Create run directory and log configuration
    mkdir -p "${{RUN_DIR}}"
    echo "Configuration: approach={approach_name}, task=${{TASK}}, model=${{MODEL}}" > "${{RUN_DIR}}/config.txt"
    echo "Parallel execution enabled" >> "${{RUN_DIR}}/config.txt"

    local log_file="${{RUN_DIR}}/execution.log"

    if python -m nova_retrieval_vlm.cli \\
      task=${{TASK}} \\
      approach={approach_name} \\
      model.name="${{MODEL}}" \\
      batch_size=${{BATCH_SIZE}} \\
      max_iterations=${{MAX_ITERS}} \\
      paths.data_dir="${{DATA_DIR}}" \\
      paths.output_dir="${{RUN_DIR}}" \\
      skip_existing=true \\
      strict_mode=true \\
      > "${{log_file}}" 2>&1; then
      echo "✓ SUCCESS: ${{MODEL}} | ${{TASK}}" | tee "${{RUN_DIR}}/status.txt"
    else
      echo "✗ FAILED: ${{MODEL}} | ${{TASK}}" | tee "${{RUN_DIR}}/status.txt"
    fi
}}

# Create array of all model-task combinations
declare -a JOBS=()
for MODEL in "${{MODELS_DEFAULT[@]}}"; do
  for TASK in "${{TASKS[@]}}"; do
    JOBS+=("${{MODEL}}:${{TASK}}")
  done
done

echo "Total jobs: ${{#JOBS[@]}}"
echo "Running up to ${{MAX_PARALLEL_MODELS}} jobs in parallel..."
echo ""

# Function to run jobs in parallel with limit
run_parallel_jobs() {{
    local max_parallel="$1"
    shift
    local jobs=("$@")

    local running_jobs=0
    local job_index=0
    declare -a pids=()

    while [[ $job_index -lt ${{#jobs[@]}} ]]; do
        # Start new jobs up to the limit
        while [[ $running_jobs -lt $max_parallel && $job_index -lt ${{#jobs[@]}} ]]; do
            local job="${{jobs[$job_index]}}"
            IFS=':' read -r model task <<< "$job"

            echo "Starting job $((job_index + 1))/${{#jobs[@]}}: ${{model}} | ${{task}}"
            run_model_task "$model" "$task" &
            local pid=$!
            pids+=("$pid:$job")

            ((running_jobs++))
            ((job_index++))
            sleep 1  # Small delay to stagger startup
        done

        # Wait for at least one job to finish
        local finished=false
        while [[ $finished == false ]]; do
            for i in "${{!pids[@]}}"; do
                local pid_job="${{pids[$i]}}"
                IFS=':' read -r pid job <<< "$pid_job"

                if ! kill -0 "$pid" 2>/dev/null; then
                    # Job finished
                    wait "$pid"
                    local exit_code=$?

                    IFS=':' read -r model task <<< "$job"
                    if [[ $exit_code -eq 0 ]]; then
                        echo "✅ Completed: ${{model}} | ${{task}}"
                    else
                        echo "❌ Failed: ${{model}} | ${{task}}"
                    fi

                    # Remove from array
                    unset 'pids[$i]'
                    pids=("${{pids[@]}}")  # Reindex array

                    ((running_jobs--))
                    finished=true
                    break
                fi
            done

            if [[ $finished == false ]]; then
                sleep 2  # Check again in 2 seconds
            fi
        done
    done

    # Wait for remaining jobs
    for pid_job in "${{pids[@]}}"; do
        IFS=':' read -r pid job <<< "$pid_job"
        wait "$pid"
        IFS=':' read -r model task <<< "$job"
        echo "✅ Final completion: ${{model}} | ${{task}}"
    done
}}

# Run all jobs in parallel
run_parallel_jobs "$MAX_PARALLEL_MODELS" "${{JOBS[@]}}"

echo ""
echo "✅ Parallel {approach_name} benchmark finished → ${{OUTPUT_DIR}}"
echo "📊 Check individual logs in each run directory for details"
"""

    return parallel_template


def main():
    parser = argparse.ArgumentParser(description="Generate parallel versions of benchmark scripts")
    parser.add_argument("approach", help="Approach name (e.g., baseline, visual, multiturn)")
    parser.add_argument("--input-script", help="Input script to convert (optional)")
    parser.add_argument(
        "--output-dir", default="scripts", help="Output directory for generated script"
    )

    args = parser.parse_args()

    approach_name = args.approach
    script_dir = Path(args.output_dir)

    # Generate output filename
    output_filename = f"run_{approach_name}_benchmark_parallel.sh"
    output_path = script_dir / output_filename

    if args.input_script:
        # Convert existing script
        input_path = Path(args.input_script)
        script_content = read_existing_script(input_path)
        print(f"Converting existing script: {input_path}")
    else:
        # Generate from template
        script_content = ""
        print(f"Generating new parallel script for approach: {approach_name}")

    # Create parallel version
    parallel_content = create_parallel_version(script_content, approach_name)

    # Extract specific CLI parameters if available from input script
    if script_content:
        # Look for approach-specific parameters in the original script
        cli_matches = re.findall(
            r"python -m nova_retrieval_vlm\.cli.*?(?=;|$)", script_content, re.DOTALL
        )
        if cli_matches:
            # Extract parameters from the last CLI call
            cli_matches[-1]
            # This is a simplified extraction - could be enhanced
            print(
                "Found CLI parameters in original script - you may want to customize these manually"
            )

    # Write parallel script
    output_path.write_text(parallel_content)
    output_path.chmod(0o755)  # Make executable

    print(f"✅ Generated parallel script: {output_path}")
    print(f"📝 Usage: ./{output_path} --help")
    print()
    print("📋 Next steps:")
    print("1. Review the generated script and customize CLI parameters if needed")
    print("2. Test with a small subset first (--max-iters 5)")
    print("3. Monitor system resources during execution")
    print("4. Adjust --max-parallel-models based on system capacity")
    print()
    print("🔧 Example customization:")
    print("   Edit the CLI parameters in the run_model_task() function")
    print("   Add approach-specific configuration options")
    print("   Modify the parallel job management logic if needed")


if __name__ == "__main__":
    main()
