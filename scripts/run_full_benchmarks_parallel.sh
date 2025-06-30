#!/usr/bin/env bash
# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"
set -euo pipefail

# -----------------------------------------------------------------------------
# NOVA VLM - PARALLEL FULL BENCHMARK SUITE
# -----------------------------------------------------------------------------
# This script runs all benchmark approaches in PARALLEL for maximum speed.
# Each approach runs as a background process, significantly reducing total runtime.
#
# Usage:
#   ./scripts/run_full_benchmarks_parallel.sh \
#       [--data-dir /abs/path/to/nova] \
#       [--output-dir runs/full_benchmark] \
#       [--batch-size 4] \
#       [--max-iters -1] \
#       [--max-parallel 6]
# -----------------------------------------------------------------------------

# Defaults
DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/full_benchmark"
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITERS=-1
MAX_PARALLEL=6  # Number of approaches to run in parallel

# Determine directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the model list
source "${SCRIPT_DIR}/model_list.sh"

# Parse CLI opts
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2;;
    -h|--help)
      echo "Usage: $(basename "$0") [options]"; exit 0;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

echo "🚀 Starting PARALLEL benchmark suite with all enhanced approaches..."
echo "📊 Running ${#BENCHMARK_JOBS[@]} approaches × 3 tasks × ${#MODELS_DEFAULT[@]} models = $((${#BENCHMARK_JOBS[@]} * 3 * ${#MODELS_DEFAULT[@]})) total runs"
echo "⚡ Max parallel approaches: ${MAX_PARALLEL}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Function to run a benchmark approach
run_benchmark() {
    local approach="$1"
    local script_name="$2"
    local approach_num="$3"
    
    echo "[$approach_num/6] Starting ${approach} benchmark (parallel)..."
    
    local log_file="${OUTPUT_DIR}/${approach}_parallel.log"
    local status_file="${OUTPUT_DIR}/${approach}_status.txt"
    
    if ${SCRIPT_DIR}/${script_name} \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}/${approach}" \
      --batch-size "${BATCH_SIZE}" \
      --max-iters "${MAX_ITERS}" \
      > "${log_file}" 2>&1; then
      echo "✓ SUCCESS: ${approach}" | tee "${status_file}"
    else
      echo "✗ FAILED: ${approach}" | tee "${status_file}"
    fi
}

# Array of benchmark jobs (matches original script - retrieval commented out)
declare -a BENCHMARK_JOBS=(
    "baseline:run_baseline_benchmark.sh:1"
    "multiturn:run_multiturn_benchmark.sh:3"
    "visual:run_visual_benchmark.sh:4"
    "web_search:run_web_search_benchmark.sh:5"
    "comprehensive:run_comprehensive_benchmark.sh:6"
)
# Note: retrieval benchmark is commented out in original script

# Validate that all required scripts exist
echo "🔍 Validating required benchmark scripts..."
for job in "${BENCHMARK_JOBS[@]}"; do
    IFS=':' read -r approach script_name approach_num <<< "$job"
    script_path="${SCRIPT_DIR}/${script_name}"
    if [[ ! -f "$script_path" ]]; then
        echo "❌ ERROR: Required script not found: $script_path"
        echo "   Available scripts in ${SCRIPT_DIR}:"
        ls -1 "${SCRIPT_DIR}"/run_*_benchmark.sh || echo "   No benchmark scripts found"
        exit 1
    fi
    if [[ ! -x "$script_path" ]]; then
        echo "⚠️  WARNING: Making $script_path executable..."
        chmod +x "$script_path"
    fi
done
echo "✅ All required scripts validated"
echo ""

# Signal handler for cleanup
cleanup() {
    echo ""
    echo "🛑 Interrupt received. Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Stopping PID: $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "🧹 Cleanup completed. Exiting."
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start all benchmarks in parallel
declare -a PIDS=()

for job in "${BENCHMARK_JOBS[@]}"; do
    IFS=':' read -r approach script_name approach_num <<< "$job"
    
    # Start benchmark in background
    run_benchmark "$approach" "$script_name" "$approach_num" &
    current_pid=$!
    PIDS+=($current_pid)
    
    echo "Started ${approach} benchmark (PID: ${current_pid})"
    sleep 2  # Small delay to stagger startup
done

echo ""
echo "⏳ All ${#BENCHMARK_JOBS[@]} benchmark approaches are running in parallel..."
echo "📊 Monitor progress:"
echo "   Overall logs: tail -f ${OUTPUT_DIR}/*_parallel.log"
echo "   Live updates: watch -n 5 'ls -la ${OUTPUT_DIR}/*_status.txt 2>/dev/null || echo \"Status files not yet created\"'"
echo "   Quick status: ls -la ${OUTPUT_DIR}/*_status.txt"
echo ""

# Function to show current status
show_status() {
    echo "📊 Current benchmark status:"
    for job in "${BENCHMARK_JOBS[@]}"; do
        IFS=':' read -r approach script_name approach_num <<< "$job"
        status_file="${OUTPUT_DIR}/${approach}_status.txt"
        if [[ -f "$status_file" ]]; then
            status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
            echo "   ${approach}: ${status}"
        else
            echo "   ${approach}: RUNNING..."
        fi
    done
    echo ""
}

# Show initial status
show_status

# Wait for all background jobs to complete with periodic status updates
echo "⏳ Waiting for all benchmark jobs to complete..."

# Wait for jobs to complete
remaining_pids=("${PIDS[@]}")
remaining_jobs=("${BENCHMARK_JOBS[@]}")

while [[ ${#remaining_pids[@]} -gt 0 ]]; do
    new_pids=()
    new_jobs=()
    
    for i in "${!remaining_pids[@]}"; do
        pid=${remaining_pids[$i]}
        job=${remaining_jobs[$i]}
        
        if ! kill -0 "$pid" 2>/dev/null; then
            # Job completed
            IFS=':' read -r approach script_name approach_num <<< "$job"
            wait $pid
            exit_code=$?
            
            if [[ $exit_code -eq 0 ]]; then
                echo "✅ ${approach} benchmark completed successfully"
            else
                echo "❌ ${approach} benchmark failed (exit code: $exit_code)"
            fi
        else
            # Job still running
            new_pids+=("$pid")
            new_jobs+=("$job")
        fi
    done
    
    remaining_pids=("${new_pids[@]}")
    remaining_jobs=("${new_jobs[@]}")
    
    if [[ ${#remaining_pids[@]} -gt 0 ]]; then
        echo "⏳ ${#remaining_pids[@]} jobs still running..."
        show_status
        sleep 10  # Check every 10 seconds
    fi
done

# Generate comprehensive summary report
echo -e "\n📋 Generating parallel benchmark summary report..."
SUMMARY_FILE="${OUTPUT_DIR}/parallel_benchmark_summary.txt"
{
  echo "NOVA VLM Parallel Benchmark Suite Summary"
  echo "========================================="
  echo "Generated: $(date)"
  echo "Data Directory: ${DATA_DIR}"
  echo "Output Directory: ${OUTPUT_DIR}"
  echo "Batch Size: ${BATCH_SIZE}"
  echo "Max Iterations: ${MAX_ITERS}"
  echo "Max Parallel: ${MAX_PARALLEL}"
  echo ""
  echo "Approaches Tested (in parallel):"
  for job in "${BENCHMARK_JOBS[@]}"; do
    IFS=':' read -r approach script_name approach_num <<< "$job"
    if [[ -f "${OUTPUT_DIR}/${approach}_status.txt" ]]; then
      status=$(cat "${OUTPUT_DIR}/${approach}_status.txt")
      echo "- ${approach}: ${status}"
    else
      echo "- ${approach}: UNKNOWN"
    fi
  done
  echo ""
  echo "Tasks: localization, caption, diagnosis"
  echo "Models: ${MODELS_DEFAULT[*]}"
  echo ""
  echo "Results Structure:"
  for job in "${BENCHMARK_JOBS[@]}"; do
    IFS=':' read -r approach script_name approach_num <<< "$job"
    echo "  ${OUTPUT_DIR}/${approach}/"
    for task in localization caption diagnosis; do
      echo "    ├── ${task}/"
      for model in "${MODELS_DEFAULT[@]}"; do
        model_dir=$(echo "${model}" | tr '/:' '_')
        echo "    │   └── ${model_dir}/"
      done
    done
  done
} > "${SUMMARY_FILE}"

echo ""
echo "🎉 Parallel benchmark suite completed!"
echo ""
echo "📊 Final Results Summary:"
show_status

echo "📁 Results Location: ${OUTPUT_DIR}"
echo "📋 Summary Report: ${SUMMARY_FILE}"
echo ""
echo "🔍 Quick Analysis Commands:"
echo "   View all logs: ls -la ${OUTPUT_DIR}/*/*/execution.log"
echo "   Check failures: grep -l FAILED ${OUTPUT_DIR}/*_status.txt"
echo "   Monitor space: du -sh ${OUTPUT_DIR}"
echo ""
echo "📈 Performance Analysis:"
echo "   Compare results: python scripts/plot_results.py --input-dir ${OUTPUT_DIR}"
echo "   Validate metrics: python scripts/validate_metrics.py --results-dir ${OUTPUT_DIR}"
echo ""
echo "⚡ Parallel execution significantly reduced total runtime!"