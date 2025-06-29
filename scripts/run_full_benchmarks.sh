#!/usr/bin/env bash
# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"
set -euo pipefail

# -----------------------------------------------------------------------------
# NOVA VLM - FULL BENCHMARK SUITE (Enhanced System Prompts)
# -----------------------------------------------------------------------------
# This script evaluates **all six** optimized approaches across the three NOVA 
# tasks (localization, caption, diagnosis) on a configurable list of OpenRouter 
# models.
#
# Enhanced Approaches:
#   - baseline: Enhanced JSON format specification and clinical accuracy
#   - retrieval: Knowledge synthesis and evidence integration
#   - multiturn: Conditional continuation and systematic analysis
#   - visual: Enhanced visual operations guidance
#   - web_search: Query formulation strategies and current information
#   - comprehensive: All capabilities combined with performance optimization
#
# Usage:
#   ./scripts/run_full_benchmarks.sh \
#       [--data-dir /abs/path/to/nova] \
#       [--output-dir runs/full_benchmark] \
#       [--batch-size 4] \
#       [--max-iters -1]
#
# Environment variables:
#   MODELS      Space-separated list of model identifiers (defaults below)
#   BATCH_SIZE  Overrides --batch-size
#
# The CLI invoked is nova_retrieval_vlm.cli with enhanced system prompts.
# -----------------------------------------------------------------------------

# Defaults
DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/full_benchmark"
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITERS=-1

# Determine directory of this script so we can invoke companion scripts reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse CLI opts
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    -h|--help)
      echo "Usage: $(basename "$0") [options]"; exit 0;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done


echo "🚀 Starting full benchmark suite with all enhanced approaches..."
echo "📊 This will run 6 approaches × 3 tasks × ${#MODELS_DEFAULT[@]} models = $((6 * 3 * ${#MODELS_DEFAULT[@]})) total runs"
echo ""

# Run all enhanced benchmarks
echo "1/6 Running baseline benchmark (enhanced JSON format)..."
${SCRIPT_DIR}/run_baseline_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/baseline" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

echo -e "\n2/6 Running retrieval benchmark (knowledge synthesis)..."
${SCRIPT_DIR}/run_retrieval_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/retrieval" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

echo -e "\n3/6 Running multiturn benchmark (conditional continuation)..."
${SCRIPT_DIR}/run_multiturn_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/multiturn" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

echo -e "\n4/6 Running visual benchmark (enhanced guidance)..."
${SCRIPT_DIR}/run_visual_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/visual" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

echo -e "\n5/6 Running web search benchmark (query formulation)..."
${SCRIPT_DIR}/run_web_search_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/web_search" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

echo -e "\n6/6 Running comprehensive benchmark (all capabilities)..."
${SCRIPT_DIR}/run_comprehensive_benchmark.sh \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/comprehensive" \
  --batch-size "${BATCH_SIZE}" \
  --max-iters "${MAX_ITERS}"

# Generate summary report
echo -e "\n📋 Generating benchmark summary report..."
SUMMARY_FILE="${OUTPUT_DIR}/benchmark_summary.txt"
{
  echo "NOVA VLM Enhanced Benchmark Suite Summary"
  echo "========================================"
  echo "Generated: $(date)"
  echo "Data Directory: ${DATA_DIR}"
  echo "Output Directory: ${OUTPUT_DIR}"
  echo "Batch Size: ${BATCH_SIZE}"
  echo "Max Iterations: ${MAX_ITERS}"
  echo ""
  echo "Enhanced Approaches Tested:"
  echo "- baseline: Enhanced JSON format specification"
  echo "- retrieval: Knowledge synthesis and evidence integration"  
  echo "- multiturn: Conditional continuation and systematic analysis"
  echo "- visual: Enhanced visual operations guidance"
  echo "- web_search: Query formulation strategies"
  echo "- comprehensive: All capabilities combined"
  echo ""
  echo "Tasks: localization, caption, diagnosis"
  echo "Models: ${MODELS_DEFAULT[*]}"
  echo ""
  echo "Results Structure:"
  for approach in baseline retrieval multiturn visual web_search comprehensive; do
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

echo "\n✅ Full enhanced benchmark suite completed!"
echo "📊 Results consolidated in: ${OUTPUT_DIR}"
echo "📋 Summary report: ${SUMMARY_FILE}"
echo "🎯 All six optimized approaches have been evaluated for maximum performance"
