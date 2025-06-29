#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/model_list.sh"

DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/comprehensive_benchmark"
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITERS=-1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    -h|--help) echo "Usage: $(basename "$0") [options]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

TASKS=(localization caption diagnosis)

mkdir -p "${OUTPUT_DIR}"

echo "==== COMPREHENSIVE BENCHMARK (Enhanced System Prompts) ===="
echo "DATA_DIR   = ${DATA_DIR}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "MODELS     = ${MODELS_DEFAULT[*]}"
echo "TASKS      = ${TASKS[*]}"
echo "APPROACH   = comprehensive (all capabilities combined)"
echo "==========================================================="

for MODEL in "${MODELS_DEFAULT[@]}"; do
  for TASK in "${TASKS[@]}"; do
    RUN_DIR="${OUTPUT_DIR}/${TASK}/$(echo "${MODEL}" | tr '/:' '_')"
    echo -e "\n▶ comprehensive (enhanced) | ${TASK} | ${MODEL}\n   → ${RUN_DIR}"
    
    # Create run directory and log configuration
    mkdir -p "${RUN_DIR}"
    echo "Configuration: approach=comprehensive, task=${TASK}, model=${MODEL}" > "${RUN_DIR}/config.txt"
    echo "Enhanced features: all capabilities combined, performance optimization, timeout handling" >> "${RUN_DIR}/config.txt"
    echo "Capabilities: retrieval + web search + visual operations + multiturn reasoning" >> "${RUN_DIR}/config.txt"
    
    python -m nova_retrieval_vlm.cli \
      task=${TASK} \
      approach=comprehensive \
      use_retrieval=true \
      use_web_search=true \
      retrieval.type=hybrid \
      retrieval.top_k=8 \
      multiturn_max_steps=3 \
      comprehensive_timeout=300 \
      model.name="${MODEL}" \
      batch_size=${BATCH_SIZE} \
      max_iterations=${MAX_ITERS} \
      paths.data_dir="${DATA_DIR}" \
      paths.output_dir="${RUN_DIR}" \
      strict_mode=true
      
    # Log completion status
    if [ $? -eq 0 ]; then
      echo "✓ SUCCESS" >> "${RUN_DIR}/status.txt"
    else
      echo "✗ FAILED" >> "${RUN_DIR}/status.txt"
    fi
  done
done

echo "\n✅ Comprehensive benchmark finished → ${OUTPUT_DIR}" 