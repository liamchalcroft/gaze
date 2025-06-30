#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/model_list.sh"

DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/full_benchmark/visual"
BATCH_SIZE=1
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

echo "==== VISUAL BENCHMARK (Enhanced System Prompts) ===="
echo "DATA_DIR   = ${DATA_DIR}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "MODELS     = ${MODELS_DEFAULT[*]}"
echo "TASKS      = ${TASKS[*]}"
echo "APPROACH   = visual (optimized with enhanced guidance)"
echo "====================================================="

for MODEL in "${MODELS_DEFAULT[@]}"; do
  for TASK in "${TASKS[@]}"; do
    RUN_DIR="${OUTPUT_DIR}/${TASK}/$(echo "${MODEL}" | tr '/:' '_')"
    echo -e "\n▶ visual (enhanced) | ${TASK} | ${MODEL}\n   → ${RUN_DIR}"
    
    # Create run directory and log configuration
    mkdir -p "${RUN_DIR}"
    echo "Configuration: approach=visual, task=${TASK}, model=${MODEL}" > "${RUN_DIR}/config.txt"
    echo "Enhanced features: visual operations guidance, systematic analysis, performance optimization" >> "${RUN_DIR}/config.txt"
    
    if python -m nova_retrieval_vlm.cli \
      task=${TASK} \
      approach=visual \
      use_retrieval=false \
      visual_rounds=2 \
      retrieval.type=hybrid \
      retrieval.top_k=5 \
      model.name="${MODEL}" \
      batch_size=${BATCH_SIZE} \
      max_iterations=${MAX_ITERS} \
      paths.data_dir="${DATA_DIR}" \
      paths.output_dir="${RUN_DIR}" \
      skip_existing=true \
      strict_mode=true; then
      echo "✓ SUCCESS" >> "${RUN_DIR}/status.txt"
    else
      echo "✗ FAILED" >> "${RUN_DIR}/status.txt"
    fi
  done
done

echo "\n✅ Visual multiturn benchmark finished → ${OUTPUT_DIR}"
