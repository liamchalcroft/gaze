#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/model_list.sh"

DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/visual_benchmark"
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

echo "==== VISUAL MULTITURN BENCHMARK ===="
echo "DATA_DIR   = ${DATA_DIR}"
echo "OUTPUT_DIR = ${OUTPUT_DIR}"
echo "MODELS     = ${MODELS_DEFAULT[*]}"
echo "TASKS      = ${TASKS[*]}"
echo "====================================="

for MODEL in "${MODELS_DEFAULT[@]}"; do
  for TASK in "${TASKS[@]}"; do
    RUN_DIR="${OUTPUT_DIR}/${TASK}/$(echo "${MODEL}" | tr '/:' '_')"
    echo -e "\n▶ visual_multiturn | ${TASK} | ${MODEL}\n   → ${RUN_DIR}"
    python -m nova_retrieval_vlm.cli \
      task=${TASK} \
      approach=visual_multiturn \
      use_retrieval=true \
      model.name="${MODEL}" \
      batch_size=${BATCH_SIZE} \
      max_iterations=${MAX_ITERS} \
      paths.data_dir="${DATA_DIR}" \
      paths.output_dir="${RUN_DIR}"
  done
done

echo "\n✅ Visual multiturn benchmark finished → ${OUTPUT_DIR}"
