#!/usr/bin/env bash
# Run local model evaluation on the AgentClinic NEJM diagnostic benchmark.
# Usage: ./run_local.sh MODEL [BASE_URL] [NUM_SAMPLES] [DATASET]
#   MODEL       required, e.g. qwen3.5-35b-a3b, glm-4.6v-flash, medgemma-1.5-4b-it
#   BASE_URL    defaults to http://localhost:1234/v1
#   NUM_SAMPLES defaults to 50 (-1 = all)
#   DATASET     defaults to the bundled agentclinic_nejm_extended.jsonl
#
# NOTE: Only load one model in LM Studio at a time. The health-check probe
# can trigger model swapping on memory-constrained GPUs (see EB-4).
#
# CONTEXT WINDOW: AgentClinic is multi-turn; conversation history grows with
# each request. Use n_ctx >= 8192. Thinking models (Qwen 3.5) need headroom
# for chain-of-thought, hence --max-tokens 8192.
#
# Image cases: NEJM cases may include an image. A vision model is needed for
# those; text-only models still run but cannot interpret the IMAGE responses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:?Usage: ./run_local.sh MODEL [BASE_URL] [NUM_SAMPLES] [DATASET]}"
BASE_URL="${2:-http://localhost:1234/v1}"
NUM_SAMPLES="${3:--1}"
DATASET="${4:-}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

echo "=== AgentClinic NEJM local model evaluation ==="
echo "Model:    ${MODEL}"
echo "Endpoint: ${BASE_URL}"
echo "Samples:  ${NUM_SAMPLES} (-1 = all)"
echo "Dataset:  ${DATASET:-<bundled default>}"
echo ""

OUT_DIR="${RESULTS_DIR}/${MODEL}__multi_turn__10t"
echo "--- Multi-turn run -> ${OUT_DIR} ---"

DATASET_ARGS=()
if [[ -n "${DATASET}" ]]; then
  DATASET_ARGS=(--dataset "${DATASET}")
fi

uv run python -m examples.agentclinic_nejm.eval \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --max-turns 10 \
  --num-samples "${NUM_SAMPLES}" \
  --max-tokens 8192 \
  --output "${OUT_DIR}" \
  "${DATASET_ARGS[@]}" \
  --verbose

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Multi-turn: ${OUT_DIR}/agentclinic_eval_${MODEL//\//_}.json"
