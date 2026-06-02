#!/usr/bin/env bash
# Run local model evaluation: single-turn then agentic.
# Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]
#   MODEL       required — e.g. glm-4.6v-flash, medgemma-1.5-4b-it
#   BASE_URL    defaults to http://localhost:1234/v1
#   MAX_SAMPLES defaults to 50
#
# NOTE: Only load one model in LM Studio at a time. The health-check probe
# can trigger model swapping on memory-constrained GPUs (see EB-4).
#
# CONTEXT WINDOW: VQA-RAD works with n_ctx >= 4096 but 8192 recommended.
# Thinking models (Qwen 3.5) need extra headroom for chain-of-thought.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:?Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]}"
BASE_URL="${2:-http://localhost:1234/v1}"
MAX_SAMPLES="${3:-50}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

echo "=== VQA-RAD local model evaluation ==="
echo "Model:    ${MODEL}"
echo "Endpoint: ${BASE_URL}"
echo "Samples:  ${MAX_SAMPLES}"
echo ""

# --- Single-turn ---
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn__${MAX_SAMPLES}s"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run python -m examples.vqa_rad.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --mode single_turn \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --output-dir "${SINGLE_DIR}" \
  -v

echo ""

# --- Agentic (with tools) ---
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__tools__5t__${MAX_SAMPLES}s"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run python -m examples.vqa_rad.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --mode agentic \
  --use-tools \
  --max-turns 5 \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --output-dir "${AGENTIC_DIR}" \
  -v

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Single-turn: ${SINGLE_DIR}/summary.json"
echo "  Agentic:     ${AGENTIC_DIR}/summary.json"
