#!/usr/bin/env bash
# Run local model evaluation: single-turn then agentic.
# Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]
#   MODEL       required — e.g. qwen3.5-35b-a3b, glm-4.6v-flash, medgemma-1.5-4b-it
#   BASE_URL    defaults to http://192.168.1.138:1234/v1
#   MAX_SAMPLES defaults to 50 (0 = all)

set -euo pipefail

MODEL="${1:?Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]}"
BASE_URL="${2:-http://192.168.1.138:1234/v1}"
MAX_SAMPLES="${3:-50}"
RESULTS_DIR="./runs/main_results"

# Use the local model for diagnosis semantic matching too
export NOVA_SEMANTIC_MATCH_MODEL="${MODEL}"
export NOVA_SEMANTIC_MATCH_BASE_URL="${BASE_URL}"

echo "=== Local model evaluation ==="
echo "Model:    ${MODEL}"
echo "Endpoint: ${BASE_URL}"
echo "Samples:  ${MAX_SAMPLES} (0 = all)"
echo "Metrics:  caption + diagnosis + localization"
echo ""

# --- Single-turn (no tools) ---
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn__notools__10t__all"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run python -m src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --mode single_turn \
  --max-turns 10 \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 1 \
  --output-dir "${SINGLE_DIR}" \
  -v

echo ""

# --- Agentic (with tools) ---
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__tools__10t__all"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run python -m src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --mode agentic \
  --use-tools \
  --max-turns 10 \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 1 \
  --output-dir "${AGENTIC_DIR}" \
  -v

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Single-turn: ${SINGLE_DIR}/summary.json"
echo "  Agentic:     ${AGENTIC_DIR}/summary.json"
