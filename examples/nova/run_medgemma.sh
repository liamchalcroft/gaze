#!/usr/bin/env bash
# Run medgemma-1.5-4b-it evaluation: single-turn then agentic.
# Usage: ./run_medgemma.sh [BASE_URL] [MAX_SAMPLES]
#   BASE_URL    defaults to http://localhost:1234/v1
#   MAX_SAMPLES defaults to 20 (0 = all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="medgemma-1.5-4b-it"
BASE_URL="${1:-http://localhost:1234/v1}"
MAX_SAMPLES="${2:-20}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

echo "=== MedGemma evaluation ==="
echo "Model:    ${MODEL}"
echo "Endpoint: ${BASE_URL}"
echo "Samples:  ${MAX_SAMPLES} (0 = all)"
echo "Metrics:  caption + localization (skipping diagnosis)"
echo ""

# --- Single-turn (no tools) ---
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn__notools__1t__all"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run --extra nova python -m examples.nova.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --eval-tasks caption localization \
  --mode single_turn \
  --max-turns 1 \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size 1 \
  --output-dir "${SINGLE_DIR}" \
  -v

echo ""

# --- Agentic (with tools) ---
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__tools__10t__all"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run --extra nova python -m examples.nova.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --eval-tasks caption localization \
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
