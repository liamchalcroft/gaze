#!/usr/bin/env bash
# Run local model evaluation: single-turn then agentic.
# Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]
#   MODEL       required — e.g. qwen3.5-35b-a3b
#   BASE_URL    defaults to http://192.168.1.138:1234/v1
#   MAX_SAMPLES defaults to 50
#
# NOTE: Only load one model in LM Studio at a time. The health-check probe
# can trigger model swapping on memory-constrained GPUs (see EB-4).
#
# CONTEXT WINDOW: PubMedQA works with n_ctx >= 4096 (text-only).
# Thinking models (Qwen 3.5) may need 8192 for chain-of-thought overhead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:?Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]}"
BASE_URL="${2:-http://192.168.1.138:1234/v1}"
MAX_SAMPLES="${3:-50}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

echo "=== PubMedQA local model evaluation ==="
echo "Model:    ${MODEL}"
echo "Endpoint: ${BASE_URL}"
echo "Samples:  ${MAX_SAMPLES}"
echo ""

# --- Single-turn ---
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn__${MAX_SAMPLES}s"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run python -m examples.pubmedqa.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --mode single_turn \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --output-dir "${SINGLE_DIR}" \
  -v

echo ""

# --- Agentic (with search) ---
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__search__5t__${MAX_SAMPLES}s"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run python -m examples.pubmedqa.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --mode agentic \
  --use-search \
  --max-turns 5 \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --output-dir "${AGENTIC_DIR}" \
  -v

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Single-turn: ${SINGLE_DIR}/summary.json"
echo "  Agentic:     ${AGENTIC_DIR}/summary.json"
