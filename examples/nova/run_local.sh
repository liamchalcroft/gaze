#!/usr/bin/env bash
# Run local model evaluation: single-turn then agentic.
# Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]
#   MODEL       required — e.g. qwen3.5-35b-a3b, glm-4.6v-flash, medgemma-1.5-4b-it
#   BASE_URL    defaults to http://localhost:1234/v1
#   MAX_SAMPLES defaults to 50 (0 = all)
#
# NOTE: Only load one model in LM Studio at a time. The health-check probe
# can trigger model swapping on memory-constrained GPUs (see EB-4).
#
# CONTEXT WINDOW: NOVA requires n_ctx >= 8192 (recommended 16384).
# The prompt+image uses ~2700 tokens; the NOVA JSON schema needs ~2000+
# tokens for caption+diagnosis+localization. With thinking models
# (Qwen 3.5), add 2000-4000 tokens for chain-of-thought overhead.
# Set n_ctx in LM Studio → Model Settings before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:?Usage: ./run_local.sh MODEL [BASE_URL] [MAX_SAMPLES]}"
BASE_URL="${2:-http://localhost:1234/v1}"
MAX_SAMPLES="${3:-50}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

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
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn__notools__1t__all"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run --extra nova python -m examples.nova.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --mode single_turn \
  --max-turns 1 \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --batch-size 1 \
  --output-dir "${SINGLE_DIR}" \
  -v

echo ""

# --- Agentic (with tools) ---
# max-turns=5: prevents rote tool exhaustion (baseline avg was 18.9/19 turns
# with every tool called in fixed order — no benefit beyond 3-5 focused tools).
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__tools__5t__all"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run --extra nova python -m examples.nova.src.cli \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --task all \
  --mode agentic \
  --use-tools \
  --max-turns 5 \
  --max-samples "${MAX_SAMPLES}" \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --batch-size 1 \
  --output-dir "${AGENTIC_DIR}" \
  -v

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Single-turn: ${SINGLE_DIR}/summary.json"
echo "  Agentic:     ${AGENTIC_DIR}/summary.json"
