#!/usr/bin/env bash
# Run local model evaluation: single-turn then agentic.
# Usage: ./run_local.sh MODEL DATASET IMAGE_DIR [BASE_URL] [NUM_SAMPLES]
#   MODEL       required — e.g. qwen3.5-35b-a3b, glm-4.6v-flash
#   DATASET     required — path to GEMeX JSONL dataset
#   IMAGE_DIR   required — MIMIC-CXR image root directory
#   BASE_URL    defaults to http://localhost:1234/v1
#   NUM_SAMPLES defaults to 50 (-1 = all)
#
# NOTE: Only load one model in LM Studio at a time. The health-check probe
# can trigger model swapping on memory-constrained GPUs (see EB-4).
#
# CONTEXT WINDOW: GEMeX requires n_ctx >= 8192 (recommended 16384).
# Image + schema prompt is large; thinking models need extra headroom.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:?Usage: ./run_local.sh MODEL DATASET IMAGE_DIR [BASE_URL] [NUM_SAMPLES]}"
DATASET="${2:?Usage: ./run_local.sh MODEL DATASET IMAGE_DIR [BASE_URL] [NUM_SAMPLES]}"
IMAGE_DIR="${3:?Usage: ./run_local.sh MODEL DATASET IMAGE_DIR [BASE_URL] [NUM_SAMPLES]}"
BASE_URL="${4:-http://localhost:1234/v1}"
NUM_SAMPLES="${5:-50}"
RESULTS_DIR="${SCRIPT_DIR}/runs/main_results"

cd "${REPO_ROOT}"

# The GEMeX JSONL is built from the HuggingFace source by prepare_data.py and is
# not shipped with the repo. Fail early with a clear hint if it is absent.
if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: dataset not found at ${DATASET}" >&2
  echo "Build it first (needs the gemex extra):" >&2
  echo "  uv run --extra gemex python -m examples.gemex_thinkvg.prepare_data --split train" >&2
  exit 1
fi

echo "=== GEMeX-ThinkVG local model evaluation ==="
echo "Model:     ${MODEL}"
echo "Dataset:   ${DATASET}"
echo "Image dir: ${IMAGE_DIR}"
echo "Endpoint:  ${BASE_URL}"
echo "Samples:   ${NUM_SAMPLES} (-1 = all)"
echo ""

# --- Single-turn ---
SINGLE_DIR="${RESULTS_DIR}/${MODEL}__single_turn"
echo "--- Single-turn run -> ${SINGLE_DIR} ---"
uv run --extra gemex python -m examples.gemex_thinkvg.eval \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --dataset "${DATASET}" \
  --image-dir "${IMAGE_DIR}" \
  --mode single_turn \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${SINGLE_DIR}" \
  --verbose

echo ""

# --- Agentic (with tools) ---
AGENTIC_DIR="${RESULTS_DIR}/${MODEL}__agentic__tools__8t"
echo "--- Agentic run -> ${AGENTIC_DIR} ---"
uv run --extra gemex python -m examples.gemex_thinkvg.eval \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --dataset "${DATASET}" \
  --image-dir "${IMAGE_DIR}" \
  --mode agentic \
  --use-tools \
  --max-turns 8 \
  --max-tokens 8192 \
  --max-image-dim 256 \
  --num-samples "${NUM_SAMPLES}" \
  --output "${AGENTIC_DIR}" \
  --verbose

echo ""
echo "=== Done ==="
echo "Results:"
echo "  Single-turn: ${SINGLE_DIR}/"
echo "  Agentic:     ${AGENTIC_DIR}/"
