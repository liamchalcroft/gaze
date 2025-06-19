#!/usr/bin/env bash
# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"
set -euo pipefail

# -----------------------------------------------------------------------------
# NOVA VLM – FULL BENCHMARK SUITE
# -----------------------------------------------------------------------------
# This script evaluates **both** approaches (baseline, multiturn) across the
# three NOVA tasks (localization, caption, diagnosis) on a configurable list of
# OpenRouter models.
#
# Usage:
#   ./scripts/run_full_benchmarks.sh \
#       [--data-dir /abs/path/to/nova] \
#       [--output-dir runs/full_benchmark] \
#       [--batch-size 4] \
#       [--max-iters -1]
#
# Environment variables:
#   MODELS      Space-separated list of model identifiers (defaults below)
#   BATCH_SIZE  Overrides --batch-size
#
# The CLI invoked is nova_retrieval_vlm.cli.  Retrieval is **disabled** for the
# baseline approach and **enabled** for the multiturn approach.
# -----------------------------------------------------------------------------

# Defaults
DATA_DIR="$HOME/data/nova"
OUTPUT_DIR="$PWD/runs/full_benchmark"
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITERS=-1

# Determine directory of this script so we can invoke companion scripts reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse CLI opts
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    -h|--help)
      echo "Usage: $(basename "$0") [options]"; exit 0;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

${SCRIPT_DIR}/run_baseline_benchmark.sh --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}/baseline" --batch-size "${BATCH_SIZE}" --max-iters "${MAX_ITERS}"

${SCRIPT_DIR}/run_multiturn_benchmark.sh --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}/multiturn" --batch-size "${BATCH_SIZE}" --max-iters "${MAX_ITERS}"

echo "\n✅ Full benchmark suite completed. Consolidated results in: ${OUTPUT_DIR}" 