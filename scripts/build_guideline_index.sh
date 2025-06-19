#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=_common_env.sh
source "$(dirname "$0")/_common_env.sh"

# Build BM25 / FAISS indexes for NICE guideline corpus used by retrieval.
# Usage: ./scripts/build_guideline_index.sh [--data-dir PATH] [--index-dir PATH]

DATA_DIR="$HOME/data/guidelines"
INDEX_DIR="$PWD/nova_retrieval_vlm/indexes"
VERBOSE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --index-dir) INDEX_DIR="$2"; shift 2;;
    --verbose) VERBOSE="--verbose"; shift 1;;
    -h|--help)
      echo "Usage: $(basename "$0") [options]"; echo "  --data-dir   PATH   Location to cache raw guideline docs"; echo "  --index-dir  PATH   Output directory for indexes"; echo "  --verbose           Show per-page crawl progress"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "${INDEX_DIR}"

# -----------------------------------------------------------------------------
# Build indexes using the dedicated Python helper.
# -----------------------------------------------------------------------------
echo "📚 Building guideline retrieval indexes"
echo "CONFIG_YAML = ${REPO_ROOT}/docs/guidelines.yaml"
echo "RAW_DIR     = ${DATA_DIR}"
echo "INDEX_DIR   = ${INDEX_DIR}"

python "${REPO_ROOT}/scripts/build_index.py" \
  --config "${REPO_ROOT}/docs/guidelines.yaml" \
  --raw-dir "${DATA_DIR}" \
  --output-dir "${INDEX_DIR}" \
  ${VERBOSE}

echo "✅ Retrieval indexes created under ${INDEX_DIR}" 