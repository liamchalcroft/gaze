#!/usr/bin/env bash
# Shared environment bootstrap for benchmark and utility scripts.
# Sets PYTHONPATH so that `python -m nova_retrieval_vlm.cli` works without
# installation and installs editable package if missing via uv or pip.

set -euo pipefail

# Determine repository root (parent directory of the current script)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# The Python package lives under <repo>/src, so expose that on PYTHONPATH
SRC_PATH="${REPO_ROOT}/src"
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH:-}"

if ! python - <<'PY'
import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('nova_retrieval_vlm') else 1)
PY
then
  echo "📦  Installing local nova_retrieval_vlm package (editable)…"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e "${REPO_ROOT}"
  else
    python -m pip install -e "${REPO_ROOT}"
  fi
fi 