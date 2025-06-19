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

# -----------------------------------------------------------------------------
# Ensure a dedicated uv-managed virtual environment is present and activated.
# -----------------------------------------------------------------------------
VENV_DIR="${REPO_ROOT}/.venv"

# If we are not already inside any virtual-env, create/activate the project one
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "🛠  Creating uv virtual environment at ${VENV_DIR}"
    if command -v uv >/dev/null 2>&1; then
      uv venv "${VENV_DIR}"
    else
      python -m venv "${VENV_DIR}"
    fi
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

# After activation, make sure the editable package (and its deps) are installed.
if ! python - <<'PY'
import sys
try:
    from timm.data import ImageNetInfo  # ensures timm ≥ 0.9.7 is available
    from haystack.document_stores.in_memory import InMemoryDocumentStore  # haystack present
    import nova_retrieval_vlm           # verifies project import works
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then
  echo "📦  Ensuring local package *and* dependencies are installed…"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --upgrade -e "${REPO_ROOT}"
  else
    python -m pip install --upgrade -e "${REPO_ROOT}"
  fi
fi 