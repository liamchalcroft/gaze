#!/usr/bin/env bash
set -euo pipefail

# Code quality checks using uv, ruff, pyright

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from project root"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: pip install uv"
    exit 1
fi

echo "Running code quality checks..."
echo "=============================="

FAILED=0

echo ""
echo "1. Checking dependencies..."
if uv sync --check 2>/dev/null; then
    echo "   OK: Dependencies synced"
else
    echo "   WARN: Run 'uv sync' to update"
fi

echo ""
echo "2. Ruff linting..."
if uv run ruff check . 2>/dev/null; then
    echo "   OK: No linting errors"
else
    echo "   FAIL: Linting errors found"
    FAILED=1
fi

echo ""
echo "3. Ruff formatting..."
if uv run ruff format --check . 2>/dev/null; then
    echo "   OK: Code formatted correctly"
else
    echo "   WARN: Run 'uv run ruff format .' to fix"
fi

echo ""
echo "4. Pyright type checking..."
if uv run pyright src/nova_retrieval_vlm/ 2>/dev/null; then
    echo "   OK: No type errors"
else
    echo "   WARN: Type errors found (non-blocking)"
fi

echo ""
echo "5. Running tests..."
if uv run pytest tests/ -q --tb=no 2>/dev/null; then
    echo "   OK: Tests passed"
else
    echo "   FAIL: Tests failed"
    FAILED=1
fi

echo ""
echo "=============================="
if [ $FAILED -eq 0 ]; then
    echo "All critical checks passed"
    exit 0
else
    echo "Some checks failed"
    exit 1
fi
