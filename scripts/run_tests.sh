#!/usr/bin/env bash
# Test runner script for GAZE

set -euo pipefail

echo "Running GAZE test suite..."

# Parse arguments
COVERAGE=false
FAST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --fast|-f)
            FAST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--coverage] [--fast]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="uv run pytest"

if [ "$FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x -v --tb=short"
    echo "Running in fast mode (stop on first failure)"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=gaze --cov-report=html --cov-report=term-missing"
    echo "Running with coverage"
fi

# Run tests
echo ""
echo "Command: $PYTEST_CMD"
eval $PYTEST_CMD

echo ""
echo "All tests passed!"
