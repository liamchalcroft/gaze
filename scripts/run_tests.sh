#!/usr/bin/env bash
# Test runner script for Radiant Harness

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 Running Radiant Harness test suite..."

# Parse arguments
COVERAGE=false
FAST=false
WATCH=false

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
        --watch|-w)
            WATCH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--coverage] [--fast] [--watch]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="uv run pytest"

if [ "$FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x -v --tb=short"
    echo -e "${YELLOW}Running in fast mode (skip failures)${NC}"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=radiant_harness --cov-report=html --cov-report=term-missing"
    echo -e "${YELLOW}Running with coverage${NC}"
fi

if [ "$WATCH" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -f"
    echo -e "${YELLOW}Running in watch mode${NC}"
fi

# Run tests
echo ""
echo "Command: $PYTEST_CMD"
eval $PYTEST_CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi