#!/usr/bin/env bash
set -euo pipefail

# Modern NOVA Retrieval VLM Quality Check Script
# Enforces modern development standards using uv, ruff, pyright

echo "🔍 Running modern code quality checks..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    elif [ "$status" = "ERROR" ]; then
        echo -e "${RED}❌ $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_status "ERROR" "Must be run from project root directory"
    exit 1
fi

# Ensure modern tools are available
if ! command -v uv &> /dev/null; then
    print_status "ERROR" "uv not found. Install with: pip install uv"
    exit 1
fi

# Initialize counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNINGS=0

run_check() {
    local check_name=$1
    local command=$2
    local allow_failure=${3:-false}
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    print_status "INFO" "Running $check_name..."
    
    if eval "$command" >/dev/null 2>&1; then
        print_status "OK" "$check_name passed"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        if [ "$allow_failure" = "true" ]; then
            print_status "WARN" "$check_name failed (non-critical)"
            WARNINGS=$((WARNINGS + 1))
        else
            print_status "ERROR" "$check_name failed"
            echo "Command: $command"
            echo "Output:"
            eval "$command" 2>&1 | head -20
            return 1
        fi
    fi
}

echo
print_status "INFO" "1. Modern Dependency Management"
echo "--------------------------------"
run_check "uv environment sync" "uv sync --check"

echo
print_status "INFO" "2. Ruff Code Quality (replaces black/isort/flake8)"
echo "--------------------------------------------------"
run_check "Ruff linting" "uv run ruff check ."
run_check "Ruff formatting" "uv run ruff format --check ."

echo
print_status "INFO" "3. Modern Type Checking"
echo "------------------------"
if command -v pyright &> /dev/null; then
    run_check "Pyright type checking" "pyright src/nova_retrieval_vlm/"
else
    print_status "WARN" "pyright not found. Install with: npm install -g pyright"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for beartype decorators on key functions
BEARTYPE_COUNT=$(fd -e py . src/ -x grep -l "@beartype" {} | wc -l || echo 0)
if [ "$BEARTYPE_COUNT" -gt 5 ]; then
    print_status "OK" "Found $BEARTYPE_COUNT files with @beartype decorators"
else
    print_status "WARN" "Only $BEARTYPE_COUNT files use @beartype - add more runtime validation"
    WARNINGS=$((WARNINGS + 1))
fi

echo
print_status "INFO" "4. Security & AI Slop Detection"
echo "--------------------------------"
# Use ruff's bandit rules instead of separate tool
run_check "Security rules (bandit via ruff)" "uv run ruff check --select S ."

# Check for AI slop patterns
AI_SLOP_FOUND=0
if fd -e py . src/ -x grep -l "robust\|fallback\|best.effort" {} 2>/dev/null | head -1 >/dev/null; then
    print_status "ERROR" "Found AI slop patterns (robust/fallback code)"
    AI_SLOP_FOUND=1
fi

if fd -e py . src/ -x grep -l "except Exception:" {} 2>/dev/null | head -1 >/dev/null; then
    print_status "ERROR" "Found broad exception handling"
    AI_SLOP_FOUND=1
fi

if fd -e py . src/ -x grep -l "Union\[" {} 2>/dev/null | head -1 >/dev/null; then
    print_status "ERROR" "Found legacy Union syntax (use X | Y instead)"
    AI_SLOP_FOUND=1
fi

if [ $AI_SLOP_FOUND -eq 0 ]; then
    print_status "OK" "No AI slop patterns detected"
else
    print_status "ERROR" "AI slop detected - fix before committing"
    return 1
fi

echo
print_status "INFO" "5. Test Execution"
echo "-----------------"
run_check "Unit tests with uv" "uv run pytest tests/ --tb=short --disable-warnings"

echo
print_status "INFO" "7. Documentation Checks"
echo "-----------------------"
run_check "Docstring presence" "python -c \"import ast; import sys; [sys.exit(1) for f in ['src/nova_retrieval_vlm/__init__.py', 'src/nova_retrieval_vlm/cli.py'] if not ast.get_docstring(ast.parse(open(f).read()))]\"" "true"

# Check for TODO/FIXME items
TODO_COUNT=$(grep -r "TODO\|FIXME" src/ tests/ --exclude-dir=__pycache__ | wc -l || echo 0)
if [ "$TODO_COUNT" -gt 0 ]; then
    print_status "WARN" "Found $TODO_COUNT TODO/FIXME items"
    WARNINGS=$((WARNINGS + 1))
else
    print_status "OK" "No TODO/FIXME items found"
fi

echo
print_status "INFO" "8. Performance Checks"
echo "---------------------"

# Check for processor pattern usage
if [ -d "src/nova_retrieval_vlm/processors" ]; then
    print_status "OK" "Modern processor pattern implemented"
else
    print_status "ERROR" "Processor pattern missing"
    return 1
fi

# Check for modern CLI
if [ -f "src/nova_retrieval_vlm/cli_new.py" ]; then
    print_status "OK" "Modern CLI implementation exists"
else
    print_status "WARN" "Modern CLI not found"
    WARNINGS=$((WARNINGS + 1))
fi

echo
print_status "INFO" "9. File Structure Checks"
echo "------------------------"

# Check for required files
REQUIRED_FILES=("README.md" "pyproject.toml" ".gitignore" "CLAUDE.md")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_status "OK" "$file exists"
    else
        print_status "ERROR" "$file missing"
        exit 1
    fi
done

# Check for large files that shouldn't be committed
LARGE_FILES=$(find . -name "*.py" -size +1M 2>/dev/null | head -5)
if [ ! -z "$LARGE_FILES" ]; then
    print_status "WARN" "Found large Python files (>1MB):"
    echo "$LARGE_FILES"
    WARNINGS=$((WARNINGS + 1))
fi

echo
print_status "INFO" "10. Environment Setup Check"
echo "----------------------------"

# Check if .env.example exists
if [ -f ".env.example" ]; then
    print_status "OK" ".env.example exists"
else
    print_status "WARN" ".env.example missing - create template for users"
    WARNINGS=$((WARNINGS + 1))
fi

# Check if critical dependencies are installable with uv
uv run python -c "
try:
    import nova_retrieval_vlm
    print('✅ Package imports successfully with uv')
except ImportError as e:
    print(f'❌ Package import failed: {e}')
    exit(1)
"

# Check for type safety imports
uv run python -c "
try:
    from nova_retrieval_vlm.types import ImageTensor, ModelResponse
    from beartype import beartype
    from jaxtyping import Float
    print('✅ Type safety modules available')
except ImportError as e:
    print(f'⚠️  Type safety modules missing: {e}')
" || print_status "WARN" "Type safety modules not fully available"

echo
echo "=================================================="
print_status "INFO" "Quality Check Summary"
echo "=================================================="
echo "Total checks: $TOTAL_CHECKS"
echo "Passed: $PASSED_CHECKS"
echo "Warnings: $WARNINGS"
echo "Failed: $((TOTAL_CHECKS - PASSED_CHECKS - WARNINGS))"

SCORE=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
echo "Quality Score: $SCORE%"

if [ $SCORE -ge 90 ]; then
    print_status "OK" "Excellent code quality! 🎉"
    exit 0
elif [ $SCORE -ge 80 ]; then
    print_status "OK" "Good code quality ✨"
    exit 0
elif [ $SCORE -ge 70 ]; then
    print_status "WARN" "Acceptable code quality, but room for improvement"
    exit 0
else
    print_status "ERROR" "Code quality needs improvement before committing"
    exit 1
fi