#!/usr/bin/env bash
"""Run baseline NOVA configurations only.

This script runs inference for baseline and baseline_reasoning configurations,
comparing single-turn analysis vs single-turn with reasoning.

Usage:
    bash scripts/run_baseline_configs.sh [--dry-run]

Options:
    --dry-run    Print what would be done without executing
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if dry run
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No commands will be executed${NC}"
    echo
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Config directory
CONFIG_DIR="$PROJECT_ROOT/config"

# Check if config directory exists
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo -e "${RED}Error: Config directory not found: $CONFIG_DIR${NC}" >&2
    exit 1
fi

# Define baseline configs only
configs=(
    "$CONFIG_DIR/baseline.yaml"
    "$CONFIG_DIR/baseline_reasoning.yaml"
)

# Check that all configs exist
missing_configs=()
for config in "${configs[@]}"; do
    if [[ ! -f "$config" ]]; then
        missing_configs+=("$config")
    fi
done

if [[ ${#missing_configs[@]} -gt 0 ]]; then
    echo -e "${RED}Error: Required baseline config files not found:${NC}" >&2
    for config in "${missing_configs[@]}"; do
        echo -e "${RED}  - $config${NC}" >&2
    done
    exit 1
fi

echo -e "${BLUE}Baseline configurations to run:${NC}"
echo

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo "  - $config_name"
done

echo
echo -e "${GREEN}Running baseline configurations (comparing reasoning vs non-reasoning)...${NC}"
echo

# Function to run inference for a config
run_inference() {
    local config="$1"
    local config_name

    config_name=$(basename "$config" .yaml)

    echo -e "${BLUE}=== Running: $config_name ===${NC}"

    # Build the command
    cmd="uv run python scripts/inference.py --config $config"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}$cmd${NC}"
    else
        echo -e "${GREEN}Executing: $cmd${NC}"
        if eval "$cmd"; then
            echo -e "${GREEN}✅ Completed: $config_name${NC}"
        else
            echo -e "${RED}❌ Failed: $config_name${NC}"
            return 1
        fi
    fi

    echo
}

# Counter for progress
total=${#configs[@]}
current=0

# Run baseline configurations
for config in "${configs[@]}"; do
    ((current++))

    config_name=$(basename "$config" .yaml)
    echo -e "${BLUE}[$current/$total] Processing: $config_name${NC}"

    run_inference "$config"

    # Add a small delay between runs to be kind to APIs
    if [[ "$DRY_RUN" == "false" ]] && [[ $current -lt $total ]]; then
        echo -e "${YELLOW}Waiting 5 seconds before next run...${NC}"
        sleep 5
    fi

    echo
done

if [[ "$DRY_RUN" == "false" ]]; then
    echo -e "${GREEN}🎉 Baseline configurations completed!${NC}"
    echo -e "${BLUE}Results will be in ./results/ directory:${NC}"
    echo -e "${BLUE}  - results/baseline_x_ai_grok_4.1_fast_free/ (non-reasoning)${NC}"
    echo -e "${BLUE}  - results/baseline_reasoning_x_ai_grok_4.1_fast_free/ (with reasoning)${NC}"
    echo
    echo -e "${YELLOW}You can now compare baseline vs baseline_reasoning performance:${NC}"
    echo -e "${YELLOW}  uv run python scripts/evaluate.py --batch --output ./eval_baseline${NC}"
else
    echo -e "${YELLOW}Dry run completed. Remove --dry-run flag to execute.${NC}"
fi

echo -e "${BLUE}💡 Tip: Use the evaluation script to compare results:${NC}"
echo -e "${BLUE}   uv run python scripts/evaluate.py --batch --output ./eval_baseline${NC}"