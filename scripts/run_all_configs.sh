#!/usr/bin/env bash
"""Run inference for all NOVA configuration files.

This script runs inference.py for each config YAML file in the config/ directory,
creating separate result directories for each configuration.

Usage:
    bash scripts/run_all_configs.sh [--dry-run]

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

# Find all YAML files in config directory (exclude config.yaml which is a template)
configs=()
while IFS= read -r -d '' config; do
    config_name=$(basename "$config")
    # Skip config.yaml as it's a template
    if [[ "$config_name" != "config.yaml" ]]; then
        configs+=("$config")
    fi
done < <(find "$CONFIG_DIR" -name "*.yaml" -type f -print0 | sort -z)

if [[ ${#configs[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No YAML config files found in $CONFIG_DIR${NC}" >&2
    exit 1
fi

echo -e "${BLUE}Found ${#configs[@]} configuration files:${NC}"
echo

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo "  - $config_name"
done

echo
echo -e "${GREEN}Running inference for all configurations...${NC}"
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

# Run inference for each config
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
    echo -e "${GREEN}🎉 All inference runs completed!${NC}"
    echo -e "${BLUE}Check the ./results/ directory for output files.${NC}"
else
    echo -e "${YELLOW}Dry run completed. Remove --dry-run flag to execute.${NC}"
fi