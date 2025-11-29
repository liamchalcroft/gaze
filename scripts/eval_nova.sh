#!/bin/bash
# NOVA Dataset Evaluation Shell Wrapper
#
# Simple wrapper for running NOVA evaluations with configuration files
#
# Usage:
#   ./scripts/eval_nova.sh config/baseline.yaml     # Run baseline evaluation
#   ./scripts/eval_nova.sh config/agentic.yaml      # Run agentic evaluation
#   ./scripts/eval_nova.sh analyze                  # Analyze all results

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        exit 1
    fi
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "pyproject.toml" ] || [ ! -d "src/nova_retrieval_vlm" ]; then
        print_error "Must be run from the nova_retrieval_vlm root directory"
        exit 1
    fi
}

# Function to run evaluation with config
run_evaluation() {
    local config_file="$1"
    local output_dir="${2:-./runs/evaluation}"

    print_status "Running NOVA evaluation with config: $config_file"

    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi

    uv run python scripts/run_nova_evaluation.py \
        --config "$config_file" \
        --output-dir "$output_dir" \
        --verbose
}

# Function to analyze results
analyze_results() {
    local input_dir="${1:-./runs}"
    local output_dir="${2:-./paper_results}"

    print_status "Analyzing results from: $input_dir"
    print_status "Output directory: $output_dir"

    uv run python scripts/analyze_results.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --verbose
}

# Function to show usage
show_usage() {
    echo "NOVA Dataset Evaluation Shell Wrapper (Simplified)"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  <config_file>                  Run evaluation with specified configuration"
    echo "  analyze [input_dir] [output]   Analyze results and create plots/tables"
    echo "  help                          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 config/baseline.yaml                    # Run baseline evaluation"
    echo "  $0 config/agentic.yaml                     # Run agentic evaluation"
    echo "  $0 analyze ./runs ./paper_results          # Analyze results"
    echo "  $0 analyze                                  # Analyze with default paths"
    echo ""
    echo "Configuration files should be in config/ directory"
    echo "Results will be saved to specified output directory"
}

# Main script logic
check_uv
check_directory

case "${1:-}" in
    "analyze")
        analyze_results "${2:-./runs}" "${3:-./paper_results}"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        print_warning "No command specified. Showing help:"
        show_usage
        ;;
    *)
        # Assume it's a configuration file
        if [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
            run_evaluation "$1" "${2:-./runs/evaluation}"
        else
            print_error "Unknown command or invalid config file: $1"
            show_usage
            exit 1
        fi
        ;;
esac

print_success "Operation completed!"