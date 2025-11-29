# NOVA Evaluation Scripts

Simplified scripts for running NOVA dataset evaluations and analyzing results for research papers.

## Scripts Overview

### 1. `run_nova_evaluation.py`
Runs NOVA dataset evaluation with specified configuration file.
Performs unified multi-task analysis (captioning + diagnosis + localization) in one pass.

```bash
# Run with specific configuration
uv run python scripts/run_nova_evaluation.py --config config/baseline.yaml

# With custom output directory
uv run python scripts/run_nova_evaluation.py \
    --config config/agentic.yaml \
    --output-dir ./runs/experiment_1
```

### 2. `analyze_results.py`
Analyzes results from multiple evaluation runs and creates plots/tables for papers.

```bash
# Analyze all results
uv run python scripts/analyze_results.py --input-dir ./runs --output-dir ./paper_results

# Generate model comparison
uv run python scripts/analyze_results.py --input-dir ./runs --compare-models
```

**Outputs:**
- `performance_comparison.png` - Performance plots
- `results_table.tex` - LaTeX table for main results
- `model_comparison_table.tex` - Model comparison table
- `summary_stats.json` - Summary statistics
- `metrics.csv` - Raw metrics data

### 3. `eval_nova.sh` (Shell Wrapper)
Convenient wrapper for common operations.

```bash
# Run evaluation with config
./scripts/eval_nova.sh config/baseline.yaml

# Analyze results
./scripts/eval_nova.sh analyze ./runs ./paper_results

# Show help
./scripts/eval_nova.sh help
```

### 4. `check_quality.sh`
Runs code quality checks (ruff, pyright, pytest).

```bash
./scripts/check_quality.sh
```

## Configuration Files

### `config/baseline.yaml`
Standard baseline evaluation without agentic processing.

### `config/agentic.yaml`
Agentic evaluation with visual tools and multi-turn reasoning.

## Usage Workflow

1. **Run evaluations:**
   ```bash
   ./scripts/eval_nova.sh config/baseline.yaml
   ./scripts/eval_nova.sh config/agentic.yaml
   ```

2. **Analyze results:**
   ```bash
   ./scripts/eval_nova.sh analyze
   ```

3. **Use generated outputs:**
   - Include LaTeX tables in paper
   - Use plots in presentations
   - Reference summary statistics

## Key Features

- **Unified Analysis**: All evaluations perform captioning + diagnosis + localization in one pass
- **Configuration-Driven**: Easy to experiment with different settings
- **Paper-Ready**: Generates LaTeX tables and publication-quality plots
- **Simplified Interface**: Only essential functionality, no complexity

## Notes

- Default model: `x-ai/grok-4.1-fast:free`
- Results are saved with timestamps
- Agentic mode uses smaller batch sizes for processing efficiency
- All scripts handle errors gracefully and provide informative output