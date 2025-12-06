# NOVA Scripts Directory

Essential scripts for NOVA dataset inference, evaluation, and result comparison.

## Core Scripts

### `inference.py` - Model Inference
Generates per-subject predictions from NOVA dataset using specified configuration.

```bash
# Run inference with a configuration file
uv run python scripts/inference.py --config config/baseline.yaml

# Run with custom output directory
uv run python scripts/inference.py --config config/agentic.yaml --output-dir ./results/my_run

# Run with verbose logging
uv run python scripts/inference.py --config config/baseline.yaml --verbose
```

### `evaluate.py` - Evaluation
Evaluates per-subject predictions against NOVA dataset ground truth.

```bash
# Evaluate results from inference
uv run python scripts/evaluate.py --results-dir ./results/baseline_model --output ./eval_baseline

# Evaluate with detailed metrics
uv run python scripts/evaluate.py --results-dir ./results/agentic_model --output ./eval_agentic --verbose
```

### `comparison.py` - Result Comparison
Aggregates metrics from multiple experimental runs and creates comparison plots/tables.

```bash
# Compare all results in a directory
uv run python scripts/comparison.py --parent-dir ./results --output ./paper_results

# Compare specific configurations
uv run python scripts/comparison.py --parent-dir ./results --models baseline agentic baseline_reasoning
```

## Typical Workflow

```bash
# 1. Run inference with baseline config
uv run python scripts/inference.py --config config/baseline.yaml

# 2. Evaluate the results
uv run python scripts/evaluate.py --results-dir ./results/baseline_* --output ./eval_baseline

# 3. Repeat for other configurations...
uv run python scripts/inference.py --config config/agentic.yaml
uv run python scripts/evaluate.py --results-dir ./results/agentic_* --output ./eval_agentic

# 4. Compare all results
uv run python scripts/comparison.py --parent-dir ./eval_* --output ./paper_results
```

## Configuration Files

Configuration files are in `config/`:
- `baseline.yaml` - Standard single-turn processing
- `baseline_reasoning.yaml` - Single-turn with reasoning enabled
- `agentic.yaml` - Agentic processing (multi-turn)
- `agentic_reasoning.yaml` - Agentic with reasoning
- `agentic_tools.yaml` - Agentic with visual tools
- `agentic_tools_reasoning.yaml` - Full agentic pipeline

## Dependencies

- **Python**: 3.10+
- **uv**: Package management (`uv sync` to install)
- **API Keys**: `OPENROUTER_API_KEY` environment variable
- **Data**: NOVA dataset access (HuggingFace)

## Output Structure

```
results/
├── baseline_model_name/
│   └── per_subject/
│       ├── subject_0000/
│       │   ├── predictions.json
│       │   └── summary.json
│       └── ...

eval_baseline/
├── evaluation_metrics.json
├── per_task_metrics.json
└── detailed_analysis.json
```
