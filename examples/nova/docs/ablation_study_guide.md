# NOVA VLM Ablation Study Guide

This guide explains how to run comprehensive ablation studies using the NOVA Retrieval VLM framework integrated with the NOVA dataset from HuggingFace.

## Overview

The ablation study framework allows systematic evaluation of different components in the agentic medical image analysis system:

- **Baseline vs Agentic**: Compare single-shot prompting with multi-turn agentic analysis
- **Reasoning Impact**: Evaluate the effect of reasoning capabilities (Grok 4.1)
- **Tool Contributions**: Analyze which visual tools contribute most to performance
- **Retrieval Augmentation**: Assess the impact of knowledge retrieval
- **Full Factorial**: Complete comparison across all dimensions

## Quick Start

### 1. Run a Complete Ablation Study

```bash
# Run baseline vs agentic comparison for all tasks
./scripts/run_nova_ablation_study.sh baseline

# Run reasoning impact analysis for diagnosis task only
./scripts/run_nova_ablation_study.sh reasoning --tasks diagnosis

# Run tool contribution analysis for localization task only
./scripts/run_nova_ablation_study.sh tools --tasks localization

# Run complete factorial study
./scripts/run_nova_ablation_study.sh full
```

### 2. Test Single Configuration

```bash
# Test baseline single-shot configuration
./scripts/run_nova_ablation_study.sh single baseline_single_shot

# Test agentic baseline with tools
./scripts/run_nova_ablation_study.sh single agentic_baseline

# Test reasoning enabled configuration
./scripts/run_nova_ablation_study.sh single reasoning_enabled
```

### 3. Direct CLI Usage

```bash
# Run baseline study using CLI directly
uv run python -m nova_retrieval_vlm.cli \
  --config-name ablation_baseline \
  task=diagnosis \
  paths.output_dir=./results/ablation/baseline_study

# Run single configuration test
uv run python -m nova_retrieval_vlm.cli \
  --config-name ablation_config \
  task=localization \
  agentic.ablation_mode=full_factorial \
  agentic.single_config=agentic_baseline \
  batch_size=1
```

## Available Configurations

### Study Modes

| Study Mode | Description | Primary Research Question |
|------------|-------------|---------------------------|
| `baseline_vs_agentic` | Compare single-shot vs multi-turn | Does agentic processing improve performance? |
| `reasoning_impact` | Reasoning enabled vs disabled | How does reasoning affect analysis? |
| `tool_contributions` | Individual tool analysis | Which tools are most valuable? |
| `retrieval_impact` | With/without knowledge retrieval | Does external knowledge help? |
| `full_factorial` | Complete comparison | What's the optimal configuration? |

### Individual Configurations

| Configuration | Tools | Reasoning | Retrieval | Multi-turn |
|---------------|-------|----------|----------|------------|
| `baseline_single_shot` | ❌ | ❌ | ❌ | ❌ |
| `no_tools` | ❌ | ❌ | ❌ | ✅ |
| `limited_visual_tools` | ✅ (basic) | ❌ | ❌ | ✅ |
| `only_visual_tools` | ✅ (visual only) | ❌ | ❌ | ✅ |
| `no_visual_tools` | ✅ (web search only) | ❌ | ❌ | ✅ |
| `agentic_baseline` | ✅ (all) | ❌ | ❌ | ✅ |
| `reasoning_enabled` | ✅ (all) | ✅ | ❌ | ✅ |
| `retrieval_only` | ❌ | ❌ | ✅ | ❌ |
| `with_retrieval` | ✅ (all) | ❌ | ✅ | ✅ |
| `reasoning_with_retrieval` | ✅ (all) | ✅ | ✅ | ✅ |

## Results and Analysis

### Output Structure

```
results/ablation_studies/
├── baseline_vs_agentic_20241129_143022/
│   ├── localization/
│   │   ├── localization_agentic_baseline_results.json
│   │   ├── localization_baseline_single_shot_results.json
│   │   └── evaluation_metrics.json
│   ├── diagnosis/
│   │   └── [similar structure]
│   ├── caption/
│   │   └── [similar structure]
│   └── analysis/
│       ├── confidence_comparison.png
│       ├── tool_usage_heatmap.png
│       ├── token_efficiency.png
│       ├── performance_comparison.csv
│       └── paper_summary.md
```

### Key Metrics Tracked

For each configuration, the system tracks:

1. **Performance Metrics**
   - Task-specific accuracy (IoU, BLEU, diagnosis accuracy)
   - Overall confidence scores
   - Calibration quality

2. **Behavioral Metrics**
   - Tool usage patterns and frequency
   - Number of turns in conversation
   - Uncertainty expression rates

3. **Efficiency Metrics**
   - Token usage
   - Execution time
   - Tool execution performance

4. **Confidence Calibration**
   - Reliability flag accuracy
   - Expected Calibration Error (ECE)
   - Confidence level analysis

### Analyzing Results

```bash
# Analyze results from a study
python scripts/analyze_ablation_results.py \
  --results-dir results/ablation_studies/baseline_vs_agentic_20241129_143022 \
  --output-dir analysis/

# Evaluate confidence calibration
python scripts/evaluate_calibration.py \
  --results-dir results/ablation_studies/baseline_vs_agentic_20241129_143022 \
  --ground-truth data/nova_ground_truth.json \
  --output-dir calibration_analysis/
```

## Configuration Customization

### Custom Study Configuration

Create a custom config file `config/custom_ablation.yaml`:

```yaml
defaults:
  - ablation_config

# Custom settings
task: localization
batch_size: 1  # Process one image at a time for detailed analysis

model:
  name: "x-ai/grok-4.1-fast:free"
  temperature: 0.0

agentic:
  ablation_mode: "full_factorial"
  enable_research_metrics: true

paths:
  output_dir: "./runs/custom_ablation"
```

### Single Configuration Testing

```bash
# Test specific tool combinations
uv run python -m nova_retrieval_vlm.cli \
  --config-name ablation_config \
  task=diagnosis \
  agentic.ablation_mode=full_factorial \
  agentic.single_config=agentic_baseline \
  agentic.enabled_tools=["zoom","crop"] \
  agentic.disabled_tools=["search_web","adjust_contrast","threshold","flip_horizontal","flip_vertical","rotate"]
```

## Research Insights

### Expected Analyses

1. **Performance Impact**
   - How much does agentic processing improve accuracy?
   - Which tasks benefit most from multi-turn analysis?

2. **Tool Effectiveness**
   - Which tools contribute most to diagnostic accuracy?
   - Are certain tools more valuable for specific tasks?

3. **Reasoning Value**
   - Does reasoning capability improve analysis quality?
   - Is the additional computational cost justified?

4. **Retrieval Benefits**
   - How does knowledge augmentation affect performance?
   - When is retrieval most valuable?

5. **Calibration Quality**
   - How well are confidence scores calibrated?
   - Can the model correctly identify uncertain cases?

### Paper-Ready Outputs

The framework generates:
- Performance comparison tables
- Tool usage heatmaps
- Confidence calibration plots
- Statistical analysis summaries
- Research-ready figures and tables

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```bash
   # Reduce batch size and add delays
   uv run python -m nova_retrieval_vlm.cli \
     task=diagnosis \
     batch_size=1 \
     max_iterations=1
   ```

2. **Memory Issues**
   ```bash
   # Use smaller batch sizes
   uv run python -m nova_retrieval_vlm.cli \
     task=localization \
     batch_size=1 \
   ```

3. **Configuration Errors**
   ```bash
   # Validate configuration
   uv run python -c "from nova_retrieval_vlm.config import Config; print('Config valid')"
   ```

### Debug Mode

Enable verbose logging:
```bash
export LOGURU_LEVEL=DEBUG
./scripts/run_nova_ablation_study.sh baseline --tasks diagnosis
```

## Integration with Existing Codebase

The ablation study framework integrates seamlessly with the existing NOVA VLM infrastructure:

- **Uses same NOVA dataset loader** from `src/nova_retrieval_vlm/data/nova_dataset.py`
- **Leverages existing evaluation metrics** in `src/nova_retrieval_vlm/evaluation/`
- **Follows same processor pattern** as other processors
- **Uses Hydra configuration system** for flexible parameter management
- **Maintains type safety** with jaxtyping and beartype

This ensures consistency with the existing codebase while adding comprehensive research capabilities.