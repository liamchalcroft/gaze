# NOVA Dataset Evaluation Guide

This guide explains how to run comprehensive evaluations of the NOVA dataset using the cleaned-up architecture.

## 🚀 Quick Start

### Simple Shell Interface
The easiest way to run evaluations is using the shell wrapper:

```bash
# Interactive quick evaluation
./scripts/eval_nova.sh quick

# Baseline evaluation
./scripts/eval_nova.sh baseline localization

# Agentic evaluation with tools
./scripts/eval_nova.sh agentic diagnosis

# Retrieval-augmented evaluation
./scripts/eval_nova.sh retrieval caption

# Comprehensive benchmark
./scripts/eval_nova.sh comprehensive

# All tasks with a specific model
./scripts/eval_nova.sh all-tasks openai/gpt-4o
```

### Direct Python Interface
For more control, use the Python scripts directly:

```bash
# Single task evaluation
uv run python scripts/evaluate_nova_dataset.py \
    --task localization \
    --model openai/gpt-4o \
    --batch-size 4 \
    --output-dir ./runs/localization_eval

# Agentic evaluation
uv run python scripts/evaluate_nova_dataset.py \
    --task diagnosis \
    --model x-ai/grok-4.1-fast:free \
    --agentic \
    --use-tools \
    --max-turns 10 \
    --output-dir ./runs/diagnosis_agentic

# All tasks evaluation
uv run python scripts/evaluate_nova_dataset.py \
    --all-tasks \
    --model openai/gpt-4o \
    --output-dir ./runs/comprehensive_eval
```

## 📊 Comprehensive Benchmarking

For systematic evaluation across multiple configurations:

```bash
# Quick benchmark (single model, basic configs)
uv run python scripts/run_comprehensive_benchmark.py --preset quick

# Standard benchmark (multiple models, standard configs)
uv run python scripts/run_comprehensive_benchmark.py --preset standard

# Comprehensive benchmark (all models, all configs)
uv run python scripts/run_comprehensive_benchmark.py --preset comprehensive

# Custom models
uv run python scripts/run_comprehensive_benchmark.py \
    --models openai/gpt-4o anthropic/claude-3.5-sonnet \
    --output-dir ./runs/custom_benchmark
```

## 🎯 Evaluation Configurations

### Baseline Configuration
- **Mode**: Single-shot, no tools
- **Features**: Direct prediction in one pass
- **Use case**: Baseline comparison, fast evaluation

### Agentic Configuration
- **Mode**: Multi-turn, tool-enabled
- **Features**: Visual tools (zoom, crop, contrast), reasoning chains
- **Use case**: Advanced analysis, detailed examination

### Retrieval Configuration
- **Mode**: Knowledge-augmented
- **Features**: BM25/FAISS search, external medical knowledge
- **Use case**: Knowledge-intensive tasks

## 🔧 Configuration Options

### Task Types
- **localization**: Bounding box detection of abnormalities
- **diagnosis**: Disease classification and differential diagnosis
- **caption**: Medical report generation and findings description

### Model Options
```bash
# Default model (xAI Grok 4.1)
x-ai/grok-4.1-fast:free

# Future models to add when available:
# z-ai/glm-4.5v
# qwen/qwen3-vl-235b-a22b-instruct
# qwen/qwen3-vl-235b-a22b-thinking
# stepfun-ai/step3

# OpenRouter has many other models available
# Check https://openrouter.ai/models for full list
```

### Processing Parameters
```bash
--batch-size 4        # Process 4 samples at once
--max-turns 10        # Max 10 turns for agentic mode
--skip-existing       # Skip already processed batches
--verbose            # Detailed logging
```

### Agentic Options
```bash
--agentic            # Enable multi-turn agentic processing
--use-tools          # Enable visual and web search tools
--no-tools           # Disable tools (conversation only)
--use-retrieval       # Enable retrieval augmentation
--retrieval-type bm25 # BM25, faiss, or hybrid retrieval
```

## 📈 Output Structure

Evaluations create the following directory structure:

```
runs/
├── evaluation_timestamp/
│   ├── localization_openai_gpt-4o/
│   │   ├── config.json           # Evaluation configuration
│   │   ├── result.json           # Summary results
│   │   └── batch_*.json          # Individual batch results
│   ├── diagnosis_openai_gpt-4o/
│   └── caption_openai_gpt-4o/
└── benchmark_summary.json          # Comprehensive summary
```

### Result Format
Each evaluation generates:

- **Metrics**: Task-specific performance metrics (mAP, accuracy, F1, etc.)
- **Config**: Complete configuration used
- **Timing**: Processing time and throughput
- **Metadata**: Model information, dataset statistics

## 🔍 Performance Monitoring

### Real-time Monitoring
```bash
# Enable verbose logging for detailed progress
--verbose

# Monitor log files
tail -f runs/*/batch_*.log
```

### Resource Management
```bash
# Small batch sizes for memory-constrained environments
--batch-size 1

# Skip existing batches to resume interrupted runs
--skip-existing
```

## 🧪 Validation and Testing

### Quick Test
```bash
# Test with minimal data (first batch only)
./scripts/eval_nova.sh quick
```

### Validation
```bash
# Verify all components work
uv run python -c "
from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.cli import create_processor
config = Config()
processor = create_processor(config)
print('✅ All components validated')
"
```

## 📋 Best Practices

1. **Start Small**: Use `--quick` or `--batch-size 1` for initial testing
2. **Check Resources**: Monitor memory usage, adjust batch size accordingly
3. **Skip Existing**: Use `--skip-existing` to resume interrupted runs
4. **Verbose Mode**: Use `--verbose` for detailed progress tracking
5. **Organize Output**: Use descriptive output directory names

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure dependencies are installed
uv sync

# Check Python path
uv run python -c "import nova_retrieval_vlm; print('✅ OK')"
```

**Memory Issues**:
```bash
# Reduce batch size
--batch-size 1

# Process tasks separately
./scripts/eval_nova.sh baseline localization
./scripts/eval_nova.sh baseline diagnosis
./scripts/eval_nova.sh baseline caption
```

**Timeout Issues**:
```bash
# Process in smaller chunks
# Use --skip-existing to resume

# Monitor with verbose logging
--verbose
```

**API Key Issues**:
```bash
# Check environment variables
echo $OPENROUTER_API_KEY
echo $OPENAI_API_KEY

# Verify API access
uv run python -c "
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
adapter = OpenAIAdapter('openai/gpt-4o')
print('✅ API access OK')
"
```

## 📚 Advanced Usage

### Custom Evaluations
Create custom evaluation configurations by modifying the benchmark presets in `scripts/run_comprehensive_benchmark.py`.

### Integration with Research
The evaluation results are structured for easy integration with research pipelines:
- JSON format for programmatic analysis
- Detailed metrics for statistical comparison
- Reproducible configurations for paper methods sections

## 🎉 Next Steps

1. **Run Quick Evaluation**: `./scripts/eval_nova.sh quick`
2. **Review Results**: Check generated metrics and outputs
3. **Scale Up**: Use comprehensive benchmarks for full evaluation
4. **Analyze**: Use the structured results for research analysis

For questions or issues, check the log files or run with `--verbose` for detailed information.