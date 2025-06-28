# NOVA Retrieval VLM

A comprehensive framework for comparing baseline and retrieval-augmented vision-language models on the NOVA brain-MRI benchmark. This framework supports both OpenAI and OpenRouter models with advanced retrieval capabilities for medical imaging analysis.

## Features

- 🧠 **NOVA Dataset Integration**: Seamless integration with the NOVA brain-MRI benchmark dataset
- 🔍 **Retrieval-Augmented Generation**: BM25, dense vector, and hybrid retrieval from medical guidelines
- 🤖 **Multi-Model Support**: Compatible with OpenAI GPT models and 100+ models via OpenRouter
- 📊 **Comprehensive Evaluation**: Automated metrics for localization, captioning, and diagnosis tasks
- 🎨 **Visualization Tools**: Rich visualization capabilities with overlay support
- ⚡ **Async Processing**: High-performance async processing with rate limiting and retry logic
- 🖼️ **Streamlit Demo**: Interactive GUI with collapsible reasoning traces
- 🔄 **Enhanced Multi-turn Prompting**: Intelligent conditional continuation for adaptive analysis

## Quick Start

### Prerequisites

- Python ≥ 3.9
- uv (recommended) or pip
- OpenRouter and/or OpenAI API keys

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/nova_retrieval_vlm.git
   cd nova_retrieval_vlm
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv pip install -e .

   # Or using pip
   pip install -e .
   ```

3. **Configure environment variables:**
   
   Create a `.env` file in the project root:
   ```dotenv
   # OpenRouter API key (for accessing 100+ models)
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # OpenAI API key (optional, for OpenAI models)
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional: App identification for OpenRouter rankings
   APP_NAME=NOVA Retrieval VLM
   APP_URL=https://your-app-url.com

   # Data and output directories
   DATA_DIR=./data/nova
   OUTPUT_DIR=./runs
   ```

4. **Load environment variables:**
   ```bash
   # Export variables for current session
   export $(grep -v '^#' .env | xargs)
   
   # Or use direnv for automatic loading
   direnv allow
   ```

## API Keys Setup

### OpenRouter (Recommended)

OpenRouter provides access to 100+ AI models through a unified API with automatic fallbacks and cost optimization:

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Generate an API key from your dashboard
3. Add to your `.env` file as `OPENROUTER_API_KEY`

**Benefits:**
- Access to latest models from OpenAI, Anthropic, Google, Meta, and more
- Automatic fallbacks and load balancing
- Cost optimization and transparent pricing
- No vendor lock-in

See [OpenRouter Documentation](https://openrouter.ai/docs) for more details.

### OpenAI (Optional)

For direct OpenAI model access:

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Generate an API key
3. Add to your `.env` file as `OPENAI_API_KEY`

See [OpenAI Documentation](https://platform.openai.com/docs/) for more details.

## Dataset Setup

Download and prepare the NOVA dataset:

```bash
# Download NOVA dataset from Hugging Face
python scripts/download_nova.py --data-dir $DATA_DIR

# Build retrieval indexes for guidelines
python scripts/build_index.py
```

## Usage

### Command Line Interface

The framework provides a flexible CLI powered by Hydra for configuration management:

#### Basic Tasks

```bash
# Localization task (no retrieval)
python -m nova_retrieval_vlm.cli \
  task=localization \
  model.name=openai/gpt-4o \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/localization

# Localization with retrieval augmentation
python -m nova_retrieval_vlm.cli \
  task=localization \
  use_retrieval=true \
  retrieval.type=bm25 \
  retrieval.top_k=5 \
  model.name=openai/gpt-4o \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/localization_retrieval

# Caption generation
python -m nova_retrieval_vlm.cli \
  task=caption \
  model.name=anthropic/claude-3.5-sonnet \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/caption

# Diagnosis task with retrieval
python -m nova_retrieval_vlm.cli \
  task=diagnosis \
  use_retrieval=true \
  retrieval.type=hybrid \
  retrieval.top_k=3 \
  model.name=meta-llama/llama-3.2-90b-vision-instruct \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/diagnosis_retrieval
```

#### Enhanced Multi-turn Analysis

The framework supports intelligent multi-turn prompting that adapts to case complexity:

```bash
# Multi-turn analysis with conditional continuation
python -m nova_retrieval_vlm.cli \
  task=diagnosis \
  approach=multiturn \
  use_retrieval=true \
  model.name=openai/gpt-4o \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/multiturn_diagnosis

# Multi-turn captioning
python -m nova_retrieval_vlm.cli \
  task=caption \
  approach=multiturn \
  model.name=anthropic/claude-3.5-sonnet \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/multiturn_caption
```

**Multi-turn Benefits:**
- **Adaptive Analysis**: Simple cases complete in 1 step, complex cases get 2-3 steps
- **Efficient Processing**: Only uses necessary computational resources
- **Comprehensive Tracking**: Complete audit trail of analysis decisions
- **Confidence Calibration**: Model self-assesses when additional analysis is needed

See [Enhanced Multi-turn System Documentation](./docs/enhanced_multiturn_system.md) for detailed information.

#### Visualization and Analysis

```bash
# Generate visualizations with overlays
python -m nova_retrieval_vlm.cli \
  task=visualize \
  visualization.num_samples=10 \
  visualization.overlay=true \
  paths.output_dir=out/viz

# Interactive prompt testing (no image required)
python -m nova_retrieval_vlm.cli \
  task=prompt \
  prompt_text="Describe the key features of brain MRI analysis." \
  model.name=openai/gpt-4o
```

##### Streamlit GUI

Launch the interactive demo to explore predictions with collapsible reasoning traces:

```bash
streamlit run src/nova_retrieval_vlm/visualization/gui.py
```

#### Batch Processing

```bash
# Run complete experiment suite
python scripts/run_experiments.sh

# Process entire dataset (remove max_iterations limit)
python -m nova_retrieval_vlm.cli \
  task=localization \
  max_iterations=0 \
  batch_size=8 \
  model.name=openai/gpt-4o
```

## Supported Models

### OpenRouter Models (100+ available)

The framework supports all OpenRouter models. Popular choices include:

**GPT Models:**
- `openai/gpt-4o` - Latest GPT-4 Omni
- `openai/gpt-4o-mini` - Cost-effective GPT-4
- `openai/gpt-4-turbo` - GPT-4 Turbo

**Claude Models:**
- `anthropic/claude-3.5-sonnet` - Latest Claude model
- `anthropic/claude-3-opus` - Most capable Claude
- `anthropic/claude-3-haiku` - Fastest Claude

**Open Source Models:**
- `meta-llama/llama-3.2-90b-vision-instruct` - Meta's vision model
- `google/gemini-pro-1.5` - Google's multimodal model
- `mistralai/pixtral-12b` - Mistral's vision model

**Free Models:**
- `openai/gpt-4o-mini:free` - Free tier GPT-4
- `google/gemma-2-9b-it:free` - Free Gemma model
- `meta-llama/llama-3.2-11b-vision-instruct:free` - Free LLaMA vision

See the [OpenRouter Models page](https://openrouter.ai/models) for the complete list with pricing and capabilities.

### Model Selection Guidelines

**For Research/Accuracy:**
- `openai/gpt-4o` or `anthropic/claude-3.5-sonnet`

**For Cost-Effectiveness:**
- `openai/gpt-4o-mini` or `anthropic/claude-3-haiku`

**For Open Source:**
- `meta-llama/llama-3.2-90b-vision-instruct`

**For Free Usage:**
- `openai/gpt-4o-mini:free` (rate limited)

## Configuration

### Hydra Configuration

The framework uses Hydra for flexible configuration management. You can override any setting via command line:

```bash
# Model configuration
python -m nova_retrieval_vlm.cli \
  model.name=openai/gpt-4o \
  model.temperature=0.3 \
  model.max_tokens=2048 \
  model.timeout=120

# Retrieval configuration
python -m nova_retrieval_vlm.cli \
  use_retrieval=true \
  retrieval.type=bm25 \
  retrieval.top_k=5

# Processing configuration
python -m nova_retrieval_vlm.cli \
  batch_size=4 \
  max_iterations=10 \
  request_delay=2.0 \
  strict_mode=false
```

### Configuration Files

Create custom configuration files in YAML format and reference them:

```yaml
# configs/experiment.yaml
model:
  name: "openai/gpt-4o"
  temperature: 0.1
  max_tokens: 1024

retrieval:
  type: "hybrid"
  top_k: 3
  hybrid_ratio: 0.7

task: "diagnosis"
use_retrieval: true
batch_size: 2
max_iterations: 5
```

```bash
# Use custom config
python -m nova_retrieval_vlm.cli --config-path=configs --config-name=experiment
```

## Advanced Features

### Retrieval Types

1. **BM25 (Keyword-based)**: Fast, interpretable retrieval based on term matching
2. **Dense Vector**: Semantic retrieval using sentence transformers
3. **Hybrid**: Combines BM25 and dense retrieval with configurable weighting

### Multi-turn Analysis Approaches

1. **Baseline**: Single-step analysis with optional retrieval
2. **Multi-turn**: Adaptive 1-3 step analysis with conditional continuation
3. **Visual Multi-turn**: Advanced visual reasoning with operations

### Rate Limiting and Error Handling

- Automatic retry logic with exponential backoff
- Rate limiting to respect API quotas
- Graceful error handling and logging
- Resume capability for interrupted runs

### Performance Optimization

- Async processing for improved throughput
- Image compression and optimization
- Batch processing support
- Intelligent caching

## Evaluation and Metrics

The framework provides comprehensive evaluation metrics:

- **Localization**: mAP@0.3, mAP@0.5 and mAP@0.50-0.95
- **Captioning**: SacreBLEU, BERTScore, RadGraph F1, METEOR and simple
  keyword-based F1 scores (modality/clinical/binary)
- **Diagnosis**: Top-1 and Top-5 accuracy, prediction coverage and entropy

Results are automatically saved with detailed logs and can be visualized using the built-in plotting tools:

```bash
# Generate evaluation plots
python scripts/plot_results.py --input-dir runs/experiment --output-dir plots/
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nova_retrieval_vlm

# Run specific test file
pytest tests/test_models.py

# Test enhanced multi-turn system
python scripts/test_enhanced_multiturn.py
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
ruff check .

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are correctly set in the `.env` file
2. **Rate Limiting**: Increase `request_delay` parameter or use free models for testing
3. **Memory Issues**: Reduce `batch_size` or enable image compression
4. **Model Not Found**: Check model name against [OpenRouter Models](https://openrouter.ai/models)

### Debugging

Enable debug logging:

```bash
# Set log level
export LOGURU_LEVEL=DEBUG

# Run with verbose output
python -m nova_retrieval_vlm.cli task=localization --verbose
```

### Support

- 📖 Check the [Documentation](./docs/)
- 🐛 Report issues on [GitHub](https://github.com/your-org/nova_retrieval_vlm/issues)
- 💬 Join our [Discord Community](https://discord.gg/your-server)

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{nova_retrieval_vlm,
  title={Retrieval-Augmented Vision-Language Models for Medical Imaging Analysis},
  author={Your Name},
  journal={Medical Image Analysis},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---

**External Documentation:**
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [NOVA Dataset](https://huggingface.co/datasets/Ano-2090/Nova)

## Prompt Templates and Reasoning Approaches

The framework ships with two families of Jinja templates under
`nova_retrieval_vlm/src/nova_retrieval_vlm/prompts/`:

| Folder | Purpose | Files |
|--------|---------|-------|
| `baseline/` | **Baseline, single-turn** prompts (no guideline retrieval). | `localization.jinja`, `caption.jinja`, `diagnosis.jinja` |
| `multiturn/` | **Clinician-style, multi-turn** reasoning chain. | `step1.jinja` (observations + differential), `step2.jinja` (guideline-aware final decision with bboxes). |
| `visual_ops/` | **Interactive visual reasoning** with zoom/crop/contrast adjustment. | `step1.jinja` (request operations), `step2.jinja` (analysis after operations). |
| `visual_multiturn/` | **Multi-turn with retrieval and visual ops**. | `ops_request.jinja` plus all multiturn templates. |

The CLI flag `approach` controls which family is used:

```bash
# One-shot baseline (default)
python -m nova_retrieval_vlm.cli task=localization approach=baseline

# Two-turn reasoning with guideline integration
python -m nova_retrieval_vlm.cli task=localization approach=multiturn use_retrieval=true

# Multi-turn with visual adjustments and retrieval
python -m nova_retrieval_vlm.cli task=localization approach=visual_multiturn use_retrieval=true
```

`multiturn/step1.jinja` asks the model for qualitative observations and a short differential diagnosis.  Those diagnoses are fed into the guideline retriever; the top-*k* passages are then injected into `multiturn/step2.jinja`, which requests the final JSON output (bounding boxes / labels / scores) under strict formatting rules.

All prompts automatically embed the image dimensions and coordinate conventions so the model knows the valid range for `(x1, y1, x2, y2)`.

### Benchmark Scripts and Ablation

The `scripts/` directory contains helper bash scripts for running controlled benchmarks across all models and tasks.  Each script accepts optional
`--data-dir`, `--output-dir`, `--batch-size`, and `--max-iters` flags.

| Script | Purpose |
|--------|---------|
| `run_baseline_benchmark.sh` | Single-turn baseline without retrieval |
| `run_retrieval_benchmark.sh` | Baseline prompts augmented with guideline retrieval |
| `run_multiturn_benchmark.sh` | Clinician-style multi-turn reasoning with retrieval |
| `run_visual_benchmark.sh` | Multi-turn reasoning with visual adjustments and retrieval |
| `run_full_benchmarks.sh` | Executes all of the above in sequence |

These scripts provide a clear stepped ablation over the baseline prompt.
