# Usage Guide

## Installation

### Prerequisites

- Python 3.10+
- uv package manager
- OpenRouter and/or OpenAI API keys

### Install Dependencies

```bash
git clone https://github.com/your-org/nova_retrieval_vlm.git
cd nova_retrieval_vlm
uv sync
```

### Environment Setup
Create a `.env` file in the project root:
```dotenv
# OpenRouter API key (for accessing 100+ models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# OpenAI API key (optional, for OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: App identification
APP_NAME=NOVA Retrieval VLM
APP_URL=https://your-app-url.com

# Data directories
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Basic Usage

### Download Dataset
```bash
python scripts/download_nova.py --data-dir $DATA_DIR
python scripts/build_index.py  # Build retrieval indexes
```

### Command Line Interface

#### Simple Localization Task
```bash
python -m nova_retrieval_vlm.cli \
  task=localization \
  model.name=openai/gpt-4o-mini:free \
  max_iterations=5
```

#### With Retrieval Augmentation
```bash
python -m nova_retrieval_vlm.cli \
  task=localization \
  use_retrieval=true \
  retrieval.type=bm25 \
  retrieval.top_k=3 \
  model.name=openai/gpt-4o
```

#### Multi-turn Analysis
```bash
python -m nova_retrieval_vlm.cli \
  task=diagnosis \
  approach=multiturn \
  use_retrieval=true \
  model.name=anthropic/claude-3.5-sonnet
```

## Task Types

### Localization
Detect and localize anomalies in brain MRI images.
- **Metrics**: mAP@0.3, mAP@0.5, mAP@0.50-0.95
- **Output**: Bounding boxes with confidence scores

### Captioning
Generate descriptive captions for medical images.
- **Metrics**: BLEU, METEOR, BERTScore, RadGraph F1
- **Output**: Natural language descriptions

### Diagnosis
Provide differential diagnosis based on imaging findings.
- **Metrics**: Top-1/Top-5 accuracy, coverage, entropy
- **Output**: Ranked list of potential diagnoses

## Analysis Approaches

### Baseline
Single-turn analysis without retrieval.
```bash
python -m nova_retrieval_vlm.cli approach=baseline
```

### Multi-turn
Iterative reasoning with conditional continuation.
```bash
python -m nova_retrieval_vlm.cli approach=multiturn
```

### Visual Operations
Interactive visual reasoning with zoom/crop/contrast.
```bash
python -m nova_retrieval_vlm.cli approach=visual
```

### Comprehensive
All capabilities combined.
```bash
python -m nova_retrieval_vlm.cli approach=comprehensive
```

### Agentic Mode
Multi-turn reasoning with visual tools and retrieval integration.
```bash
# Enable agentic processing
python -m nova_retrieval_vlm.cli task=localization agentic.enabled=true

# With visual reasoning and tools
python -m nova_retrieval_vlm.cli task=diagnosis agentic.enabled=true agentic.use_tools=true

# Configure max turns
python -m nova_retrieval_vlm.cli task=localization agentic.enabled=true agentic.max_turns=5
```

## Retrieval Configuration

### BM25 (Keyword-based)
```bash
python -m nova_retrieval_vlm.cli \
  use_retrieval=true \
  retrieval.type=bm25 \
  retrieval.top_k=5
```

### Dense Vector
```bash
python -m nova_retrieval_vlm.cli \
  use_retrieval=true \
  retrieval.type=dense \
  retrieval.top_k=3
```

### Hybrid (Recommended)
```bash
python -m nova_retrieval_vlm.cli \
  use_retrieval=true \
  retrieval.type=hybrid \
  retrieval.hybrid_ratio=0.7 \
  retrieval.top_k=5
```

## Model Selection

### Free Models (Rate Limited)
- `openai/gpt-4o-mini:free`
- `meta-llama/llama-3.2-11b-vision-instruct:free`
- `google/gemma-2-9b-it:free`

### Premium Models
- `openai/gpt-4o` - Best performance
- `anthropic/claude-3.5-sonnet` - Excellent reasoning
- `meta-llama/llama-3.2-90b-vision-instruct` - Strong open source

See [OpenRouter Models](https://openrouter.ai/models) for complete list.

## Configuration Files

Create custom YAML configs:
```yaml
# configs/experiment.yaml
model:
  name: "openai/gpt-4o"
  temperature: 0.1

retrieval:
  type: "hybrid"
  top_k: 3

task: "diagnosis"
use_retrieval: true
max_iterations: 15
```

Use with:
```bash
python -m nova_retrieval_vlm.cli --config-path=configs --config-name=experiment
```

## Batch Processing

### Process Full Dataset
```bash
python -m nova_retrieval_vlm.cli \
  task=localization \
  max_iterations=0 \
  batch_size=8
```

### Resume Interrupted Runs
```bash
python -m nova_retrieval_vlm.cli \
  task=localization \
  skip_existing=true
```

## Visualization

### Generate Plots
```bash
python scripts/plot_results.py --input-dir runs/experiment
```

### Interactive GUI
```bash
streamlit run src/nova_retrieval_vlm/visualization/gui.py
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: OPENROUTER_API_KEY not set
   ```
   Solution: Check your `.env` file and ensure API keys are set.

2. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   Solution: Increase `request_delay` or use free models for testing.

3. **Memory Issues**
   ```
   Error: Out of memory
   ```
   Solution: Reduce `batch_size` or enable image compression.

### Debug Mode
```bash
export LOGURU_LEVEL=DEBUG
python -m nova_retrieval_vlm.cli task=localization --verbose
```

For more details, see the [main README](../README.md) and [troubleshooting section](../README.md#troubleshooting).