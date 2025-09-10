# NOVA Retrieval VLM

A research framework for benchmarking vision-language models on medical imaging tasks using the NOVA brain-MRI dataset. This system compares baseline models against retrieval-augmented approaches across localization, captioning, and diagnosis tasks.

## Overview

The NOVA Retrieval VLM framework enables systematic evaluation of vision-language models on medical imaging tasks. It provides tools for:

- **Multi-model evaluation**: Support for OpenAI models and 100+ models via OpenRouter
- **Retrieval augmentation**: BM25, dense vector, and hybrid retrieval from medical guidelines  
- **Comprehensive metrics**: Automated evaluation for all NOVA benchmark tasks
- **Batch processing**: Efficient async processing with rate limiting and retry logic
- **Interactive analysis**: Streamlit GUI and visualization tools

## Repository Structure

```mermaid
graph TD
    A[nova_retrieval_vlm/] --> B[src/nova_retrieval_vlm/]
    A --> C[tests/]
    A --> D[scripts/]
    A --> E[docs/]
    A --> F[paper/]
    A --> G[data/]
    A --> H[runs/]
    A --> I[indexes/]
    
    B --> B1[cli.py - Main CLI interface]
    B --> B2[config.py - Configuration management]
    B --> B3[models/ - Model adapters]
    B --> B4[evaluation/ - Task metrics]
    B --> B5[retrieval/ - Retrieval systems]
    B --> B6[knowledge_base/ - Index building]
    B --> B7[processors/ - Task processors]
    B --> B8[prompts/ - Jinja2 templates]
    B --> B9[visualization/ - GUI & plotting]
    B --> B10[types.py - Type definitions]
    B --> B11[utils/ - Utilities]
    
    C --> C1[test_types.py - Type system tests]
    C --> C2[test_processors.py - Processor tests]
    C --> C3[test_integration.py - End-to-end tests]
    C --> C4[test_edge_cases_and_stress.py - Edge cases]
    
    D --> D1[Benchmark scripts]
    D --> D2[Data processing utilities]
    D --> D3[Analysis tools]
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/nova_retrieval_vlm/` | Main Python package with all core functionality |
| `tests/` | Comprehensive test suite (84+ passing tests) |
| `scripts/` | Benchmarking and data processing utilities |
| `docs/` | Documentation and usage guides |
| `paper/` | Research paper LaTeX source |
| `data/` | NOVA dataset storage |
| `runs/` | Experiment outputs and results |
| `indexes/` | Retrieval system indexes |
| `notebooks/` | Analysis and exploration notebooks |

## Installation

### Prerequisites

- Python ≥ 3.10
- uv package manager (recommended) or pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/nova_retrieval_vlm.git
   cd nova_retrieval_vlm
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Configure environment variables:**
   
   Create a `.env` file:
   ```bash
   # OpenRouter API key (for 100+ models)
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # OpenAI API key (optional)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Data directories
   DATA_DIR=./data/nova
   OUTPUT_DIR=./runs
   ```

4. **Download dataset and build indexes:**
   ```bash
   # Download NOVA dataset
   python scripts/download_nova.py --data-dir $DATA_DIR
   
   # Build retrieval indexes
   python scripts/build_index.py
   ```

## Usage

### Command Line Interface

The framework uses Hydra for configuration management, allowing flexible parameter overrides:

#### Basic Tasks

```bash
# Localization (object detection)
python -m nova_retrieval_vlm.cli \
  task=localization \
  model.name=openai/gpt-4o \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/localization

# Caption generation
python -m nova_retrieval_vlm.cli \
  task=caption \
  model.name=anthropic/claude-3.5-sonnet \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/caption

# Medical diagnosis
python -m nova_retrieval_vlm.cli \
  task=diagnosis \
  model.name=openai/gpt-4o \
  paths.data_dir=$DATA_DIR \
  paths.output_dir=runs/diagnosis
```

#### Retrieval-Augmented Tasks

```bash
# Localization with BM25 retrieval
python -m nova_retrieval_vlm.cli \
  task=localization \
  use_retrieval=true \
  retrieval.type=bm25 \
  retrieval.top_k=5 \
  model.name=openai/gpt-4o

# Caption with hybrid retrieval
python -m nova_retrieval_vlm.cli \
  task=caption \
  use_retrieval=true \
  retrieval.type=hybrid \
  retrieval.top_k=3 \
  model.name=anthropic/claude-3.5-sonnet

# Multi-turn diagnosis with retrieval
python -m nova_retrieval_vlm.cli \
  task=diagnosis \
  approach=multiturn \
  use_retrieval=true \
  model.name=openai/gpt-4o
```

#### Batch Processing

```bash
# Process entire dataset
python -m nova_retrieval_vlm.cli \
  task=localization \
  max_iterations=0 \
  batch_size=8 \
  model.name=openai/gpt-4o

# Run complete benchmark suite
bash scripts/run_full_benchmarks.sh
```

### Interactive Analysis

Launch the Streamlit GUI for interactive exploration:

```bash
streamlit run src/nova_retrieval_vlm/visualization/gui.py
```

### Model Support

The framework supports all OpenRouter models and direct OpenAI access:

**Popular choices:**
- `openai/gpt-4o` - Latest GPT-4 Omni
- `anthropic/claude-3.5-sonnet` - Latest Claude model
- `meta-llama/llama-3.2-90b-vision-instruct` - Open source vision model
- `openai/gpt-4o-mini:free` - Free tier option

See [OpenRouter Models](https://openrouter.ai/models) for the complete list.

## Configuration

### Hydra Configuration System

Override any parameter via command line:

```bash
# Model settings
python -m nova_retrieval_vlm.cli \
  model.temperature=0.3 \
  model.max_tokens=2048 \
  model.timeout=120

# Retrieval settings  
python -m nova_retrieval_vlm.cli \
  retrieval.type=hybrid \
  retrieval.top_k=5 \
  retrieval.hybrid_ratio=0.7

# Processing settings
python -m nova_retrieval_vlm.cli \
  batch_size=4 \
  max_iterations=10 \
  request_delay=2.0
```

### Custom Configuration Files

Create YAML configurations for repeated experiments:

```yaml
# config/experiment.yaml
model:
  name: "openai/gpt-4o"
  temperature: 0.1

retrieval:
  type: "hybrid"
  top_k: 3

task: "diagnosis"
use_retrieval: true
batch_size: 2
```

```bash
python -m nova_retrieval_vlm.cli --config-path=config --config-name=experiment
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Localization**: mAP@0.3, mAP@0.5, mAP@0.75 using torchmetrics
- **Captioning**: BLEU, BERTScore, RadGraph F1, METEOR scores  
- **Diagnosis**: Top-1/Top-5 accuracy, F1, precision, recall

Results are automatically saved with detailed logs and can be visualized using built-in plotting tools.

## Development

### Running Tests

The framework includes a comprehensive test suite with 84+ passing tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=nova_retrieval_vlm

# Run specific test categories
uv run python scripts/run_comprehensive_tests.py --mode=unit
uv run python scripts/run_comprehensive_tests.py --mode=integration

# Run performance tests
uv run python scripts/run_comprehensive_tests.py --mode=stress
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code  
ruff check .

# Type checking
pyright src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the full test suite
5. Submit a pull request

## Next Steps & Improvements

### Research Extensions

**Multi-Modal Retrieval Enhancement**
- Implement cross-modal retrieval combining text and image embeddings
- Explore graph-based retrieval using medical knowledge graphs
- Investigate attention-based fusion mechanisms for retrieved content

**Model Architecture Innovation**  
- Fine-tune specialized medical VLMs on domain-specific data
- Develop multi-turn reasoning chains with self-correction mechanisms
- Implement uncertainty quantification for model predictions

**Evaluation Framework Enhancement**
- Conduct human expert validation studies with radiologists
- Implement statistical significance testing with multiple comparison corrections
- Develop interpretability analysis tools for model decision processes

### Technical Improvements

**Performance Optimization**
- Implement streaming inference for processing large datasets
- Add GPU memory optimization with dynamic batching
- Develop distributed processing capabilities across multiple nodes

**System Architecture**
- Add model serving optimization with TorchScript/ONNX conversion  
- Implement advanced caching strategies for retrieval results
- Build real-time monitoring and alerting systems

**User Experience Enhancement**
- Create web-based dashboard for experiment management
- Add interactive result visualization with drill-down capabilities  
- Implement automated report generation with statistical analysis

### Infrastructure Development

**Deployment & Operations**
- Container orchestration with Kubernetes for scalable deployment
- CI/CD pipeline optimization with automated testing and deployment
- Production monitoring with metrics, logging, and tracing

**Data Management**
- Implement versioned dataset management system
- Add data quality validation and monitoring tools
- Create automated data pipeline for new medical imaging datasets

### Research Applications

**Clinical Integration**
- Develop clinical decision support system integration
- Implement DICOM compatibility for hospital system integration  
- Create regulatory compliance framework for medical AI applications

**Cross-Domain Generalization**
- Extend framework to other medical imaging modalities (CT, X-ray, ultrasound)
- Implement domain adaptation techniques for cross-dataset generalization
- Develop few-shot learning capabilities for rare conditions

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{nova_retrieval_vlm,
  title={Retrieval-Augmented Vision-Language Models for Medical Imaging Analysis},
  author={Research Team},
  journal={Medical Image Analysis},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.