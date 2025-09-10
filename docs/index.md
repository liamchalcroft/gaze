# NOVA Retrieval VLM Documentation

Welcome to the NOVA Retrieval VLM documentation.

## Quick Links
- [Getting Started](./usage.md)
- [System Prompts Guide](./system_prompts.md)
- [Enhanced Multi-turn System](./enhanced_multiturn_system.md)
- [Main README](../README.md)

## About
NOVA Retrieval VLM is a comprehensive framework for comparing vision-language models on medical imaging tasks using the NOVA brain-MRI benchmark. It supports both baseline and retrieval-augmented approaches with advanced evaluation metrics.

## Key Features
- **Multi-model Support**: OpenAI GPT models and 100+ models via OpenRouter
- **Retrieval-Augmented Generation**: BM25, dense vector, and hybrid retrieval
- **Comprehensive Evaluation**: Automated metrics for localization, captioning, and diagnosis
- **Interactive Tools**: Streamlit GUI with visualization capabilities
- **Multi-turn Reasoning**: Adaptive analysis with conditional continuation

## Quick Start
1. Install dependencies: `uv pip install -e .`
2. Configure API keys in `.env` file
3. Download dataset: `python scripts/download_nova.py`
4. Run your first analysis: `python -m nova_retrieval_vlm.cli task=localization`

## Architecture Overview
```
NOVA Dataset ’ Data Loader ’ VLM Pipeline ’ Evaluation
                   “
              Retrieval System ’ Guidelines Index
```

For detailed usage instructions and examples, see the [Usage Guide](./usage.md).