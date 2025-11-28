# NOVA Retrieval VLM Documentation

## Quick Links

- [Getting Started](./usage.md)
- [Agentic Workflow](./agentic_workflow.md)
- [System Prompts](./system_prompts.md)
- [Prompt System](./prompt_system.md)
- [Multi-turn System](./multiturn_system.md)
- [Main README](../README.md)

## About

Framework for benchmarking vision-language models on the NOVA brain-MRI dataset. Compares baseline and retrieval-augmented approaches across localization, captioning, and diagnosis tasks.

## Features

- **Multi-model Support**: 100+ models via OpenRouter, direct OpenAI access
- **Retrieval Augmentation**: BM25, dense vector, and hybrid retrieval
- **Agentic Processing**: Multi-turn reasoning with visual tools and retrieval integration
- **Evaluation**: Automated metrics matching the NOVA benchmark protocol
- **Visualization**: Streamlit GUI and plotting tools

## Quick Start

```bash
uv sync
# Configure .env with API keys
python scripts/download_nova.py
python -m nova_retrieval_vlm.cli task=localization
```

## Architecture

```
NOVA Dataset → Data Loader → VLM Pipeline → Evaluation
                    ↓
              Retrieval System → Guidelines Index
```

See the [Usage Guide](./usage.md) for details.