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

## Quick Start (example)

```bash
cd examples/nova
uv sync
# Configure .env with API keys (OPENROUTER_API_KEY / OPENAI_API_KEY)
python scripts/download_nova.py --data-dir ./data/nova
uv run python -m src.cli task=localization
```

## Architecture

```text
NOVA Dataset → Data Loader → VLM Pipeline → Evaluation
                    ↓
              Retrieval System → Guidelines Index
```

See the [Usage Guide](./usage.md) for details.
