# NOVA Retrieval VLM Documentation

## Quick Links

- [Getting Started](./usage.md)
- [Agentic Workflow](./agentic_workflow.md)
- [Main README](../README.md)

## About

Framework for benchmarking vision-language models on the NOVA brain-MRI dataset. Runs localization, captioning, and diagnosis with optional agentic tools and PubMed search.

## Features

- **Multi-model Support**: 100+ models via OpenRouter, direct OpenAI access
- **Agentic Processing**: Multi-turn reasoning with visual tools and PubMed search
- **Evaluation**: Automated metrics matching the NOVA benchmark protocol
- **Visualization**: Streamlit GUI and plotting tools

## Quick Start (example)

```bash
cd examples/nova
uv sync
# Configure .env with API keys (OPENROUTER_API_KEY / OPENAI_API_KEY)
uv run python -m src.cli --task localization --data-dir ./data/nova --output-dir ./runs
```

## Architecture

```text
NOVA Dataset → Data Loader → VLM Pipeline → Evaluation → Runs
```

See the [Usage Guide](./usage.md) for details.
