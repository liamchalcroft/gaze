# NOVA Example Documentation

- [Usage Guide](./usage.md) -- setup, CLI options, data requirements
- [Agentic Workflow](./agentic_workflow.md) -- tool loop, search integration, processor architecture
- [README](../README.md) -- quick start

## Overview

NOVA VLM benchmarks vision-language models on brain-MRI analysis with three tasks: captioning, diagnosis, and localization. The agentic mode enables multi-turn reasoning with visual tools and PubMed search.

```bash
cd examples/nova
uv sync
uv run python -m src.cli --task localization --model openai/gpt-4o --data-dir ./data/nova --output-dir ./runs
```
