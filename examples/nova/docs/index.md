# NOVA Example Documentation

- [Usage Guide](./usage.md) -- setup, CLI options, data requirements
- [Agentic Workflow](./agentic_workflow.md) -- tool loop, search integration, processor architecture
- [README](../README.md) -- quick start

## Overview

NOVA VLM benchmarks vision-language models on brain-MRI analysis with three tasks: captioning, diagnosis, and localization. The agentic mode enables multi-turn reasoning with visual tools and PubMed search.

Run from the repository root:

```bash
uv sync --extra nova
uv run --extra nova python -m examples.nova.src.cli --task localization --model openai/gpt-4o --output-dir ./runs
```
