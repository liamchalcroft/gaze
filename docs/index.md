# GAZE

A modular Python framework for building multi-turn agentic vision-language model (VLM) systems. Built for medical image analysis but applicable to any visual reasoning task.

## Key features

- **Multi-turn agentic loop** -- JSON-structured tool-calling with configurable turn limits, schema validation, and automatic error recovery
- **25 built-in tools (23 visual + 2 search)** -- visual manipulation (zoom, crop, contrast, threshold, flip, rotate, etc.) and literature/image retrieval (PubMed, Open-i)
- **Task processors** -- abstract base class with dependency injection
- **Model adapters** -- OpenAI, LM Studio, HuggingFace Transformers
- **Verifiers integration** -- reward functions for RL training

GAZE runs against cloud APIs (OpenAI, OpenRouter) or local models (LM Studio).

## Installation

```bash
pip install gaze-vlm
```

## Next steps

- [Getting started](getting-started.md) -- build your first processor
- [Architecture](architecture.md) -- understand the design
- [Examples](examples.md) -- five complete applications
- [API reference](api/index.md) -- full API documentation
