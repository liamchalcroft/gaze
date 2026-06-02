# GAZE

A modular Python framework for building multi-turn agentic vision-language model (VLM) systems. Built for medical image analysis but applicable to any visual reasoning task.

## Key Features

- **Multi-turn agentic loop** -- JSON-structured tool-calling with configurable turn limits, schema validation, and automatic error recovery
- **25 built-in tools** -- visual manipulation and literature search
- **Task processors** -- abstract base class with dependency injection
- **Model adapters** -- OpenAI, LM Studio, HuggingFace Transformers
- **Verifiers integration** -- reward functions for RL training

## Installation

```bash
pip install gaze-vlm
```

## Next Steps

- [Getting Started](getting-started.md) -- build your first processor
- [Architecture](architecture.md) -- understand the design
- [Examples](examples.md) -- five complete applications
- [API Reference](api/index.md) -- full API documentation
