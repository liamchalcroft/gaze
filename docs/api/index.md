# API reference

Generated documentation for the `gaze` package, organised by area. Each page
renders the module's public classes and functions from their docstrings.

## Core modules

- [Base processor](base.md): `AgenticProcessorBase` and the agentic loop
- [Types](types.md): `ToolCall`, `ToolResult`, `Turn`, `AgenticResult`
- [Configuration](config.md): frozen config dataclasses and `config_context()`
- [Exceptions](exceptions.md): `GazeError` hierarchy
- [Prompts](prompts.md): Jinja template loading and rendering

## Models

- [Adapter protocol](models/adapter_protocol.md): the interface all adapters implement
- [OpenAI adapter](models/openai_adapter.md): OpenAI API and OpenRouter
- [LM Studio adapter](models/lmstudio_adapter.md): local model inference
- [HuggingFace adapter](models/huggingface_adapter.md): local torch/transformers models

## Tools

- [Tool registry](tools/registry.md): `ToolRegistry` management
- [Visual tools](tools/visual.md): 23 image manipulation tools
- [Search tools](tools/search.md): PubMed and Open-i search

## Retrieval

- [PubMed search](retrieval/web_search.md): NCBI E-utilities
- [Image search](retrieval/image_search.md): Open-i medical images

## Extensions

- [Verifiers integration](verifiers.md): RL reward functions and environments
- [Utilities](utils.md): IoU, JSON extraction, type coercion
