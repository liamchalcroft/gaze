# API Reference

## Core Modules

- [Base Processor](base.md) -- `AgenticProcessorBase` and the agentic loop
- [Types](types.md) -- `ToolCall`, `ToolResult`, `Turn`, `AgenticResult`
- [Configuration](config.md) -- frozen config dataclasses and `config_context()`
- [Exceptions](exceptions.md) -- `GazeError` hierarchy

## Models

- [Adapter Protocol](models/adapter_protocol.md) -- the interface all adapters implement
- [OpenAI Adapter](models/openai_adapter.md) -- OpenAI API and OpenRouter
- [LM Studio Adapter](models/lmstudio_adapter.md) -- local model inference

## Tools

- [Registry](tools/registry.md) -- `ToolRegistry` management
- [Visual Tools](tools/visual.md) -- 23 image manipulation tools
- [Search Tools](tools/search.md) -- PubMed and Open-i search

## Retrieval

- [PubMed Search](retrieval/web_search.md) -- NCBI E-utilities
- [Image Search](retrieval/image_search.md) -- Open-i medical images

## Extensions

- [Verifiers](verifiers.md) -- RL reward functions and environments
- [Utilities](utils.md) -- IoU, JSON extraction, type coercion
