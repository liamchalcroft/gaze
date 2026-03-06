# Radiant Harness - Project Guide for Claude

## Project Overview

Radiant Harness is a modular framework for building multi-turn agentic VLM systems for medical image analysis. It provides tool-augmented reasoning over radiological images.

The primary example implementation is `examples/nova/` -- a benchmark for VLMs on the NOVA brain-MRI dataset. Additional examples exist for GEMeX visual grounding, AgentClinic diagnostic reasoning, PubMedQA, and VQA-RAD (each with its own CLI, processor, dataset, evaluation, and schemas).

## Project Structure

```
src/radiant_harness/
    __init__.py                 # Public API exports
    __main__.py                 # CLI entry point (info only)
    _frozen.py                  # deep_freeze utility for immutable config
    base.py                     # AgenticProcessorBase, ImageInput
    config.py                   # HarnessConfig, AgenticConfig, SearchConfig, etc.
    types.py                    # ToolCall, ToolResult, Turn, AgenticResult
    exceptions.py               # HarnessError hierarchy
    cache.py                    # TTLCache
    models/
        adapter_protocol.py     # AdapterProtocol, GenerationLog
        openai_adapter.py       # OpenAIAdapter
        lmstudio_adapter.py     # LMStudioAdapter (local LM Studio server)
        huggingface_adapter.py  # HuggingFaceAdapter, HuggingFaceVLMAdapter (optional)
    tools/
        tool.py                 # Tool class
        registry.py             # ToolRegistry, ToolDocumenter, EncodedImage
        visual.py               # zoom, crop, adjust_contrast, adjust_brightness, adjust_sharpness, threshold, window_level, equalize_histogram, adaptive_equalize, detect_edges, denoise, morphological, get_intensity_stats, intensity_profile, symmetry_diff, invert, annotate_region, flip_horizontal, flip_vertical, rotate, show_grid, measure, reset
        search.py               # search_web (PubMed), search_images (Open-i)
        image_manager.py        # Image loading, transformation state, reset-to-original
    retrieval/
        base.py                 # Base search engine class
        web_search.py           # PubMed search via NCBI E-utilities
        image_search.py         # NIH Open-i image search
    prompts/
        __init__.py             # Jinja2 template loading (minijinja)
    verifiers/
        adapter.py              # RadiantHarnessAdapter
        base.py                 # BaseMultiTurnEnv
        mixin.py                # VerifiableProcessorMixin
        rewards.py              # ExactMatchReward, TokenF1Reward, IoUReward, CombinedReward
    utils/
        iou.py                  # compute_iou
        json_extract.py         # extract_json_from_text
examples/
    nova/                       # NOVA brain-MRI benchmark (full CLI + evaluation)
    gemex_thinkvg/              # GEMeX visual grounding with RL rewards
    agentclinic_nejm/           # Multi-turn diagnostic reasoning
    pubmedqa/                   # PubMedQA (CLI + processor + evaluation)
    vqa_rad/                    # VQA-RAD (CLI + processor + evaluation)
environments/
    nova_brain_mri/             # MedMarks-compatible NOVA evaluation environment
tests/
docs/
    verifiers_integration.md
    MEDMARKS_INTEGRATION.md
pyproject.toml
AUDIT_REPORT.md
Makefile
```

## Key Commands

```bash
# Install
uv sync

# Run all quality checks
make check

# Individual checks
uv run ruff check .
uv run ruff format .
uv run pyright

# Tests
uv run pytest
uv run pytest --cov=radiant_harness --cov-report=html

# NOVA example
cd examples/nova
uv run python -m src.cli --task localization --model openai/gpt-4o --data-dir ./data/nova --output-dir ./runs
```

## Technology Stack

- Python 3.10+, asyncio
- OpenAI API (OpenRouter, OpenAI) via `openai` SDK, plus LM Studio for local models
- PubMed (NCBI E-utilities), Open-i image search
- minijinja for templating
- beartype for runtime type validation
- uv, ruff, pyright for dev tooling

## Architecture

### AgenticProcessorBase

Dependency injection pattern: subclasses provide task-specific prompts, schemas, and validation. The base class handles the agentic loop, tool execution, and conversation management.

Abstract methods to implement:
- `get_system_prompt(images: list[ImageInput], metadata: dict) -> str`
- `get_user_message(images: list[ImageInput], metadata: dict) -> str`
- `get_response_schema() -> dict | None`
- `validate_response(response: dict) -> bool`
- `calculate_confidence(response: dict) -> float` (optional, default returns 0.0)

Entry point: `await processor.analyze(images, metadata, image_labels)` returns `AgenticResult`.

### Tool System

Tools registered via `ToolRegistry`:
- **Visual**: zoom, crop, adjust_contrast, adjust_brightness, adjust_sharpness, threshold, window_level, equalize_histogram, adaptive_equalize, detect_edges, denoise, morphological, get_intensity_stats, intensity_profile, symmetry_diff, invert, annotate_region, flip_horizontal, flip_vertical, rotate, show_grid, measure, reset
- **Search**: search_web (PubMed), search_images (Open-i)

### Response Format

Models return JSON each turn with a `continue` boolean:
- `true` -- model wants another turn
- `false` -- final response

## Code Standards

1. **uv** for all dependency management
2. **ruff** for linting and formatting
3. **pyright** for type checking
4. **beartype** for runtime validation (`@beartype` decorator)
5. **Fail fast** -- use specific exceptions from `radiant_harness.exceptions`, never bare `except:` or broad `except Exception:`
6. **No slop** -- no overly defensive code, no silent fallbacks

## Common Tasks

### Adding a Task Processor

1. Subclass `AgenticProcessorBase`
2. Implement `get_system_prompt`, `get_user_message`, `get_response_schema`, `validate_response`
3. Add `@beartype` decorators
4. Optionally mix in `VerifiableProcessorMixin` for RL integration

### Adding a Tool

1. Write async execute function: `async def _execute_my_tool(registry: ToolRegistry, **kwargs) -> ToolResult`
2. Create `Tool` instance with name, description, parameters schema, execute function
3. Include in the tools list passed to `ToolRegistry`

### Adding a Model Adapter

1. Implement `AdapterProtocol` from `radiant_harness.models`
2. Implement `generate_chat()` returning `(content, tool_calls, GenerationLog)`

## Configuration

Frozen dataclasses in `config.py`:
- `HarnessConfig` -- root config
- `AgenticConfig` -- turn limits, temperature, max tokens
- `SearchConfig` -- timeouts, retries, rate limiting, NCBI/Open-i URLs
- `CacheConfig` -- size, TTL, eviction
- `ImageProcessingConfig` -- image dimension limits, zoom/contrast bounds

Access via `get_config()` / `set_config()`.

## API Keys

- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` -- model API
- `NCBI_API_KEY` (optional) -- PubMed search
- `NCBI_EMAIL` (optional) -- PubMed API compliance
