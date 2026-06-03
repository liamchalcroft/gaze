# GAZE

Modular Python framework for multi-turn agentic VLM systems in medical image analysis. Tool-augmented reasoning over radiological images with PubMed/Open-i retrieval. Python 3.10+, asyncio, OpenAI-compatible APIs (OpenRouter, LM Studio). Built with uv, ruff, pyright, beartype.

# Architecture

```
src/gaze/          # Core library
    base.py                   # AgenticProcessorBase — subclass this, implement 4 methods
    config.py                 # Frozen dataclasses: GazeConfig, SearchConfig, ImageProcessingConfig, etc.
    types.py                  # ToolCall, ToolResult, Turn, AgenticResult (all frozen)
    exceptions.py             # GazeError hierarchy — always use specific subtypes
    _frozen.py                # deep_freeze / deep_thaw utilities
    cache.py                  # TTLCache with eviction
    models/                   # AdapterProtocol, OpenAIAdapter, LMStudioAdapter, HuggingFaceAdapter
    tools/                    # Tool, ToolRegistry, visual tools (23), search tools (2)
    retrieval/                # PubMed (NCBI E-utilities), Open-i image search
    prompts/                  # Jinja2 templates via minijinja
    verifiers/                # RL integration: rewards, multi-turn env, mixin
    utils/                    # IoU, JSON extraction, JSON type coercion, confidence clamping
examples/
    nova/                     # NOVA brain-MRI benchmark (CLI + evaluation + rewards)
    gemex_thinkvg/            # GEMeX visual grounding
    agentclinic_nejm/         # Multi-turn diagnostic reasoning
    pubmedqa/                 # PubMedQA text-only QA
    vqa_rad/                  # VQA-RAD radiology VQA
environments/nova_brain_mri/  # Standalone MedMarks-compatible environment
tests/                        # ~65 test files, pytest + pytest-asyncio
```

# Development Commands

```bash
uv sync                                    # Install dependencies
make check                                 # Full quality gate: lint + format + typecheck + lockfile + tests
uv run ruff check .                        # Lint
uv run ruff format .                       # Format (auto-fix)
uv run ruff format --check .               # Format (check only)
uv run pyright src/                         # Type check core package
uv lock --check                             # Verify lockfile is in sync
uv run pytest tests/ -x --tb=short          # Run core tests
uv run pytest tests/ examples/nova/tests/ -x --tb=short  # All tests including NOVA
uv run pytest --cov=gaze --cov-report=html     # Coverage
```

# Code Standards

- Use `@beartype` on all public functions and class methods.
- Use specific exception types from `gaze.exceptions` — never bare `except:` or broad `except Exception:`. See `src/gaze/exceptions.py` for the full hierarchy.
- Config objects are frozen dataclasses. Use `config_context()` (ContextVar-based) for task-scoped overrides, never mutate globals. See `src/gaze/config.py`.
- All `ToolCall.arguments` are frozen via `deep_freeze()`. Use `deep_thaw()` before JSON serialization. See `src/gaze/_frozen.py`.
- `coerce_json_types(response, schema)` handles type mismatches from local models centrally. Do not add per-field coercion in processors. See `src/gaze/utils/json_coerce.py`.
- `clamp_confidence()` clamps out-of-range values to [0,1] but rejects NaN/inf/bool. All example validators use it.
- Ruff config: `line-length = 100`, `target-version = "py310"`, `extend-exclude = ["examples/"]`. See `pyproject.toml` for full rule set.
- Pyright: `typeCheckingMode = "basic"`. Excludes `huggingface_adapter.py`, `openai_adapter.py`, and `verifiers/`.

# Testing

- Framework: pytest + pytest-asyncio (`asyncio_mode = "strict"`)
- Markers: `unit`, `slow`, `integration` (needs real APIs), `stress`, `edge_case`, `performance`
- Default: `pytest -m "not integration"` (skips integration tests)
- Tests live in `tests/` (core) and `examples/nova/tests/` (NOVA-specific)
- Add `@pytest.mark.asyncio` on all async test functions

# Common Pitfalls

- `base.py` skips multi-turn POLICY injection when `max_turns == 1` (single-turn mode). Do not assume policy is always present.
- `base.py` injects `continue: false` when the key is missing from model output. Do not rely on models always returning it.
- LMStudioAdapter: no retries, no `response_format`, allows HTTP (not just HTTPS), 300s timeout. Context overflow detection on 400 status with "context size"/"n_ctx" in error.
- `SchemaValidationError` is a subclass of `AgenticProcessingError`, not `GazeError` directly.
- HuggingFace adapters are lazy-imported to avoid torch dependency. The `__getattr__` in `__init__.py` handles this.
- `examples/` is excluded from ruff via `extend-exclude`. Example code has its own conventions.
- GLM-4.6V and Qwen 3.5 put content in `reasoning_content` — the content-empty fallback in `base.py` handles this.
- Thinking models (Qwen 3.5) need `max_tokens >= 4096` because reasoning tokens count against the limit.
- LM Studio can only run one model at a time on current hardware. Health checks for other models cause unloads.

# Extension Points

To add a new task processor: subclass `AgenticProcessorBase`, implement `get_system_prompt()`, `get_user_message()`, `get_response_schema()`, `validate_response()`. See `examples/nova/src/processor.py` for the pattern.

To add a tool: write an async execute function (`async def _execute_foo(registry, **kwargs) -> ToolResult`), create a `Tool` instance, include it in the tools list. See `src/gaze/tools/visual.py`.

To add a model adapter: implement `AdapterProtocol` with `generate_chat()`. See `src/gaze/models/adapter_protocol.py`.
