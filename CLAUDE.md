# Radiant Harness - Project Guide for Claude

## Project Overview
Radiant Harness is a modular framework for building multi-turn agentic vision-language model systems for medical image analysis. It provides core infrastructure for tool-augmented reasoning over radiological images.

The repository also includes `examples/nova/` - a complete example implementation for benchmarking VLMs on the NOVA brain-MRI dataset.

## Quick Reference

### Project Structure
```
radiant-harness/
├── src/radiant_harness/        # Main Python package
│   ├── __init__.py             # Public API exports
│   ├── __main__.py             # CLI entry point
│   ├── base.py                 # AgenticProcessorBase abstract class
│   ├── config.py               # Configuration dataclasses
│   ├── types.py                # Core types (ToolCall, ToolResult, Turn, AgenticResult)
│   ├── protocols.py            # Protocol definitions
│   ├── exceptions.py           # Exception hierarchy
│   ├── cache.py                # TTLCache implementation
│   ├── models/                 # Model adapters
│   │   ├── adapter_protocol.py # AdapterProtocol
│   │   ├── openai_adapter.py   # OpenAI API adapter
│   │   └── huggingface_adapter.py # HuggingFace adapter (optional)
│   ├── tools/                  # Tool system
│   │   ├── tool.py             # Tool class definition
│   │   ├── registry.py         # ToolRegistry, EncodedImage
│   │   ├── visual.py           # Visual tools (zoom, crop, contrast, etc.)
│   │   ├── search.py           # Search tools (web, image)
│   │   ├── image_manager.py    # Image loading and transformation
│   │   ├── image_ops.py        # Image operations
│   │   ├── decorators.py       # Tool decorator helpers
│   │   └── tool_documenter.py  # Schema generation
│   ├── retrieval/              # External search integrations
│   │   ├── web_search.py       # PubMed search
│   │   └── image_search.py     # Open-i image search
│   ├── prompts/                # Template loading utilities
│   │   └── __init__.py         # Jinja template loading
│   ├── verifiers/              # RL training integration
│   │   ├── adapter.py          # RadiantHarnessAdapter
│   │   ├── base.py             # BaseMultiTurnEnv
│   │   ├── mixin.py            # VerifiableProcessorMixin
│   │   ├── rewards.py          # Reward functions
│   │   └── tool_bridge.py      # Tool execution bridge
│   └── utils/
│       ├── iou.py              # IoU calculation
│       └── json_extract.py     # JSON extraction
├── examples/                   # Example implementations
│   ├── nova/                   # NOVA brain-MRI benchmark
│   ├── gemex_thinkvg/          # Visual grounding with RL
│   ├── agentclinic_nejm/       # Diagnostic reasoning
│   ├── pubmedqa/               # Medical Q&A
│   └── vqa_rad/                # Radiology VQA
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml              # Project configuration
├── AUDIT_LOG.md                # Audit tracking
└── README.md
```

## Key Commands

### Development Commands
```bash
# Environment management (REQUIRED - use uv exclusively)
uv sync
uv run python -m radiant_harness

# Run all quality checks (recommended)
make check

# Run tests
uv run pytest
uv run pytest --cov=radiant_harness --cov-report=html

# Code quality checks (individual)
uv run ruff check .          # Linting
uv run ruff format .         # Formatting
uv run pyright               # Type checking

# Pre-commit hooks
pre-commit run --all-files
```

### Running NOVA Examples
```bash
# Run from examples/nova directory or use full paths
cd examples/nova
python -m src.cli task=localization model.name=openai/gpt-4o
```

## Technology Stack
- **Core**: Python 3.10+, asyncio
- **Models**: OpenAI API compatible (OpenRouter, OpenAI)
- **Web Search**: PubMed, Open-i integration
- **Templating**: minijinja
- **Type Safety**: beartype (runtime validation)
- **Development**: uv, ruff, pyright

## Important Files
- `src/radiant_harness/__init__.py` - Public API
- `src/radiant_harness/base.py` - AgenticProcessorBase abstract class
- `src/radiant_harness/config.py` - Configuration dataclasses
- `src/radiant_harness/types.py` - Core type definitions
- `src/radiant_harness/tools/registry.py` - ToolRegistry
- `src/radiant_harness/tools/visual.py` - Visual tool implementations
- `src/radiant_harness/models/openai_adapter.py` - OpenAI API adapter
- `src/radiant_harness/verifiers/` - RL training integration (verifiers package)

## Architecture & Design Principles

### Core Pattern: AgenticProcessorBase
The harness follows a dependency injection pattern where task-specific details (prompts, schemas, validation) are provided by subclasses while the core agentic loop, tool execution, and conversation management are handled by the base class.

```python
from radiant_harness import AgenticProcessorBase, ToolRegistry

class MyProcessor(AgenticProcessorBase):
    def get_system_prompt(self, images, metadata) -> str:
        return "You are a medical imaging expert..."

    def get_user_message(self, images, metadata) -> str:
        return f"Analyze this scan. History: {metadata.get('history')}"

    def get_response_schema(self) -> dict | None:
        return {"type": "json_schema", ...}

    def validate_response(self, response) -> bool:
        return "findings" in response
```

### Tool System
Tools are registered via ToolRegistry and can be:
- **Visual tools**: zoom, crop, contrast, threshold, flip, rotate, reset
- **Search tools**: search_web (PubMed), search_images (Open-i)

### Response Format
Models must return JSON with a `continue` field:
- `continue: true` - Model needs another turn
- `continue: false` - Model is done, response is final

## Code Standards (ENFORCED)

### Required Practices
1. **uv**: All dependency management via `uv sync`, `uv add`, `uv run`
2. **ruff**: Linting and formatting
3. **pyright**: Type checking
4. **beartype**: Runtime validation with `@beartype` decorator
5. **Fail fast**: Use proper exceptions instead of silent fallbacks
6. **No AI slop**: No "robust" parsers, no overly defensive code

### Exception Handling
- Use specific exception types from `radiant_harness.exceptions`
- Never use bare `except:` or broad `except Exception:`
- Let exceptions propagate - don't swallow errors

## Common Development Tasks

### Adding a New Task Processor
1. Create processor subclassing `AgenticProcessorBase`
2. Implement abstract methods: `get_system_prompt`, `get_user_message`, `get_response_schema`, `validate_response`
3. Add `@beartype` decorators
4. Create prompt templates if needed

### Adding a New Tool
1. Create async execute function: `async def _execute_my_tool(registry: ToolRegistry, **kwargs) -> ToolResult`
2. Create Tool instance with parameters schema
3. Register with ToolRegistry

### Adding a New Model Adapter
1. Implement `AdapterProtocol` from `radiant_harness.models`
2. Implement `generate_chat()` method
3. Return `(content, tool_calls, GenerationLog)` tuple

## Configuration
Configuration uses frozen dataclasses in `config.py`:
- `HarnessConfig` - Root configuration
- `AgenticConfig` - Agentic processing settings
- `SearchConfig` - Search operation settings
- `CacheConfig` - Caching behavior
- `ImageProcessingConfig` - Image operation limits

Access via `get_config()` or create custom instances.

## API Keys
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` - For model API
- `NCBI_API_KEY` (optional) - For PubMed search
- `NCBI_EMAIL` (optional) - For PubMed API

## Testing
```bash
uv run pytest tests/                    # Run all tests
uv run pytest tests/test_tool_registry.py  # Specific test file
uv run pytest -v --tb=short             # Verbose with short traceback
```

---
*This file documents the radiant_harness package structure for Claude Code assistance.*
