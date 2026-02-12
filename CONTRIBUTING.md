# Contributing to Radiant Harness

## Setup

1. Fork and clone:
   ```bash
   git clone https://github.com/your-username/radiant_harness.git
   cd radiant_harness
   ```

2. Install:
   ```bash
   uv sync
   pre-commit install
   ```

3. Configure API keys in `.env` (see README.md).

## Workflow

1. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes, then check:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format .
   uv run pyright src/
   ```

3. Commit (conventional commits):
   ```bash
   git commit -m "feat: description of changes"
   ```

## What to Contribute

### Bug Reports
- Steps to reproduce, system info, logs/error messages.

### New Model Adapters
Implement `AdapterProtocol`:
```python
# src/radiant_harness/models/your_adapter.py
from radiant_harness.models import AdapterProtocol, GenerationLog

class YourAdapter:
    async def generate_chat(self, messages, max_tokens, temperature, tools, response_format, stream):
        # ...
        return content, tool_calls, GenerationLog(prompt_tokens, completion_tokens, finish_reason)
```

### New Tools
Create an async execute function and a `Tool` instance:
```python
from radiant_harness.tools import Tool, ToolRegistry
from radiant_harness.types import ToolResult

async def _execute_my_tool(registry: ToolRegistry, **kwargs) -> ToolResult:
    # ...
    return ToolResult(tool_name="my_tool", description="Did something")

my_tool = Tool(
    name="my_tool",
    description="Does something useful",
    parameters={"type": "object", "properties": {...}},
    execute=_execute_my_tool,
)
```

### Evaluation Metrics
Add to the relevant example's `evaluation/` directory. Follow the existing pattern in `examples/nova/src/evaluation/`.

## Code Standards

- **Style**: ruff handles formatting (line length 100). Use type hints.
- **Runtime validation**: `@beartype` decorator on public functions.
- **Exceptions**: use specific types from `radiant_harness.exceptions`. Never bare `except:`.
- **Commits**: conventional format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`).
- **Tests**: pytest, >60% coverage target, mock external API calls.

## Pull Requests

- Clear title and description
- All CI checks pass
- New tests for new functionality
- Squash and merge for feature branches

## Research Contributions

When contributing experimental results, include:
- Dataset splits, model configurations, evaluation metrics
- Configuration files, random seeds, dependency versions
- Multiple runs with confidence intervals where possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
