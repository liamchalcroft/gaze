# Radiant Harness

A lightweight, standalone tool and prompt harness for multi-turn VLM/LLM workflows, built for medical imaging but usable anywhere you need agentic tool-calling with images.

## What’s here
- `src/radiant_harness/`: core library (tool registry, visual/search tools, adapter protocol, prompts, agentic loop).
- `examples/nova/`: full NOVA benchmark example, kept as an optional add-on.

## Quick start
```bash
uv sync
uv run python - <<'PY'
from pathlib import Path
from radiant_harness import ToolRegistry, create_visual_tools

img = Path("tests/data/test_image.png")  # provide your own image path
registry = ToolRegistry(image_path=img, tools=create_visual_tools())
print([t["function"]["name"] for t in registry.get_tool_schemas()])
PY
```

## Running tests
- Core harness tests (fast): `uv run pytest`
- NOVA example tests (optional): `uv run pytest examples/nova/tests`

## Repo layout
- Core: `src/radiant_harness`, `tests/`, `pyproject.toml`
- Example: `examples/nova/` (NOVA-specific code, configs, docs, data caches, and tests)

## Naming
Name: **Radiant Harness**. Package: `radiant-harness`. Import: `radiant_harness`.

