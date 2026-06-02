# GAZE

[![CI](https://github.com/liamchalcroft/gaze/actions/workflows/ci.yml/badge.svg)](https://github.com/liamchalcroft/gaze/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gaze-vlm.svg)](https://pypi.org/project/gaze-vlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://liamchalcroft.github.io/gaze/)

A modular Python framework for building multi-turn agentic vision-language model (VLM) systems. Built for medical image analysis but applicable to any visual reasoning task.

## Features

- **Multi-turn agentic loop** -- JSON-structured tool-calling with configurable turn limits, schema validation, and automatic error recovery
- **25 built-in tools** (23 visual + 2 search) -- visual manipulation (zoom, crop, contrast, threshold, flip, rotate, etc.) and literature/image retrieval (PubMed, Open-i)
- **Task processors** -- abstract base class with dependency injection for prompts, schemas, and validation
- **Model adapters** -- OpenAI API (including OpenRouter), LM Studio for local models, HuggingFace Transformers
- **Verifiers integration** -- reward functions and multi-turn environments for RL training via [verifiers](https://github.com/primeintellect-ai/verifiers)

## Installation

```bash
pip install gaze-vlm
```

With extras for specific examples:

```bash
pip install gaze-vlm[nova]          # NOVA brain-MRI benchmark
pip install gaze-vlm[gemex]         # GEMeX visual grounding
pip install gaze-vlm[agentclinic]   # AgentClinic diagnostic reasoning
pip install gaze-vlm[pubmedqa]      # PubMedQA text-only QA
pip install gaze-vlm[vqa-rad]       # VQA-RAD radiology VQA
pip install gaze-vlm[medmarks]      # MedMarks-compatible NOVA environment
pip install gaze-vlm[verifiers]     # RL reward functions
```

For development:

```bash
git clone https://github.com/liamchalcroft/gaze.git
cd gaze
uv sync
```

## Quick Start

Subclass `AgenticProcessorBase` and implement four methods:

```python
import asyncio
from pathlib import Path
from gaze import AgenticProcessorBase

class MyProcessor(AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a medical imaging expert."

    def get_user_message(self, images, metadata):
        return f"Analyze this scan. History: {metadata.get('history', '')}"

    def get_response_schema(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "findings": {"type": "string"},
                        "continue": {"type": "boolean"},
                    },
                    "required": ["findings", "continue"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_response(self, response):
        return "findings" in response

async def main():
    # `async with` releases shared search/HTTP resources on exit.
    async with MyProcessor(model_name="openai/gpt-4o", use_tools=True) as processor:
        result = await processor.analyze(
            images=Path("scan.jpg"),
            metadata={"modality": "MRI", "history": "Patient presents with headache"},
        )
        print(result.final_response)

asyncio.run(main())
```

The model returns JSON each turn with `"continue": true` to keep reasoning or `"continue": false` when done.

## Architecture

```
gaze/
    base.py          AgenticProcessorBase -- subclass this
    types.py         ToolCall, ToolResult, Turn, AgenticResult (all frozen)
    config.py        Frozen dataclasses: GazeConfig, SearchConfig, etc.
    exceptions.py    GazeError hierarchy
    models/          AdapterProtocol, OpenAIAdapter, LMStudioAdapter, HuggingFaceAdapter
    tools/           Tool, ToolRegistry, 23 visual tools, 2 search tools
    retrieval/       PubMed (NCBI E-utilities), Open-i image search
    prompts/         Jinja2 templates via minijinja
    verifiers/       RL reward functions and multi-turn environments
    utils/           IoU, JSON extraction, type coercion, confidence clamping
```

## Examples

Five complete example applications are included:

| Example | Task | Dataset |
|---------|------|---------|
| [`nova/`](examples/nova/) | Brain MRI analysis (caption + diagnosis + localization) | [c-i-ber/Nova](https://huggingface.co/datasets/c-i-ber/Nova) |
| [`gemex_thinkvg/`](examples/gemex_thinkvg/) | Visual grounding with chain-of-thought | MIMIC-CXR (PhysioNet) |
| [`agentclinic_nejm/`](examples/agentclinic_nejm/) | Multi-turn diagnostic reasoning | AgentClinic NEJM |
| [`pubmedqa/`](examples/pubmedqa/) | Medical Q&A (text-only) | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| [`vqa_rad/`](examples/vqa_rad/) | Radiology VQA | [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) |

Each example includes a CLI, evaluation metrics, and run scripts for local models.

## Local Models (LM Studio)

All examples support local model inference via LM Studio:

```bash
uv run python -m examples.nova.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode single_turn \
  --max-samples 5
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Yes (for cloud models) | Model API access |
| `NCBI_API_KEY` | No | Higher PubMed rate limits |
| `NCBI_EMAIL` | No | PubMed API compliance |

## Development

```bash
uv sync                          # Install dependencies
make check                       # Quality gate: lint + format + typecheck + lockfile + tests
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run pyright src/              # Type check
uv run pytest tests/ -x          # Run tests
```

## Documentation

- [API Reference](https://liamchalcroft.github.io/gaze/)
- [Tool Reference](docs/tools.md)
- [Configuration](docs/configuration.md)
- [Verifiers Integration](docs/verifiers_integration.md)
- [MedMarks Integration](docs/MEDMARKS_INTEGRATION.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Stability & Versioning

GAZE follows [Semantic Versioning](https://semver.org). While the project is
pre-1.0, minor releases may include breaking changes to the public API; each is
recorded in the [Changelog](CHANGELOG.md). The public API is the set of names
exported from the top-level `gaze` package (`gaze.__all__`); anything
underscore-prefixed or imported from a submodule is internal and may change
without notice. From 1.0 onward, removals will ship with a deprecation warning
for at least one minor release.

## Citation

If you use GAZE in your research, please cite:

```bibtex
@inproceedings{chalcroft2026gaze,
  title={GAZE: A Modular Framework for Agentic Vision-Language Models in Medical Image Analysis},
  author={Chalcroft, Liam},
  year={2026}
}
```

## License

[MIT](LICENSE)
