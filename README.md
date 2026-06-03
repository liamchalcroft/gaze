# GAZE

[![CI](https://github.com/liamchalcroft/gaze/actions/workflows/ci.yml/badge.svg)](https://github.com/liamchalcroft/gaze/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gaze-vlm.svg)](https://pypi.org/project/gaze-vlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/liamchalcroft/gaze/branch/main/graph/badge.svg)](https://codecov.io/gh/liamchalcroft/gaze)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/liamchalcroft/gaze/badge)](https://securityscorecards.dev/viewer/?uri=github.com/liamchalcroft/gaze)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://liamchalcroft.github.io/gaze/)

**GAZE** (Grounded Agentic Zero-shot Evaluation) is a modular Python framework for multi-turn agentic vision-language model (VLM) systems, built for medical image analysis.

A radiologist rarely reads a scan in a single glance: they zoom, adjust the window, compare regions, and consult the literature before writing a report. A vision-language model, by contrast, reads an image once and produces text in a single forward pass. GAZE closes that gap by giving a VLM viewer-level tools (zoom, windowing, contrast, edge detection) and literature retrieval (PubMed, Open-i), then running it as a multi-turn loop with schema-validated outputs and full tool-call traces for auditability. It applies to any visual reasoning task, not only medical imaging.

## Features

- **Multi-turn agentic loop** -- JSON-structured tool-calling with configurable turn limits, schema validation, and automatic error recovery
- **25 built-in tools** (23 visual + 2 search) -- visual manipulation (zoom, crop, contrast, threshold, flip, rotate, etc.) and literature/image retrieval (PubMed, Open-i)
- **Task processors** -- abstract base class with dependency injection for prompts, schemas, and validation
- **Model adapters** -- OpenAI API (including OpenRouter), LM Studio for local models, HuggingFace Transformers
- **Verifiers integration** -- reward functions and multi-turn environments for RL training via [verifiers](https://github.com/primeintellect-ai/verifiers)

## Tools at a glance

The model can call these during reasoning (multi-turn mode); the full set of 25 is in the [tool reference](https://liamchalcroft.github.io/gaze/tools/).

| Category | Representative tools |
|----------|----------------------|
| Inspect | `zoom`, `crop`, `rotate`, `flip_horizontal` |
| Enhance | `adjust_contrast`, `adjust_brightness`, `window_level`, `equalize_histogram` |
| Analyze | `threshold`, `detect_edges`, `morphological`, `symmetry_diff` |
| Retrieve | `search_web` (PubMed), `search_images` (Open-i) |

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

## Quick start

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

The model returns JSON each turn with `"continue": true` to keep reasoning or `"continue": false` when done. `result.final_response` is the validated JSON from the last turn, for example:

```json
{"findings": "No acute intracranial abnormality.", "continue": false}
```

## Architecture

```
src/gaze/
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

The import path is `gaze` (the package lives under `src/gaze/`).

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

## Local models (LM Studio)

All examples support local model inference via LM Studio:

```bash
uv run python -m examples.nova.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode single_turn \
  --max-samples 5
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Yes (for cloud models) | Model API access |
| `NCBI_API_KEY` | No | Higher PubMed rate limits |
| `NCBI_EMAIL` | No | PubMed API compliance |
| `GAZE_ALLOW_CUSTOM_BASE_URL` | No | Set to `1` to send API keys to a non-allowlisted model host |

## Development

```bash
uv sync                          # Install dependencies
make check                       # Quality gate: lint + format + typecheck + lockfile + tests
make check-nova                  # Torch-gated + example tests (installs the nova extra)
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run pyright src/              # Type check
uv run pytest tests/ -x          # Run tests
```

## Stability and versioning

GAZE follows [Semantic Versioning](https://semver.org). While the project is pre-1.0, minor releases may include breaking changes to the public API; each is recorded in the [Changelog](CHANGELOG.md). The public API is the set of names exported from the top-level `gaze` package (`gaze.__all__`); anything underscore-prefixed or imported from a submodule is internal and may change without notice. From 1.0 onward, removals will ship with a deprecation warning for at least one minor release.

## Documentation

- [Documentation site](https://liamchalcroft.github.io/gaze/)
- [Getting started](https://liamchalcroft.github.io/gaze/getting-started/)
- [Tool reference](https://liamchalcroft.github.io/gaze/tools/)
- [Configuration](https://liamchalcroft.github.io/gaze/configuration/)
- [Verifiers integration](https://liamchalcroft.github.io/gaze/verifiers_integration/)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Citation

If you use GAZE in your research, please cite:

```bibtex
@article{alim2026gaze,
  title   = {{GAZE}: Grounded Agentic Zero-shot Evaluation with Viewer-Level Tools and Literature Retrieval on Rare Brain {MRI}},
  author  = {Alim, Duaa and Alim, Mogtaba and Chalcroft, Liam},
  journal = {arXiv preprint arXiv:2605.00876},
  year    = {2026},
  note    = {Accepted at AIiH 2026},
}
```

The preprint is available at [arXiv:2605.00876](https://arxiv.org/abs/2605.00876).

## License

[MIT](LICENSE)
