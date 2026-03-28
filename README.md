# Radiant Harness

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular framework for building multi-turn agentic vision-language model systems. Built for medical image analysis but usable for any visual reasoning task.

## Features

- **Tool system** -- extensible registry with visual manipulation (zoom, crop, contrast, threshold, flip, rotate) and search tools (PubMed, Open-i)
- **Multi-turn agentic loop** -- full tool-calling support with configurable turn limits and JSON-structured output
- **Task processors** -- abstract base class with dependency injection for prompts, schemas, and validation
- **Model adapters** -- OpenAI API (including OpenRouter), LM Studio for local models, plus optional HuggingFace adapters
- **Verifiers integration** -- reward functions and multi-turn environments for RL training

## Installation

```bash
git clone https://github.com/liamchalcroft/nova_retrieval_vlm.git
cd nova_retrieval_vlm
uv sync
```

## Usage

Subclass `AgenticProcessorBase` and implement four methods:

```python
from pathlib import Path

from radiant_harness import AgenticProcessorBase

class MyProcessor(AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a medical imaging expert. Analyze the provided images."

    def get_user_message(self, images, metadata):
        return f"Analyze this scan. History: {metadata.get('history', '')}"

    def get_response_schema(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "findings": {"type": "string"},
                        "continue": {"type": "boolean"}
                    },
                    "required": ["findings", "continue"],
                    "additionalProperties": False,
                }
            }
        }

    def validate_response(self, response):
        return "findings" in response

processor = MyProcessor(model_name="openai/gpt-4o", use_tools=True)
result = await processor.analyze(
    images=Path("scan.jpg"),
    metadata={"modality": "MRI", "history": "Patient presents with..."}
)
```

The model returns JSON each turn with `"continue": true` to keep reasoning or `"continue": false` when done.

## LM Studio Baseline

Set `LMSTUDIO_BASE_URL` or pass `--base-url` to the example CLIs to point at your LM Studio instance (e.g. `http://localhost:1234/v1`).

```bash
uv run python -m examples.pubmedqa.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode single_turn \
  --max-samples 1

uv run python -m examples.vqa_rad.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode agentic \
  --use-tools \
  --max-samples 1

uv run python -m examples.nova.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode single_turn \
  --max-turns 1 \
  --max-samples 1
```

## Project Structure

```
src/radiant_harness/
    base.py                 # AgenticProcessorBase abstract class
    types.py                # ToolCall, ToolResult, Turn, AgenticResult
    config.py               # Configuration dataclasses
    exceptions.py           # Exception hierarchy
    _frozen.py              # deep_freeze / deep_thaw utilities
    cache.py                # TTLCache
    models/                 # AdapterProtocol, OpenAIAdapter, LMStudioAdapter, HuggingFaceAdapter
    tools/                  # Tool, ToolRegistry, visual tools, search tools
    retrieval/              # PubMed search, Open-i image search
    prompts/                # Jinja2 template loading
    verifiers/              # BaseMultiTurnEnv, reward functions, adapter
    utils/                  # IoU, JSON extraction, type coercion, confidence clamping
examples/
    nova/                   # NOVA brain-MRI benchmark (fully implemented)
    gemex_thinkvg/          # GEMeX visual grounding with RL rewards
    agentclinic_nejm/       # Multi-turn diagnostic reasoning
    pubmedqa/               # Medical Q&A (CLI + processor + evaluation)
    vqa_rad/                # Radiology VQA (CLI + processor + evaluation)
environments/
    nova_brain_mri/         # MedMarks-compatible NOVA environment
tests/
docs/
```

## Evaluation Metrics

The NOVA example computes metrics across three tasks. All scores are normalized to **0-1** unless noted:

- **Caption**: BLEU (sacrebleu, exponential smoothing), BERTScore F1 (roberta-large, baseline-rescaled), METEOR, ROUGE-L, RadGraph F1 (optional), modality keyword F1, clinical keyword F1, binary abnormality accuracy/F1
- **Diagnosis**: Top-1 and Top-5 accuracy via exact match, synonym matching, and LLM semantic matching (configurable judge model with majority vote)
- **Localization**: mAP@0.3, mAP@0.5, mAP@[50:95] (COCO-style), box-level precision/recall at IoU 0.5

**IoU thresholds**: Localization uses both 0.3 (lenient) and 0.5 (standard). The 0.3 threshold is retained for NOVA protocol compatibility -- for a small brain lesion (~10px box in a 240mm FOV scan), IoU 0.3 tolerates ~15mm spatial error, appropriate for lobe-level screening. IoU 0.5 is recommended for clinical-grade evaluation.

**Dataset splits**: NOVA downloads the full dataset via `huggingface_hub.snapshot_download("c-i-ber/Nova")`. PubMedQA uses the `pqa_labeled` config, `train` split (1,000 expert-annotated samples). VQA-RAD defaults to the `test` split (configurable via `--split`).

**Reproducibility**: All evaluations use temperature=0.0 (greedy decoding). Pass `--seed N` to fix random seeds across Python, NumPy, PyTorch, and model API calls. Summary output captures harness version, model config, and dependency versions. BERTScore is pinned to `roberta-large` with baseline rescaling; sacrebleu uses `13a` tokenizer with exponential smoothing.

## Tests

```bash
uv run pytest                                    # all tests
uv run pytest --cov=radiant_harness --cov-report=html  # with coverage
uv run pytest tests/test_tool_registry.py        # specific file
```

## Development

```bash
uv sync                          # install all dependencies (dev included by default)
uv run ruff check .              # lint
uv run ruff format --check .     # format check
uv run pyright src/              # type check
make check                       # all of the above + lock check + tests
```

## Documentation

- [Verifiers Integration](docs/verifiers_integration.md)
- [MedMarks Integration](docs/MEDMARKS_INTEGRATION.md)
- [NOVA Example](examples/nova/README.md)
- [Contributing](CONTRIBUTING.md)

## API Keys

- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` -- for model API access
- `NCBI_API_KEY` (optional) -- for PubMed search
- `NCBI_EMAIL` (optional) -- for PubMed API compliance

## License

MIT License.
