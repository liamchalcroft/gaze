# VQA-RAD Example

Radiology visual question answering using `gaze` with visual tools and optional web search.

## Overview

Demonstrates medical VQA with tool-augmented reasoning. The processor analyzes radiology images and answers clinical questions from the VQA-RAD dataset, in either single-turn mode (answer directly from the image) or agentic mode (zoom, crop, adjust contrast, and search before answering).

## Contents

- `VQARadProcessor` -- processor with visual tools and optional search
- CLI: `python -m examples.vqa_rad.src.cli` (run from the repository root)
- Evaluation: exact match and token-F1, split across open-ended and closed-ended (yes/no) questions
- Dataset loader for [VQA-RAD](https://osf.io/89kps/)

## Data

The dataset downloads automatically from the HuggingFace hub (`flaviagiammarino/vqa-rad`) on first run; no manual download or local files are needed. Use `--split {train,test}` to choose the split (default `test`).

## Run

Run from the repository root. The example needs the `vqa-rad` extra (HuggingFace `datasets`). The model must accept image input:

```bash
uv sync --extra vqa-rad

uv run python -m examples.vqa_rad.src.cli \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --max-samples 50 \
  --max-image-dim 256 \
  --output-dir ./runs/vqa_rad
```

Useful flags:

- `--mode {single_turn,agentic}` -- single-turn answers directly; agentic can use tools and iterate
- `--use-tools` -- enable visual manipulation tools (agentic mode)
- `--use-search` -- enable medical literature/image search (agentic mode)
- `--max-turns N` -- agentic turn limit (single-turn forces 1; agentic defaults to 5)
- `--max-tokens N` -- completion tokens per turn (thinking models need >= 4096)
- `--max-image-dim N` -- downscale images so neither side exceeds N pixels before encoding
- `-v` -- verbose logging

## Run locally (LM Studio)

`run_local.sh` sweeps single-turn then agentic against a local OpenAI-compatible server. Pass `--base-url` (default `http://localhost:1234/v1`) to point at LM Studio or any compatible vision-capable endpoint:

```bash
./examples/vqa_rad/run_local.sh glm-4.6v-flash http://localhost:1234/v1 50
```

Use `n_ctx >= 4096` (8192 recommended). The script sets `--max-image-dim 256` to keep the encoded image within a small context budget. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs).

## Output

- `summary.json` with aggregate metrics and the run configuration
- per-sample records under the chosen `--output-dir`

## Structure

```
vqa_rad/
├── src/
│   ├── __init__.py       # Package exports
│   ├── processor.py      # VQARadProcessor
│   ├── cli.py            # CLI entry point
│   ├── dataset.py        # HuggingFace dataset loader
│   ├── evaluation.py     # Exact-match / token-F1 metrics
│   └── schemas.py        # Response schemas + validation
├── tests/                # Hermetic smoke tests
├── run_local.sh          # Local (LM Studio) evaluation sweep
└── README.md
```
