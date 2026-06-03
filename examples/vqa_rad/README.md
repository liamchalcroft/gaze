# VQA-RAD example

Radiology visual question answering using GAZE with visual tools and optional web search.

## Overview

Demonstrates medical VQA with tool-augmented reasoning. The `VQARadProcessor` analyzes radiology images and answers clinical questions from the VQA-RAD dataset, in single-turn mode (answer directly from the image) or agentic mode (zoom, crop, adjust contrast, and search before answering). Evaluation reports exact match and token-F1, split across open-ended and closed-ended (yes/no) questions.

## Dataset

The dataset downloads automatically from the HuggingFace hub ([`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)) on first run; no manual download or local files are needed. Use `--split {train,test}` to choose the split (default `test`).

## Install

Run from the repository root. The example needs the `vqa-rad` extra (HuggingFace `datasets`). The model must accept image input:

```bash
uv sync --extra vqa-rad
# or
pip install gaze-vlm[vqa-rad]
```

## Run

```bash
uv run python -m examples.vqa_rad.src.cli \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --max-samples 50 \
  --max-image-dim 256 \
  --output-dir ./runs/vqa_rad
```

`--model openai/...` runs go through OpenRouter/OpenAI: set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) first, or the run fails with "No API key found". The local LM Studio path via `--base-url` (below) needs no key.

## Run locally (LM Studio)

`run_local.sh` sweeps single-turn then agentic against a local OpenAI-compatible server. Pass `--base-url` (default `http://localhost:1234/v1`) to point at LM Studio or any compatible vision-capable endpoint:

```bash
./examples/vqa_rad/run_local.sh glm-4.6v-flash http://localhost:1234/v1 50
```

Use `n_ctx >= 4096` (8192 recommended). The script sets `--max-image-dim 256` to keep the encoded image within a small context budget. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs).

## Flags

- `--model NAME`: model name (must accept image input; OpenRouter format, or local ID for `--base-url`; default `openai/gpt-4o`)
- `--base-url URL`: OpenAI-compatible server, e.g. `http://localhost:1234/v1`
- `--mode {single_turn,agentic}`: single-turn answers directly; agentic can use tools and iterate (default `agentic`)
- `--split {train,test}`: dataset split (default `test`)
- `--use-tools`: enable visual manipulation tools (agentic mode)
- `--use-web-search`: enable medical literature/image search (agentic mode)
- `--max-samples N`: maximum samples to evaluate (default: all)
- `--max-turns N`: agentic turn limit (single-turn forces 1; agentic defaults to 5)
- `--max-tokens N`: completion tokens per turn (thinking models need >= 4096)
- `--max-image-dim N`: downscale images so neither side exceeds N pixels before encoding
- `--batch-size N`: samples processed concurrently (default 1)
- `--reasoning`: enable model reasoning mode
- `--seed N`: random seed for reproducibility
- `--output-dir PATH`: output directory (default `./runs/vqa_rad`)
- `-v`, `--verbose`: verbose logging

## Output

- `summary.json` with aggregate metrics and the run configuration
- per-sample records under the chosen `--output-dir`

## Structure

```
vqa_rad/
    src/
        __init__.py       # Package exports
        processor.py      # VQARadProcessor
        cli.py            # CLI entry point
        dataset.py        # HuggingFace dataset loader
        evaluation.py     # Exact-match / token-F1 metrics
        schemas.py        # Response schemas + validation
    tests/                # Hermetic smoke tests
    run_local.sh          # Local (LM Studio) evaluation sweep
    README.md
```

## References

- [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [VQA-RAD on OSF](https://osf.io/89kps/)
- [GAZE](https://github.com/liamchalcroft/gaze)

## License

MIT, following the GAZE framework.
