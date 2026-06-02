# PubMedQA example

Text-only biomedical question answering using GAZE with optional PubMed web search.

## Overview

Demonstrates agentic analysis without images. The `PubmedQAProcessor` answers yes/no/maybe questions from the PubMedQA dataset, in single-turn mode (answer directly from the provided context) or agentic mode (issue PubMed searches to gather evidence before answering). Evaluation reports accuracy, per-class precision/recall/F1, and macro-F1.

## Dataset

The dataset downloads automatically from the HuggingFace hub ([`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)) on first run; no manual download or local files are needed. The `--config` flag selects the subset (`pqa_labeled`, `pqa_artificial`, or `pqa_unlabeled`); the default is `pqa_labeled`.

## Install

Run from the repository root. The example only needs the lightweight `pubmedqa` extra (HuggingFace `datasets`):

```bash
uv sync --extra pubmedqa
# or
pip install gaze-vlm[pubmedqa]
```

## Run

```bash
uv run python -m examples.pubmedqa.src.cli \
  --model openai/gpt-4o \
  --mode agentic \
  --use-search \
  --max-samples 50 \
  --output-dir ./runs/pubmedqa
```

## Run locally (LM Studio)

`run_local.sh` sweeps single-turn then agentic against a local OpenAI-compatible server. Pass `--base-url` (default `http://localhost:1234/v1`) to point at LM Studio or any compatible endpoint:

```bash
./examples/pubmedqa/run_local.sh qwen3.5-35b-a3b http://localhost:1234/v1 50
```

PubMedQA is text-only, so it works with `n_ctx >= 4096`; thinking models benefit from 8192. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs).

## Flags

- `--mode {single_turn,agentic}`: single-turn answers from context; agentic can search and iterate
- `--use-search`: enable PubMed retrieval (agentic mode)
- `--max-turns N`: agentic turn limit (single-turn forces 1; agentic defaults to 5)
- `--max-tokens N`: completion tokens per turn (thinking models need >= 4096)
- `--config {pqa_labeled,pqa_artificial,pqa_unlabeled}`: dataset subset
- `-v`: verbose logging

## Output

- `summary.json` with aggregate metrics and the run configuration
- per-sample records under the chosen `--output-dir`

## Structure

```
pubmedqa/
    src/
        __init__.py       # Package exports
        processor.py      # PubmedQAProcessor
        cli.py            # CLI entry point
        dataset.py        # HuggingFace dataset loader
        evaluation.py     # Accuracy / F1 metrics
        schemas.py        # Response schema + answer normalization
    tests/                # Hermetic smoke tests
    run_local.sh          # Local (LM Studio) evaluation sweep
    README.md
```

## References

- [PubMedQA dataset](https://huggingface.co/datasets/qiaojin/PubMedQA)
- [PubMedQA project](https://pubmedqa.github.io/)
- [GAZE](https://github.com/liamchalcroft/gaze)

## License

MIT, following the GAZE framework.
