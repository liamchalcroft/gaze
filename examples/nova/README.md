# NOVA example

Benchmarks vision-language models on the NOVA brain-MRI dataset using GAZE, across localization, captioning, and diagnosis.

## Overview

The `NOVAAgenticProcessor` runs three brain-MRI sub-tasks (caption generation, diagnosis prediction, lesion localization) in single-turn or agentic mode. Agentic mode adds visual tools (zoom, crop, contrast, windowing, and more) and optional PubMed search. Task-specific evaluation reports caption token-F1, diagnosis accuracy, and detection F1.

## Dataset

Both images and ground-truth annotations load from the HuggingFace dataset [`c-i-ber/Nova`](https://huggingface.co/datasets/c-i-ber/Nova) (parquet) at runtime; it auto-downloads on first run. No local CSVs are required by default. If `--data-dir` points to a directory containing `captions.csv`, `case_metadata.csv`, or `bboxes_gold.csv`, those override the HuggingFace data.

## Install

Run all commands from the repository root. Install the NOVA extra (it pulls in torch, torchvision, torchmetrics, datasets, and the metric libraries):

```bash
uv sync --extra nova
# or
pip install gaze-vlm[nova]
```

Create `.env`:

```dotenv
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Run

```bash
uv run --extra nova python -m examples.nova.src.cli \
  --task localization \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --max-turns 5 \
  --max-samples 10 \
  --output-dir ./runs
```

## Run locally (LM Studio)

`run_local.sh` sweeps single-turn then agentic against a local OpenAI-compatible server, passing `--base-url`:

```bash
./examples/nova/run_local.sh qwen3.5-35b-a3b http://localhost:1234/v1 50
```

NOVA needs `n_ctx >= 8192` (16384 recommended); the agentic schema is large and thinking models need extra headroom. The script sets `--max-image-dim 256` to keep encoded images within budget. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs). A MedGemma variant lives in `run_medgemma.sh`.

## Flags

- `--task {all,caption,diagnosis,localization}`: task to evaluate (default `all`)
- `--mode {agentic,single_turn}`: multi-turn agentic or single-turn (default `agentic`)
- `--use-tools`: enable visual tools (zoom, crop, contrast, windowing, etc.)
- `--use-web-search`: enable PubMed search
- `--max-turns N`: agentic turn limit (default 10)
- `--max-tokens N`: completion tokens per turn (default: 16384)
- `--max-image-dim N`: downscale images so neither side exceeds N pixels before encoding
- `--judge-model NAME`: model for diagnosis semantic matching (default: `NOVA_SEMANTIC_MATCH_MODEL` env or `openai/gpt-5-nano`)
- `--reasoning`: enable model reasoning mode
- `--batch-size N`: concurrent processing (default 4)
- `--max-samples N`: cap samples processed (0 = all)
- `--eval-tasks ...`: metrics to compute (default: all matching `--task`)
- `--seed N`: random seed for reproducibility
- `--no-skip-existing`: reprocess existing outputs
- `--dry-run`: print resolved configuration and exit
- `-v`: verbose logging

## Output

- `sample_<index>.json`: per-sample result, turns, and tool-call log
- `summary.json`: aggregate metrics, run configuration, and token summary
- `diagnosis_judgment_log.json`: per-sample diagnosis judging trace (when diagnosis is evaluated)

## Structure

```
nova/
    src/
        __init__.py
        cli.py              # CLI entry point (argparse + run_evaluation)
        config.py           # Frozen NOVAConfig dataclass
        processor.py        # NOVAAgenticProcessor
        schemas.py          # Response schemas + validate_response()
        rewards.py          # RL reward functions
        data/               # NOVA dataset + ground-truth loading
        evaluation/         # Metrics package, one module per sub-task
            caption.py
            detection.py
            diagnosis.py
        prompts/            # Jinja2 templates (single_turn, agentic)
    docs/                   # Usage and agentic-workflow guides
    tests/                  # NOVA-specific tests
    run_local.sh            # Local (LM Studio) evaluation sweep
    run_medgemma.sh         # MedGemma local sweep
    README.md
```

## References

- [NOVA dataset](https://huggingface.co/datasets/c-i-ber/Nova)
- [Usage guide](docs/usage.md)
- [Agentic workflow](docs/agentic_workflow.md)
- [Tool reference](../../docs/tools.md)
- [GAZE](https://github.com/liamchalcroft/gaze)

## License

MIT, following the GAZE framework.
