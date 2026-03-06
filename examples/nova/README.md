# NOVA VLM Example

Benchmarks vision-language models on the NOVA brain-MRI dataset using `radiant_harness`. Supports localization, captioning, and diagnosis tasks with optional agentic tools and PubMed search.

## Contents

- `NOVAAgenticProcessor` -- processor with visual tools and optional PubMed search
- CLI: `python -m src.cli`
- Task-specific evaluation (caption F1, diagnosis accuracy, detection mAP)

## Setup

```bash
cd examples/nova
uv sync
```

Create `.env`:
```dotenv
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Data

- Both images and ground-truth annotations load from the HuggingFace dataset `c-i-ber/Nova` (parquet) at runtime.
- No local CSVs are required by default. If `--data-dir` points to a directory with `captions.csv`, `case_metadata.csv`, or `bboxes_gold.csv`, those are used as overrides.

## Run

```bash
uv run python -m src.cli \
  --task localization \
  --model openai/gpt-4o \
  --data-dir ./data/nova \
  --output-dir ./runs
```

Flags:
- `--use-tools` -- enable visual tools (zoom, crop, contrast, etc.)
- `--use-web-search` -- enable PubMed search
- `--max-turns N` -- set agentic turn limit
- `--reasoning` -- enable model reasoning mode
- `--batch-size N` -- concurrent processing
- `--no-skip-existing` -- reprocess existing outputs
- `-v` -- verbose logging

## Output

- `sample_<index>.json` per sample
- `summary.json` with metrics and configuration

## Docs

- [Usage Guide](docs/usage.md)
- [Agentic Workflow](docs/agentic_workflow.md)
