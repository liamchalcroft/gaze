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

- Images are fetched from the HuggingFace dataset `c-i-ber/Nova` at runtime.
- Ground-truth CSVs must exist in `DATA_DIR`: `captions.csv`, `case_metadata.csv`, `bboxes_gold.csv`

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
