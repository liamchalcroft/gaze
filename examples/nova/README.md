# NOVA VLM Example

Benchmarks vision-language models on the NOVA brain-MRI dataset using `gaze`. Supports localization, captioning, and diagnosis tasks with optional agentic tools and PubMed search.

## Contents

- `NOVAAgenticProcessor` -- processor with visual tools and optional PubMed search
- CLI: `python -m examples.nova.src.cli` (run from the repository root)
- Task-specific evaluation (caption F1, diagnosis accuracy, detection mAP)

## Setup

Run all commands from the repository root. Install the NOVA extra (it pulls in
torch, torchmetrics, datasets, and the metric libraries):

```bash
uv sync --extra nova
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
uv run --extra nova python -m examples.nova.src.cli \
  --task localization \
  --model openai/gpt-4o \
  --data-dir ./data/nova \
  --output-dir ./runs
```

For local inference against an LM Studio (or other OpenAI-compatible) server,
see `run_local.sh`, which sweeps single-turn and agentic modes and passes
`--base-url`.

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
