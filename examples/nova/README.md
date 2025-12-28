# NOVA VLM (Example)

This example shows how to run the NOVA brain-MRI benchmark with `radiant_harness`.
It provides a minimal CLI, agentic processor, and evaluation helpers for
localization, captioning, and diagnosis.

## What’s Included

- `NOVAAgenticProcessor` with visual tools and optional PubMed search
- CLI entry point: `python -m src.cli`
- Task-specific evaluation utilities
- Optional Streamlit visualization

## Setup

```bash
cd examples/nova
uv sync
```

Set API keys in `.env` (OpenRouter/OpenAI as needed):

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Data Requirements

- Images are fetched from the HuggingFace dataset `c-i-ber/Nova` at runtime.
- Ground-truth CSVs must exist in `DATA_DIR`:
  - `captions.csv`
  - `case_metadata.csv`
  - `bboxes_gold.csv`

## Run

```bash
uv run python -m src.cli \
  --task localization \
  --model openai/gpt-4o \
  --data-dir ./data/nova \
  --output-dir ./runs
```

Optional flags:

- `--use-tools` enable visual tools
- `--use-web-search` enable PubMed search
- `--max-turns` set agentic turn limit
- `--reasoning` enable model reasoning mode
- `--batch-size` set batch size
- `--no-skip-existing` reprocess existing outputs
- `-v` enable verbose logging

## Outputs

Results are saved to the output directory:

- `sample_<index>.json` per sample
- `summary.json` with metrics and configuration

## Docs

- `examples/nova/docs/usage.md`
- `examples/nova/docs/agentic_workflow.md`
