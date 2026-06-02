# Usage guide

## Prerequisites

- Python 3.10+
- uv package manager
- OpenRouter and/or OpenAI API keys

## Install

```bash
cd examples/nova
uv sync
```

## Environment

Create `.env` in `examples/nova/`:

```dotenv
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Data

- Both images and ground-truth annotations load from the HuggingFace dataset `c-i-ber/Nova` (parquet format) at runtime.
- No local CSV files are required by default.
- If `--data-dir` is provided, the CLI checks for local CSVs (`captions.csv`, `case_metadata.csv`, `bboxes_gold.csv`) and uses them as an override; otherwise it falls back to the HuggingFace dataset.

## CLI

### Basic run

```bash
uv run python -m src.cli \
  --task localization \
  --model openai/gpt-4o \
  --data-dir ./data/nova \
  --output-dir ./runs
```

### With tools and search

```bash
uv run python -m src.cli \
  --task diagnosis \
  --model openai/gpt-4o \
  --use-tools \
  --use-web-search \
  --max-turns 5
```

### Re-run existing samples

```bash
uv run python -m src.cli \
  --task caption \
  --model openai/gpt-4o-mini \
  --no-skip-existing
```

### Verbose

```bash
uv run python -m src.cli --task all --model openai/gpt-4o -v
```

## Output

- `sample_<index>.json` -- per-sample result
- `summary.json` -- aggregate metrics and configuration

See [Agentic workflow](./agentic_workflow.md) for the tool loop and prompt flow.
