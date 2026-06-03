# Usage guide

## Prerequisites

- Python 3.10+
- uv package manager
- OpenRouter and/or OpenAI API keys

## Install

Run all commands from the repository root:

```bash
uv sync --extra nova
```

## Environment

Create `.env` at the repository root (the CLI reads API keys from the
environment; data and output directories are passed with `--data-dir` /
`--output-dir`, not env vars):

```dotenv
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
# Optional: override the diagnosis semantic-match judge
# NOVA_SEMANTIC_MATCH_MODEL=openai/gpt-5-nano
# NOVA_SEMANTIC_MATCH_BASE_URL=http://localhost:1234/v1
```

## Data

- Both images and ground-truth annotations load from the HuggingFace dataset `c-i-ber/Nova` (parquet format) at runtime.
- No local CSV files are required by default.
- If `--data-dir` is provided, the CLI checks for local CSVs (`captions.csv`, `case_metadata.csv`, `bboxes_gold.csv`) and uses them as an override; otherwise it falls back to the HuggingFace dataset.

## CLI

### Basic run

```bash
uv run --extra nova python -m examples.nova.src.cli \
  --task localization \
  --model openai/gpt-4o \
  --output-dir ./runs
```

### With tools and search

```bash
uv run --extra nova python -m examples.nova.src.cli \
  --task diagnosis \
  --model openai/gpt-4o \
  --use-tools \
  --use-web-search \
  --max-turns 5
```

### Re-run existing samples

```bash
uv run --extra nova python -m examples.nova.src.cli \
  --task caption \
  --model openai/gpt-4o-mini \
  --no-skip-existing
```

### Verbose

```bash
uv run --extra nova python -m examples.nova.src.cli --task all --model openai/gpt-4o -v
```

## Output

- `sample_<index>.json` -- per-sample result
- `summary.json` -- aggregate metrics and configuration

See [Agentic workflow](./agentic_workflow.md) for the tool loop and prompt flow.
