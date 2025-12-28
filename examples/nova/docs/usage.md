# Usage Guide

## Setup

### Prerequisites

- Python 3.10+
- uv package manager
- OpenRouter and/or OpenAI API keys (required for hosted models)

### Install Dependencies

```bash
# From repo root
cd examples/nova
uv sync
```

### Environment

Create a `.env` file in `examples/nova`:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional defaults
DATA_DIR=./data/nova
OUTPUT_DIR=./runs
```

## Data Requirements

- Images are loaded from the HuggingFace dataset `c-i-ber/Nova` at runtime.
- Ground-truth CSVs must exist in `DATA_DIR`:
  - `captions.csv`
  - `case_metadata.csv`
  - `bboxes_gold.csv`

If these files are missing, the CLI will fail fast when loading ground truth.

## CLI Usage

### Basic Run

```bash
uv run python -m src.cli \
  --task localization \
  --model openai/gpt-4o \
  --data-dir ./data/nova \
  --output-dir ./runs
```

### Enable Tools + Web Search

```bash
uv run python -m src.cli \
  --task diagnosis \
  --model openai/gpt-4o \
  --use-tools \
  --use-web-search \
  --max-turns 5
```

### Re-run Existing Samples

```bash
uv run python -m src.cli \
  --task caption \
  --model openai/gpt-4o-mini \
  --no-skip-existing
```

### Verbose Logs

```bash
uv run python -m src.cli --task all --model openai/gpt-4o -v
```

## Outputs

The CLI writes per-sample results and a summary:

- `sample_<index>.json` for each processed sample
- `summary.json` with aggregate metrics and configuration

## Visualization (Optional)

```bash
uv run streamlit run src/visualization/gui.py
```

See [Agentic Workflow](./agentic_workflow.md) for the tool loop and prompt flow.
