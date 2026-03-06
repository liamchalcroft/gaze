# PubMedQA Example

Text-only biomedical question answering using `radiant_harness` with optional PubMed web search.

## Overview

Demonstrates agentic analysis without images. The processor uses PubMed search to gather evidence before answering yes/no/maybe questions from the PubMedQA dataset.

## Contents

- `PubmedQAProcessor` -- processor with web search support (no visual tools)
- CLI: `python -m src.cli`
- Evaluation: accuracy on yes/no/maybe classification
- Dataset loader for [PubMedQA](https://pubmedqa.github.io/)

## Run

```bash
cd examples/pubmedqa
uv run python -m src.cli --help
```

## Structure

```
pubmedqa/
├── src/
│   ├── __init__.py       # Package exports
│   ├── processor.py      # PubmedQAProcessor
│   ├── cli.py            # CLI entry point
│   ├── dataset.py        # Dataset loader
│   ├── evaluation.py     # Accuracy metrics
│   └── schemas.py        # Answer normalization
└── README.md
```
