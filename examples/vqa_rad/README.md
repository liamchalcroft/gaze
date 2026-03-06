# VQA-RAD Example

Radiology visual question answering using `radiant_harness` with visual tools and optional web search.

## Overview

Demonstrates medical VQA with tool-augmented reasoning. The processor analyzes radiology images and answers clinical questions from the VQA-RAD dataset.

## Contents

- `VQARadProcessor` -- processor with visual tools and optional search
- CLI: `python -m src.cli`
- Evaluation: accuracy on open-ended and closed-ended questions
- Dataset loader for [VQA-RAD](https://osf.io/89kps/)

## Run

```bash
cd examples/vqa_rad
uv run python -m src.cli --help
```

## Structure

```
vqa_rad/
├── src/
│   ├── __init__.py       # Package exports
│   ├── processor.py      # VQARadProcessor
│   ├── cli.py            # CLI entry point
│   ├── dataset.py        # Dataset loader
│   ├── evaluation.py     # Accuracy metrics
│   └── schemas.py        # Response schemas
└── README.md
```
