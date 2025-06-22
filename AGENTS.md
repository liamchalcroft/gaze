# Agent Guidelines for NOVA Retrieval VLM

This file summarises key points about the repository to help when completing tasks.

## Project Overview
- **Purpose**: Retrieval-augmented vision-language modelling for medical images, particularly the NOVA brain MRI benchmark.
- **Main package**: `src/nova_retrieval_vlm/` contains CLI, evaluation utilities, prompt templates, guideline indexing and advanced visual reasoning modules.
- **Scripts**: under `scripts/` for dataset download, index building and experiment orchestration.
- **Tests**: minimal pytest suite in `tests/` with fixtures in `conftest.py`.

## Development Workflow
- Python >=3.9.
- Dependencies managed via **uv** or pip (`uv pip install -e .`).
- For a quick environment check run `make check`.
- Run the test suite with `pytest` (or `make test`).
- Lint/format using `black`, `isort` and `ruff` (`make quality`).
- Use type hints and write docstrings for public functions/classes.

## Coding Style
- Follow PEP 8 conventions.
- The project uses **Black** with a line length of **100** (see `pyproject.toml`).
- Import sorting handled by **isort**; linting by **ruff**.
- Commit messages should follow the conventional format (e.g. `feat:`, `fix:`, `docs:`).

## Testing
- Always run `pytest` after modifying code. Ensure tests pass before committing.
- If adding new features, include corresponding tests under `tests/`.

## Useful Make Targets
- `make setup` – install dependencies and create `.env` from template.
- `make data` – download the NOVA dataset and build retrieval indexes.
- `make exp` – run the full experiment suite.
- `make clean` – remove caches and temporary files.

## Additional Notes
- Prompt templates live in `src/nova_retrieval_vlm/prompts/`.
- Retrieval indexes are built from guideline documents defined in `docs/guidelines.yaml`.
- Visual reasoning utilities are in `src/nova_retrieval_vlm/visual_reasoning/`.

Keep this reference handy when working on future tasks.
