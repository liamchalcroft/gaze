# NOVA Retrieval VLM Project Guide for Claude

## Project Overview
This is a research framework for comparing vision-language models on the NOVA brain-MRI benchmark dataset. It evaluates baseline and retrieval-augmented models on medical imaging analysis tasks including localization, captioning, and diagnosis.

## Quick Reference

### Project Structure
```
nova_retrieval_vlm/
├── src/nova_retrieval_vlm/     # Main Python package
│   ├── cli.py                  # Main CLI interface
│   ├── config.py               # Hydra configuration classes
│   ├── agentic/                # Agentic processing (multi-turn, visual tools)
│   ├── data/                   # Dataset handling
│   ├── evaluation/             # Task-specific evaluation metrics
│   ├── models/                 # Model adapters (OpenAI, OpenRouter)
│   ├── processors/             # Task processors (localization, diagnosis, caption)
│   ├── prompts/                # Jinja2 prompt templates
│   ├── retrieval/              # Retrieval system (BM25, FAISS, Hybrid)
│   └── visualization/          # Streamlit GUI and plotting
├── scripts/                    # Utility scripts for benchmarking
├── tests/                      # Test suite (150+ tests)
├── paper/                      # LaTeX paper files
└── docs/                       # Documentation
```

## Key Commands

### Development Commands
```bash
# Environment management (REQUIRED - use uv exclusively)
uv sync
uv run python -m nova_retrieval_vlm.cli_new task=localization

# Run tests with proper coverage
uv run pytest
uv run pytest --cov=nova_retrieval_vlm --cov-report=html

# Modern code quality checks (REQUIRED tools)
uv run ruff check .          # Linting with modern rules
uv run ruff format .         # Formatting (replaces black/isort)
uv run pyright              # Type checking (replaces mypy)

# Quality assurance script
bash scripts/check_quality.sh  # Comprehensive quality checks

# Pre-commit hooks (configured for modern tools)
pre-commit run --all-files
```

### Running Experiments
```bash
# Basic localization task
python -m nova_retrieval_vlm.cli task=localization model.name=openai/gpt-4o

# With retrieval augmentation
python -m nova_retrieval_vlm.cli task=localization use_retrieval=true retrieval.type=bm25

# Multi-turn analysis
python -m nova_retrieval_vlm.cli task=diagnosis approach=multiturn
```

### Benchmark Scripts
```bash
# Run complete benchmark suite
bash scripts/run_full_benchmarks.sh

# Individual benchmarks
bash scripts/run_baseline_benchmark.sh
bash scripts/run_retrieval_benchmark.sh
bash scripts/run_multiturn_benchmark.sh
```

## Technology Stack
- **Core**: Python 3.10+, PyTorch, Hydra configuration
- **Models**: OpenRouter API (100+ models), OpenAI API
- **Retrieval**: Haystack (BM25, FAISS), Sentence Transformers
- **Evaluation**: TorchMetrics, BERTScore, RadGraph
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Type Safety**: jaxtyping (tensor shapes), beartype (runtime validation)
- **Development**: uv (env management), ruff (linting/format), pyright (type check)
- **Modern Tools**: fd (file search), ast-grep (code analysis)

## Important Files
- `src/nova_retrieval_vlm/cli.py` - Main CLI interface
- `src/nova_retrieval_vlm/config.py` - Configuration dataclasses (includes AgenticConfig)
- `src/nova_retrieval_vlm/types.py` - Type definitions with jaxtyping/beartype
- `src/nova_retrieval_vlm/agentic/` - Agentic processing module:
  - `processor.py` - Core AgenticProcessor with multi-turn reasoning
  - `tools.py` - ToolRegistry with visual tools (zoom, crop, contrast, threshold)
  - `localization.py` - AgenticLocalizationProcessor
  - `diagnosis.py` - AgenticDiagnosisProcessor
  - `retrieval_manager.py` - RetrievalManager for knowledge retrieval
- `src/nova_retrieval_vlm/processors/` - Task processors (localization, diagnosis, caption)
- `src/nova_retrieval_vlm/models/openai_adapter.py` - Model API interface
- `src/nova_retrieval_vlm/evaluation/` - Evaluation metrics (NOVA benchmark protocol)

## Architecture & Design Principles

### Modern Code Standards (ENFORCED)
- **No AI slop**: No "robust" fallbacks, "best-effort" parsers, or overly defensive code
- **Fail fast**: Use proper exception handling instead of silent fallbacks
- **Type safety**: jaxtyping for tensor shapes, beartype for runtime validation
- **Modern tooling**: uv for deps, ruff for lint/format, pyright for types
- **Clean imports**: No try/except import patterns for optional dependencies

### Architecture Patterns
- **Processor pattern**: Task-specific processors instead of monolithic CLI
- **Dependency injection**: Clear interfaces and testable components
- **Modern Python**: Use new union syntax (X | Y), type hints, dataclasses
- **Structured data**: Pydantic models for validation, proper error types

### Performance & Reliability
- **Memory management**: Explicit cleanup for image processing
- **Async/await**: Consistent async patterns throughout
- **Batch processing**: Efficient vectorized operations where possible

## Common Development Tasks

### Adding a New Task Processor
1. Create processor in `src/nova_retrieval_vlm/processors/`
2. Inherit from `BaseProcessor` with proper type annotations
3. Implement `process_batch()` and `evaluate_responses()` methods
4. Add `@beartype` decorators for runtime validation
5. Register in `PROCESSORS` dict in `cli_new.py`

### Adding a New Model Adapter
1. Create adapter in `src/nova_retrieval_vlm/models/`
2. Use proper type hints with `ModelResponse` return type
3. Add `@beartype` validation for all methods
4. No fallback mechanisms - fail fast on errors

### Adding Type-Safe Functions
1. Use jaxtyping for tensor shapes: `Float[torch.Tensor, "batch height width"]`
2. Add `@beartype` for runtime validation
3. Use modern union syntax: `str | None` instead of `Optional[str]`
4. Define custom types in `types.py` for reuse

### Code Quality Requirements
- All functions must have type hints and `@beartype` decorators
- No broad exception handling - use specific exception types
- Use `fd` and `ast-grep` for code analysis instead of `find`/`grep`
- Follow ruff rules - no ignored violations

## Performance Considerations
- **Batch processing**: Use `batch_size` parameter
- **Rate limiting**: Configure `request_delay`
- **Memory**: Enable image compression for large batches
- **Caching**: Results cached in output directory

## Testing Strategy
```bash
# Modern test execution with uv
uv run pytest tests/test_models.py         # Unit tests
uv run pytest tests/test_processors.py     # Processor tests
uv run pytest tests/test_types.py          # Type validation tests

# Integration tests with proper fixtures
uv run pytest tests/integration/ -v

# Performance and benchmark tests
uv run pytest tests/benchmarks/ --benchmark-only

# Type checking validation
uv run pytest tests/ --mypy-only           # Deprecated, use pyright
uv run pyright tests/                      # Modern type checking

# Coverage reporting
uv run pytest --cov=nova_retrieval_vlm --cov-report=html
```

## Debugging & Analysis Tips
```bash
# Modern debugging with structured logging
export LOGURU_LEVEL=DEBUG
uv run python -m nova_retrieval_vlm.cli_new task=localization --verbose

# Code analysis with modern tools
fd -e py . src/ | head -10                  # Find Python files
ast-grep --lang python -p 'def $func($$$)' src/  # Find function definitions
uv run ruff check --select E,W,F .         # Focused linting

# Type checking and validation
uv run pyright src/nova_retrieval_vlm/     # Static type checking
uv run python -c "from nova_retrieval_vlm.types import *"  # Runtime validation

# Performance profiling
uv run python -m cProfile -o profile.out -m nova_retrieval_vlm.cli_new

# Memory debugging
uv run python -m tracemalloc -c "import nova_retrieval_vlm"
```

## Configuration Management
- Main config: `src/nova_retrieval_vlm/config.py`
- Hydra configs: Override via CLI or YAML files
- Environment: `.env` file for API keys
- Logging: `LOGURU_LEVEL` environment variable

## API Keys Required
- `OPENROUTER_API_KEY` - For OpenRouter models
- `OPENAI_API_KEY` - Optional, for OpenAI direct access

## Data Paths
- Input data: `./data/nova/` (configurable)
- Output runs: `./runs/` (configurable)
- Indexes: `./indexes/` (for retrieval)

## Contact & Resources
- GitHub Issues: Report bugs and feature requests
- Documentation: `./docs/` directory
- Paper: `./paper/main.tex`
- Contributing: See CONTRIBUTING.md

## Development Standards (ENFORCED)

### Required Tools & Practices
1. **uv**: All dependency management must use `uv sync`, `uv add`, `uv run`
2. **ruff**: Code formatting and linting - replaces black, isort, flake8
3. **pyright**: Type checking - replaces mypy for better performance
4. **jaxtyping**: Tensor shape annotations for all ML functions
5. **beartype**: Runtime type validation with `@beartype` decorator
6. **fd/ast-grep**: Modern code analysis tools over find/grep

### Code Quality Gates
- All commits must pass `uv run ruff check .`
- All functions require type hints and `@beartype`
- No "robust" parsers or silent fallbacks
- Use structured exceptions, never broad `except:`
- Processor pattern for all new task implementations

### Performance Requirements
- Use async/await consistently
- Explicit memory management for large tensors
- Vectorized operations where possible
- Proper resource cleanup (context managers)

### Migration Status
- ✅ Modern pyproject.toml with uv dependency groups
- ✅ Ruff configuration with strict rules
- ✅ Pyright configuration for type checking
- ✅ Processor pattern architecture
- ✅ Type-safe core modules with jaxtyping/beartype
- ✅ Agentic processing module with visual tools and retrieval integration
- ✅ Test suite (150+ tests)

---
*This file was created to help Claude understand the project structure and common tasks. Update it when making significant architectural changes.*