# Radiant Harness - Audit Log

## Current Phase: Phase 0 Complete, Starting Phase 1

## Baseline Summary

### Toolchain Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v --tb=short

# Run linter
uv run ruff check src/

# Run type checker
uv run pyright src/

# Format code
uv run ruff format .

# Pre-commit hooks
pre-commit run --all-files
```

### Baseline Test Results

- **Tests**: 54 passed, 12 skipped (12.84s)
- **Ruff**: All checks passed
- **Pyright**: 0 errors, 40 warnings (mostly unknown type annotations)
- **Package Import**: OK

### Repo Map

```
nova_retrieval_vlm/
├── src/radiant_harness/          # Core framework (24 Python files)
│   ├── __init__.py               # Public API (~55 exports)
│   ├── __main__.py               # CLI entry point
│   ├── base.py                   # AgenticProcessorBase (~640 lines)
│   ├── cache.py                  # TTLCache implementation
│   ├── config.py                 # Configuration dataclasses
│   ├── exceptions.py             # Exception hierarchy
│   ├── protocols.py              # Protocol definitions
│   ├── types.py                  # Core types (ToolCall, ToolResult, etc.)
│   ├── models/                   # Model adapters
│   │   ├── adapter_protocol.py   # AdapterProtocol
│   │   ├── openai_adapter.py     # OpenAI/OpenRouter adapter
│   │   └── huggingface_adapter.py # HuggingFace adapter (lazy import)
│   ├── tools/                    # Tool system
│   │   ├── registry.py           # ToolRegistry, EncodedImage
│   │   ├── tool.py               # Tool class
│   │   ├── visual.py             # Visual tools (zoom, crop, etc.)
│   │   ├── search.py             # Search tools (PubMed, Open-i)
│   │   ├── image_manager.py      # Image loading/transformation
│   │   ├── image_ops.py          # Image operations
│   │   ├── decorators.py         # Tool decorators
│   │   └── tool_documenter.py    # Schema generation
│   ├── retrieval/                # External search
│   │   ├── web_search.py         # PubMed search
│   │   └── image_search.py       # Open-i search
│   ├── prompts/                  # Template loading
│   │   └── __init__.py           # Jinja template utilities
│   ├── verifiers/                # RL training integration
│   │   ├── adapter.py            # RadiantHarnessAdapter
│   │   ├── base.py               # BaseMultiTurnEnv
│   │   ├── mixin.py              # VerifiableProcessorMixin
│   │   ├── rewards.py            # Reward functions
│   │   └── tool_bridge.py        # Tool execution bridge
│   └── utils/
│       ├── iou.py                # IoU calculation
│       └── json_extract.py       # JSON extraction
├── examples/                     # Example implementations
│   ├── nova/                     # NOVA brain-MRI benchmark
│   ├── agentclinic_nejm/         # Diagnostic reasoning
│   ├── gemex_thinkvg/            # Visual grounding with RL
│   ├── pubmedqa/                 # Medical Q&A
│   └── vqa_rad/                  # Radiology VQA
├── tests/                        # Test suite (9 test files)
└── docs/                         # Documentation (1 file)
```

### Key Architectural Patterns

1. **AgenticProcessorBase**: Abstract base class with dependency injection for task-specific logic
2. **ToolRegistry**: Manages tool registration, execution, and image state
3. **AdapterProtocol**: Interface for model backends (OpenAI, HuggingFace)
4. **TTLCache**: Generic caching with automatic eviction
5. **Verifiers Integration**: RL training support via mixin and reward functions

---

## Issues Found (Prioritized)

### High Priority

| # | Severity | Location | Problem | Failure Mode | Status |
|---|----------|----------|---------|--------------|--------|
| 1 | High | `README.md:3-4` | Badge URLs point to non-existent `your-org` placeholder | Broken links, unprofessional | Pending |
| 2 | High | `README.md:72` | Project structure shows `nova_retrieval_vlm/` but actual root is `.` | Confusing documentation | Pending |
| 3 | High | `README.md:102-106` | Links to non-existent doc files (`docs/api.md`, `docs/tools.md`, `docs/adapters.md`) | 404 errors | Pending |
| 4 | High | `docs/` | Only contains `verifiers_integration.md`, missing core docs | Users cannot find documentation | Pending |

### Medium Priority

| # | Severity | Location | Problem | Failure Mode | Status |
|---|----------|----------|---------|--------------|--------|
| 5 | Medium | `pyproject.toml:56-57` | `typeCheckingMode = "off"` disables type checking | Type errors go undetected | Pending |
| 6 | Medium | `test_evaluation_metrics.py` | 12 tests skipped (missing fixtures/dependencies) | Reduced test coverage | Pending |
| 7 | Medium | `types.py:85-86` | Pyright warnings about partially unknown types in `tool_calls` and `tool_results` | Type safety gap | Pending |
| 8 | Medium | `Makefile:34` | Wrong coverage source: `nova_retrieval_vlm` vs `radiant_harness` | Coverage doesn't work | Pending |

### Low Priority

| # | Severity | Location | Problem | Failure Mode | Status |
|---|----------|----------|---------|--------------|--------|
| 9 | Low | `pyright` config | 40 warnings about partially unknown types | Reduced type safety | Pending |
| 10 | Low | `prompts/__init__.py:75` | `except Exception` is acceptable (wraps minijinja) | None (acceptable pattern) | N/A |
| 11 | Low | `json_extract.py:47,66` | `pass` in except blocks for failed JSON parsing | Acceptable (fallback parsing) | N/A |

---

## Patches Applied

| # | Goal | Files Changed | Tests | Notes |
|---|------|---------------|-------|-------|
| - | - | - | - | No patches applied yet |

---

## Next Steps (Phase 1)

1. Fix README documentation issues (broken links, incorrect paths)
2. Delete or fix non-existent doc references
3. Fix Makefile coverage source
4. Investigate skipped tests
5. Add type annotations to resolve pyright warnings in core types

---

## Open Questions

1. Should `typeCheckingMode` be turned on for stricter type checking?
2. Are the 12 skipped tests intentionally skipped or broken?
3. Is `paper-old/` directory still needed?

---

## Repo Contract (WIP)

### Code Standards
- Use `uv` for dependency management
- Use `ruff` for linting and formatting
- Use `pyright` for type checking (when enabled)
- Use `beartype` for runtime validation
- All exceptions should be specific (no bare `except:`)
- Fail fast on invalid inputs at boundaries

### Test Standards
- Tests must be in `tests/` directory
- Use pytest markers: `slow`, `integration`, `unit`, `edge_case`
- Integration tests require `-m integration` flag
- All tests must pass before merge

### Documentation Standards
- Keep CLAUDE.md up to date with project structure
- Update CHANGELOG.md for user-facing changes
- Doc links must point to existing files
