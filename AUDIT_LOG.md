# Radiant Harness - Audit Log

## Current Phase: Phase 2 Complete, Starting Phase 3

## Baseline Summary

### Toolchain Commands

```bash
# Install dependencies
uv sync

# Run all quality checks
make check

# Run tests
uv run pytest tests/ -v --tb=short

# Run linter
uv run ruff check src/

# Run type checker
uv run pyright src/

# Format code
uv run ruff format .
```

### Baseline Test Results

- **Tests**: 54 passed, 12 skipped (12.84s)
- **Ruff**: All checks passed
- **Pyright**: 0 errors, 40 warnings (mostly unknown type annotations)
- **Package Import**: OK

### Repo Map

```
radiant-harness/
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
| 1 | High | `README.md:3-4` | Badge URLs point to non-existent `your-org` placeholder | Broken links | **Fixed** |
| 2 | High | `README.md:72` | Project structure shows `nova_retrieval_vlm/` but actual root is `.` | Confusing documentation | **Fixed** |
| 3 | High | `README.md:102-106` | Links to non-existent doc files | 404 errors | **Fixed** |
| 4 | High | `Makefile` | References non-existent scripts and wrong paths | Commands fail | **Fixed** |

### Medium Priority

| # | Severity | Location | Problem | Failure Mode | Status |
|---|----------|----------|---------|--------------|--------|
| 5 | Medium | `pyproject.toml:56-57` | `typeCheckingMode = "off"` disables type checking | Type errors go undetected | Noted |
| 6 | Medium | `test_evaluation_metrics.py` | 12 tests skipped (torch not installed) | Reduced test coverage | Expected |
| 7 | Medium | `types.py:116` | Pyright warning about untyped set | Type safety gap | **Fixed** |
| 8 | Medium | `Makefile:34` | Wrong coverage source: `nova_retrieval_vlm` vs `radiant_harness` | Coverage doesn't work | **Fixed** |

### Low Priority

| # | Severity | Location | Problem | Failure Mode | Status |
|---|----------|----------|---------|--------------|--------|
| 9 | Low | `pyright` config | 40 warnings about partially unknown types | Reduced type safety | Noted |
| 10 | Low | `prompts/__init__.py:75` | `except Exception` wraps minijinja | Acceptable pattern | N/A |
| 11 | Low | `json_extract.py:47,66` | `pass` in except blocks for JSON fallback | Acceptable pattern | N/A |

---

## Patches Applied

| # | Goal | Files Changed | Tests | Notes |
|---|------|---------------|-------|-------|
| 1 | Fix README broken links/structure | `README.md` | Pass | Removed placeholder badges, fixed paths |
| 2 | Fix Makefile naming and targets | `Makefile` | Pass | Updated to Radiant Harness, fixed coverage source |
| 3 | Remove broken Makefile targets | `Makefile` | Pass | Removed eval/analyze targets referencing non-existent scripts |
| 4 | Add type annotation | `types.py` | Pass | Fixed pyright warning in `get_tools_used()` |
| 5 | Create AUDIT_LOG.md | `AUDIT_LOG.md` | N/A | Tracking audit progress |

---

## Completed Phases

### Phase 0: Baseline (Complete)
- Built repo map
- Established toolchain commands
- Recorded baseline test results (54 passed, 12 skipped)

### Phase 1: Correctness and Invariant Hardening (Complete)
- Fixed README documentation issues
- Fixed Makefile naming and broken references
- Added type annotation to resolve pyright warning

### Phase 2: De-slop and De-bloat (Complete)
- Removed broken Makefile targets (eval, analyze, full-paper-workflow)
- Fixed `check` target to run actual quality commands
- No dead code found in core library

---

## Next Steps (Phase 3: Testing and Observability)

1. Review test coverage for critical paths
2. Ensure error messages are actionable
3. Check logging levels are appropriate

## Open Questions (Resolved)

1. **typeCheckingMode**: Left as-is per project conventions
2. **12 skipped tests**: Intentionally skipped (require torch which is optional)
3. **paper-old/** directory: Research material, not blocking code quality

---

## Repo Contract

### Code Standards
- Use `uv` for dependency management
- Use `ruff` for linting and formatting
- Use `pyright` for type checking
- Use `beartype` for runtime validation
- All exceptions should be specific (no bare `except:`)
- Fail fast on invalid inputs at boundaries

### Test Standards
- Tests in `tests/` directory
- Use pytest markers for categorization
- 54 core tests must pass
- Optional tests (torch) may be skipped

### Documentation Standards
- Keep CLAUDE.md up to date
- Doc links must point to existing files
- Update AUDIT_LOG.md for audit changes
