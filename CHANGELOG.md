# Changelog

## [Unreleased]

### Removed
- `radiant_harness.tools.tool` and `radiant_harness.tools.visual_tool` decorators (unused wrapper API)
- `ToolDocumenter` helpers: `get_all_tools`, `get_categories`, `validate_all_tools`, `has_tool`, `count_tools`
- `ImageManager` helpers: `get_size`, `copy_current`
- `OpenAIAdapter.health_check` (silent error swallowing and unused)
- Verifiers utilities: `ToolBridge`, `create_tool_bridge`, `create_verifiers_rubric`, `create_verifiable_processor`
- `CombinedReward.get_rubric` (unused helper)
- `RadiantHarnessAdapter` params `registry` and `max_tool_calls` (unused)
- Built-in prompt examples under `radiant_harness.prompts.examples` (unused templates)
- `SearchResult.to_llm_dict` and `WebSearchManager.format_for_llm` (unused formatting helpers)
- `ImageSearchResult.to_dict` and `MedicalImageSearchManager.format_for_llm` (unused formatting helpers)
- `ToolRegistry.set_image` and async context manager hooks (unused in repo)
- `ImageManager` async context manager hooks (unused in repo)
- `ToolRegistry.aclose` (redundant async close)
- Stale NOVA example configs, scripts, and docs tied to removed Hydra workflow

### Changed
- HuggingFace adapters now accept `stream` and fail fast with `ModelError` when streaming is requested
- HuggingFace adapters now fail fast on non-`None` `response_format` inputs instead of silently ignoring them
- `AgenticProcessorBase._create_tool_registry` no longer takes `active_image_index`
- Tool execution now wraps unexpected tool exceptions in `AgenticProcessingError`
- Tool execution history now uses a bounded deque for O(1) eviction
- Web search ranking now reuses per-query values to avoid per-result recomputation
- HuggingFace tool-call parsing reuses a compiled regex and skips cleanup when no tool blocks are present
- Web search ranking now uses a compiled publication-year regex to reduce per-result overhead
- Image search keyword extraction and extension lookup now reuse module-level constants to reduce per-call allocations
- Search tool metadata accumulation now uses sets to avoid redundant list-to-set conversions

## [0.2.0] - 2025-12-09

### Added

#### Examples
- **PubmedQA Example** (`examples/pubmedqa/`)
  - Text-only medical Q&A with yes/no/maybe answers
  - Supports optional PubMed search integration
  - Accuracy, F1, and per-class metrics

- **VQA-RAD Example** (`examples/vqa_rad/`)
  - Visual question answering for radiology
  - Supports closed and open-ended questions
  - Image caching from HuggingFace
  - Exact match and token F1 evaluation

- **GEMeX-ThinkVG Example** (`examples/gemex_thinkvg/`)
  - Visual grounding with chain-of-thought reasoning
  - Three verifiable reward components:
    - Answer semantic matching
    - Anatomical location matching
    - Bounding box IoU accuracy
  - Integration with MIMIC-CXR dataset
  - Training and evaluation scripts

- **AgentClinic NEJM Example** (`examples/agentclinic_nejm/`)
  - Multi-turn diagnostic reasoning
  - Information gathering via HISTORY/EXAM/TESTS/IMAGE requests
  - Requires at least one info request before diagnosis
  - NEJM clinical case dataset integration

#### Verifiers Integration (`src/radiant_harness/verifiers/`)
- **BaseMultiTurnEnv**: Reusable base class for multi-turn environments
- **Reward Functions**:
  - `ExactMatchReward`: String matching with normalization
  - `TokenF1Reward`: Token-level F1 scoring
  - `IoUReward`: Bounding box overlap calculation
  - `CombinedReward`: Weighted combination of multiple rewards
- **Adapter Utilities**:
  - `RadiantHarnessAdapter`: Bridge between processors and verifiers
  - `wrap_processor_for_verifiers`: Quick wrapper function
- **Optional Dependencies**:
  - Added `rl` dependency group with `verifiers`, `datasets`, `transformers`, `torch`, `accelerate`

### Changed
- Updated README to document all new examples
- Improved error messages and documentation throughout

### Fixed
- Removed overly defensive exception handling in `image_manager.py`
- Removed dead code accessing private attributes in `registry.py`
- Simplified exception handling in registry execution path

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of Radiant Harness
- Core tool registry and visual tools
- OpenAI API adapter
- Agentic processor base class
- NOVA benchmark example
