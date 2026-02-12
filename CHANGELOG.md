# Changelog

## [Unreleased]

### Removed
- Tool decorator wrappers (`radiant_harness.tools.tool`, `radiant_harness.tools.visual_tool`)
- `ToolDocumenter` helpers: `get_all_tools`, `get_categories`, `validate_all_tools`, `has_tool`, `count_tools`
- `ImageManager` helpers: `get_size`, `copy_current`
- `OpenAIAdapter.health_check`
- Verifiers utilities: `ToolBridge`, `create_tool_bridge`, `create_verifiers_rubric`, `create_verifiable_processor`, `wrap_processor_for_verifiers`
- `CombinedReward.get_rubric`
- `RadiantHarnessAdapter` params `registry` and `max_tool_calls`
- Built-in prompt examples under `radiant_harness.prompts.examples`
- `SearchResult.to_llm_dict`, `WebSearchManager.format_for_llm`
- `ImageSearchResult.to_dict`, `MedicalImageSearchManager.format_for_llm`
- `ToolRegistry.set_image` and async context manager hooks
- `ImageManager` async context manager hooks
- `ToolRegistry.aclose`
- Stale NOVA example configs and docs tied to removed Hydra workflow

### Changed
- HuggingFace adapters accept `stream` and fail fast with `ModelError` when streaming is requested
- HuggingFace adapters fail fast on non-`None` `response_format` instead of silently ignoring it
- `AgenticProcessorBase._create_tool_registry` no longer takes `active_image_index`
- Tool execution wraps unexpected exceptions in `AgenticProcessingError`
- Tool execution history uses a bounded deque for O(1) eviction
- Web search ranking reuses per-query values to avoid per-result recomputation
- HuggingFace tool-call parsing reuses a compiled regex and skips cleanup when no tool blocks are present
- Web search ranking uses a compiled publication-year regex
- Image search keyword extraction and extension lookup reuse module-level constants
- Search tool metadata accumulation uses sets instead of list-to-set conversions

## [0.2.0] - 2025-12-09

### Added

#### Examples
- **PubmedQA** (`examples/pubmedqa/`) -- text-only medical Q&A with yes/no/maybe answers, optional PubMed search
- **VQA-RAD** (`examples/vqa_rad/`) -- radiology VQA with closed and open-ended questions
- **GEMeX-ThinkVG** (`examples/gemex_thinkvg/`) -- visual grounding with chain-of-thought reasoning, three reward components (answer, location, bbox IoU), MIMIC-CXR integration
- **AgentClinic NEJM** (`examples/agentclinic_nejm/`) -- multi-turn diagnostic reasoning with information gathering via HISTORY/EXAM/TESTS/IMAGE requests

#### Verifiers Integration (`src/radiant_harness/verifiers/`)
- `BaseMultiTurnEnv` -- reusable base for multi-turn environments
- Reward functions: `ExactMatchReward`, `TokenF1Reward`, `IoUReward`, `CombinedReward`
- `RadiantHarnessAdapter` -- bridge between processors and verifiers
- Optional `rl` dependency group with `verifiers`, `datasets`, `transformers`, `torch`, `accelerate`

### Fixed
- Removed overly defensive exception handling in `image_manager.py`
- Removed dead code accessing private attributes in `registry.py`
- Simplified exception handling in registry execution path

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Core tool registry and visual tools
- OpenAI API adapter
- `AgenticProcessorBase` abstract class
- NOVA benchmark example
