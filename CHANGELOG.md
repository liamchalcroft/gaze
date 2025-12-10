# Changelog

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