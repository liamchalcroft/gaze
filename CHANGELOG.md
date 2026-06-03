# Changelog

All notable changes to GAZE are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.1] - 2026-06-03

### Changed
- Relaxed `requires-python` from `>=3.10,<3.13` to `>=3.10`, so `pip install
  gaze-vlm` works on Python 3.13 and 3.14. The CI matrix now covers 3.10 through
  3.14 on Linux and 3.14 on macOS.

## [0.1.0] - 2026-06-02

First public release. GAZE (Grounded Agentic Zero-shot Evaluation) is a modular
framework for multi-turn agentic vision-language model systems, built for
medical image analysis. It was developed previously under a different internal
name; this is its first release on PyPI as `gaze-vlm`, so earlier internal
version history is not carried onto this version line.

### Added
- `AgenticProcessorBase`: a multi-turn agentic loop with JSON-structured
  tool-calling, schema validation, and automatic error recovery. Subclass it
  and implement four methods (`get_system_prompt`, `get_user_message`,
  `get_response_schema`, `validate_response`).
- `analyze()`: a high-level convenience function (and the underlying
  `SimpleProcessor`) for one-off analyses without defining a subclass.
- Tunable agentic loop via `AgenticConfig` (turn, token, and temperature
  defaults plus the nudge, idle-tool, and tool-content budgets), a settable
  `temperature`, an overridable `should_continue()` stop hook, and a direct
  `adapter=` argument alongside `adapter_factory`.
- 25 built-in tools: 23 visual-manipulation tools (zoom, crop, contrast,
  windowing, thresholding, edge detection, morphology, and more) and 2
  retrieval tools (PubMed via NCBI E-utilities, Open-i image search).
- Model adapters: OpenAI / OpenRouter, LM Studio (local models), and
  HuggingFace Transformers.
- Frozen result types (`AgenticResult`, `Turn`, `ToolCall`, `ToolResult`) and a
  ContextVar-based configuration system (`config_context`) for task-scoped
  overrides.
- `GazeError` exception hierarchy and `@beartype` runtime validation on the
  public API. The package ships a `py.typed` marker.
- Verifiers integration for RL: reward functions and multi-turn environments.
- Five example applications (NOVA brain-MRI, GEMeX visual grounding, AgentClinic
  NEJM, PubMedQA, VQA-RAD) and a standalone MedMarks-compatible NOVA environment.
- mkdocs documentation site with API reference, a tool reference, and a
  configuration guide.

### Security
- Outbound retrieval is constrained by host allowlists with DNS-resolution IP
  rejection and streaming response-size caps (SSRF mitigation).
- API credentials are scrubbed from logged exception messages.
- Tool results are wrapped in randomized boundary markers to contain
  prompt-injection from external content.
- Image decoding gates on header dimensions before allocating the pixel buffer,
  with a process-wide `MAX_IMAGE_PIXELS` backstop (decompression-bomb mitigation).
