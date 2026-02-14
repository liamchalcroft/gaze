# Radiant Harness Review Prompts

Structured review prompts for code quality, architecture, correctness, and research integrity across the Radiant Harness framework. Each prompt is fully self-contained — just copy the single code block for your tool and paste it.

Every prompt includes the evidence/anti-reward-hacking preamble and the appropriate tool header baked in.

---

## Table of Contents

1. [High-Level Architecture & Strategy](#high-level-architecture--strategy)
2. [Core Framework (src/radiant_harness/)](#core-framework-srcradiant_harness)
3. [Examples (examples/)](#examples-examples)
4. [Environments (environments/)](#environments-environments)
5. [Cross-Cutting Concerns](#cross-cutting-concerns)

---

## High-Level Architecture & Strategy

### Full Framework Architecture Review

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors. Flag reward-hackable patterns and replace with stronger invariants.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations (framework ↔ examples, OpenAI adapter ↔ HuggingFace adapter, etc.).

Claude Code mode:
- Start by producing a brief repo map (packages, modules, dependency graph, key entrypoints) and list 5-10 highest-risk areas.
- Then ask up to 5 targeted questions only if truly blocking.
- Execute in small, reviewable patch sets. Avoid cross-file conflicts; one owner per file at a time.
- Output: (1) top findings table (max ~15 items), (2) patch set plan (3-8 PRs), (3) then implement Patch Set #1 with tests.
- Agent Teams: spawn teammates with disjoint scopes and each teammate: objective, directory scope, forbidden areas, and required outputs. Maintain a shared task list with owners; do not edit the same files in parallel.

---

You are a senior software architect reviewing a medical imaging AI research framework.

## Context

Radiant Harness is a modular framework for building multi-turn agentic vision-language model (VLM) systems for medical image analysis. The repository contains:
- Core framework: src/radiant_harness/ (base processor, tool system, model adapters, retrieval, prompts, verifiers, config, types)
- 5 example implementations: nova (brain MRI), gemex_thinkvg (chest X-ray visual grounding), agentclinic_nejm (diagnostic reasoning), pubmedqa (medical Q&A), vqa_rad (radiology VQA)
- 1 standalone environment: environments/nova_brain_mri/ (MedMarks leaderboard)
- Test suite: tests/ (16 modules)
- Tech stack: Python 3.10+, asyncio, OpenAI/OpenRouter API, HuggingFace, minijinja, beartype, verifiers (RL), PIL, httpx

## Task

### Phase 1: Map the System (do this first)
- Produce an inventory of all modules in src/radiant_harness/ and their public APIs
- Draw the dependency graph: base.py → config.py → types.py → exceptions.py, tools/ → models/ → retrieval/ → prompts/ → verifiers/
- Map cross-module boundaries: how examples/ depend on core, how environments/ depend on core
- Identify the top 5-10 highest-risk areas (clinical accuracy, tool execution safety, model adapter parity, schema validation, reward correctness)

### Phase 2: Prioritise Top Risks
For each risk area, investigate:

1. **Dependency Graph** — circular imports, tight coupling between modules, leaking abstractions (e.g. does base.py depend on concrete adapters?)
2. **Data Flow** — image input → ImageManager → tool execution → model API → response parsing → validation → AgenticResult; trace the full agentic loop in base.py
3. **Abstraction Quality** — AgenticProcessorBase abstract methods are sufficient for all 5 examples? AdapterProtocol cleanly separates OpenAI vs HuggingFace? ToolRegistry decoupled from specific tools?
4. **Type Safety** — beartype coverage on public APIs, pyright compliance, frozen dataclass invariants, Literal types used correctly
5. **Error Handling** — exception hierarchy complete (exceptions.py), no bare except, no swallowed errors, fail-fast in tool execution
6. **Technical Debt** — TODO/FIXME/HACK/XXX comments, dead code, unused imports, naming inconsistencies

### Phase 3: Verify with Checks
For the top 5 issues, provide a concrete validation step (test command, grep, build check).

## Output Format

1. **Top findings table** (max 15 items, ranked by severity: Critical > Important > Suggestion)
   - For each: path + symbol + snippet, description, validation step
2. **Dependency graph** (text diagram showing module relationships)
3. **Patch set plan** (3-8 PRs to address findings)

Stop after top 5 Critical + top 7 Important unless you find a patient-safety or correctness blocker.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors. Flag reward-hackable patterns and replace with stronger invariants.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations (framework ↔ examples, OpenAI adapter ↔ HuggingFace adapter, etc.).

Codex CLI mode:
- Don't write a long upfront plan or preamble. Start by exploring the repo, locating entrypoints, and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests. Prefer contract tests for adapter boundaries.
- When unsure, stop and propose the smallest experiment to decide.

---

You are a senior software architect reviewing a medical imaging AI research framework.

## Context

Radiant Harness is a modular framework for building multi-turn agentic vision-language model (VLM) systems for medical image analysis. The repository contains:
- Core framework: src/radiant_harness/ (base processor, tool system, model adapters, retrieval, prompts, verifiers, config, types)
- 5 example implementations: nova (brain MRI), gemex_thinkvg (chest X-ray visual grounding), agentclinic_nejm (diagnostic reasoning), pubmedqa (medical Q&A), vqa_rad (radiology VQA)
- 1 standalone environment: environments/nova_brain_mri/ (MedMarks leaderboard)
- Test suite: tests/ (16 modules)
- Tech stack: Python 3.10+, asyncio, OpenAI/OpenRouter API, HuggingFace, minijinja, beartype, verifiers (RL), PIL, httpx

## Task

### Phase 1: Map the System (do this first)
- Produce an inventory of all modules in src/radiant_harness/ and their public APIs
- Draw the dependency graph: base.py → config.py → types.py → exceptions.py, tools/ → models/ → retrieval/ → prompts/ → verifiers/
- Map cross-module boundaries: how examples/ depend on core, how environments/ depend on core
- Identify the top 5-10 highest-risk areas (clinical accuracy, tool execution safety, model adapter parity, schema validation, reward correctness)

### Phase 2: Prioritise Top Risks
For each risk area, investigate:

1. **Dependency Graph** — circular imports, tight coupling between modules, leaking abstractions
2. **Data Flow** — image input → ImageManager → tool execution → model API → response parsing → validation → AgenticResult
3. **Abstraction Quality** — AgenticProcessorBase sufficient for all examples? AdapterProtocol cleanly separates adapters? ToolRegistry decoupled?
4. **Type Safety** — beartype coverage, pyright compliance, frozen dataclass invariants
5. **Error Handling** — exception hierarchy complete, no bare except, fail-fast
6. **Technical Debt** — TODO/FIXME/HACK/XXX, dead code, naming inconsistencies

### Phase 3: Verify with Checks
For the top 5 issues, provide a concrete validation step (test command, grep, build check).

## Output Format

1. **Top findings table** (max 15 items, ranked by severity: Critical > Important > Suggestion)
   - For each: path + symbol + snippet, description, validation step
2. **Dependency graph** (text diagram showing module relationships)
3. **Patch set plan** (3-8 PRs to address findings)

Stop after top 5 Critical + top 7 Important unless you find a patient-safety or correctness blocker.
```

---

### Medical Safety & Clinical Accuracy Review

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors. Flag reward-hackable patterns and replace with stronger invariants.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations.

Claude Code mode:
- Start by producing a brief repo map and list 5-10 highest-risk areas.
- Then ask up to 5 targeted questions only if truly blocking.
- Execute in small, reviewable patch sets.
- Output: (1) top findings table (max ~15 items), (2) patch set plan (3-8 PRs), (3) then implement Patch Set #1 with tests.

---

You are a clinical informatics specialist reviewing a medical imaging AI framework for patient safety and clinical accuracy.

## Context

Radiant Harness builds multi-turn agentic VLM systems for medical image analysis. Key clinical surfaces:
- src/radiant_harness/base.py — AgenticProcessorBase runs multi-turn analysis loops over medical images with tool use and model interaction
- src/radiant_harness/retrieval/web_search.py — PubMed search with ranking, used to ground model outputs in medical literature
- src/radiant_harness/retrieval/image_search.py — Open-i medical image search for reference images
- src/radiant_harness/tools/visual.py — Image manipulation tools (zoom, crop, contrast, threshold) applied to medical images
- src/radiant_harness/verifiers/rewards.py — Reward functions (ExactMatch, TokenF1, IoU, Combined) used for RL training on clinical tasks
- examples/nova/ — Brain MRI analysis: captioning, diagnosis, localization with evaluation metrics
- examples/gemex_thinkvg/ — Chest X-ray visual grounding with anatomical location matching
- examples/agentclinic_nejm/ — Clinical diagnostic reasoning with NEJM cases
- examples/pubmedqa/ — Medical Q&A with PubMed context
- examples/vqa_rad/ — Radiology visual question answering

Target users: medical imaging researchers. Authoritative sources: radiology literature, PubMed, established medical imaging benchmarks.

## Task

### Phase 1: Map Safety Boundaries
Identify every point where clinical content is generated, evaluated, or used for training. Map the pipeline: image input → tools → model generation → response parsing → evaluation/reward.

### Phase 2: Investigate (prioritise by patient risk)
1. **Reward Function Correctness** — verify rewards.py: ExactMatchReward, TokenF1Reward, IoUReward produce correct scores. Check extract_completion_text() handles all completion formats. Are rewards reward-hackable (e.g. can a model game IoU by predicting full-image boxes)?
2. **NOVA Evaluation Accuracy** — review examples/nova/src/evaluation/ (caption.py, diagnosis.py, detection.py). Do BLEU/METEOR/CIDEr metrics match standard implementations? Does mAP calculation follow COCO conventions? Are IoU thresholds clinically appropriate?
3. **Schema Validation** — do response schemas (get_response_schema()) enforce clinically necessary fields? Does validate_response() catch malformed outputs? Can a model bypass schema with free-text?
4. **Tool Safety** — visual tools (zoom, crop, contrast, threshold) preserve diagnostic information? Can aggressive thresholding destroy clinically relevant features? Are bounds in ImageProcessingConfig appropriate?
5. **Search Quality** — PubMed ranking (web_search.py RankingWeights) appropriate for clinical queries? Does reliability_score reflect evidence quality? Are search results correctly attributed?
6. **Prompt Engineering** — do system/task prompts (src/radiant_harness/prompts/ and examples/*/prompts/) include appropriate disclaimers? Do they avoid encouraging confabulation? Is the "continue" field in response format robust?
7. **RL Training Safety** — verifiers integration (verifiers/, BaseMultiTurnEnv): can reward hacking occur during training? Are reward signals aligned with clinical correctness (not just format compliance)?

### Phase 3: Validate
For each finding, provide a concrete validation step (test to write, metric to check against reference, formula to verify).

## Output Format

Top findings (max 15), classified as: **Patient Safety Risk** (immediate fix), **Evaluation Integrity** (fix before benchmarking), or **Improvement** (strengthen over time). Each with: path + symbol + snippet, description, validation step.

Stop after all Patient Safety Risks + top 5 Evaluation Integrity issues.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors. Flag reward-hackable patterns and replace with stronger invariants.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations.

Codex CLI mode:
- Don't write a long upfront plan or preamble. Start by exploring the repo, locating entrypoints, and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests.
- When unsure, stop and propose the smallest experiment to decide.

---

You are a clinical informatics specialist reviewing a medical imaging AI framework for patient safety and clinical accuracy.

## Context

Radiant Harness builds multi-turn agentic VLM systems for medical image analysis. Key clinical surfaces:
- src/radiant_harness/base.py — AgenticProcessorBase runs multi-turn analysis loops
- src/radiant_harness/retrieval/web_search.py — PubMed search with ranking
- src/radiant_harness/tools/visual.py — Image manipulation tools applied to medical images
- src/radiant_harness/verifiers/rewards.py — Reward functions for RL training on clinical tasks
- examples/nova/ — Brain MRI: captioning, diagnosis, localization
- examples/gemex_thinkvg/ — Chest X-ray visual grounding
- examples/agentclinic_nejm/ — Clinical diagnostic reasoning
- examples/pubmedqa/ — Medical Q&A
- examples/vqa_rad/ — Radiology VQA

## Task

### Phase 1: Map Safety Boundaries
Identify every point where clinical content is generated, evaluated, or used for training.

### Phase 2: Investigate (prioritise by patient risk)
1. **Reward Function Correctness** — verify rewards.py scores. Are rewards reward-hackable?
2. **NOVA Evaluation Accuracy** — do metrics match standard implementations? Are IoU thresholds appropriate?
3. **Schema Validation** — can a model bypass schema with free-text?
4. **Tool Safety** — can aggressive thresholding destroy clinically relevant features?
5. **Search Quality** — PubMed ranking appropriate for clinical queries?
6. **Prompt Engineering** — do prompts avoid encouraging confabulation?
7. **RL Training Safety** — can reward hacking occur during training?

### Phase 3: Validate
For each finding, provide a concrete validation step.

## Output Format

Top findings (max 15), classified as: **Patient Safety Risk**, **Evaluation Integrity**, or **Improvement**. Each with: path + symbol + snippet, description, validation step.
```

---

### Security & API Key Safety Review

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Claude Code mode:
- Start by producing a brief repo map and list 5-10 highest-risk areas.
- Then ask up to 5 targeted questions only if truly blocking.
- Output: (1) top findings table (max ~15 items), (2) patch set plan (3-8 PRs), (3) then implement Patch Set #1 with tests.

---

You are a security engineer reviewing a research framework that makes API calls with user credentials.

## Context

Radiant Harness makes external API calls and handles sensitive credentials:
- src/radiant_harness/models/openai_adapter.py — OpenAI/OpenRouter API calls with API keys (OPENROUTER_API_KEY, OPENAI_API_KEY)
- src/radiant_harness/retrieval/web_search.py — PubMed/NCBI API calls (NCBI_API_KEY, NCBI_EMAIL)
- src/radiant_harness/retrieval/image_search.py — Open-i API calls
- src/radiant_harness/cache.py — TTLCache stores API responses in memory
- src/radiant_harness/config.py — Configuration with API base URLs
- External dependencies: openai, httpx, aiohttp, requests

Attack surfaces: API key exposure in logs/errors, prompt injection via tool results, SSRF via configurable base URLs, cache poisoning, dependency vulnerabilities.

## Task

### Phase 1: Map Attack Surfaces
Inventory all external API calls, credential sources, user-controllable inputs (image paths, metadata, search queries).

### Phase 2: Investigate Top Risks
1. **API Key Safety** — keys never logged (check loguru calls near API usage), not in error messages, not in AgenticResult, not serialised to disk
2. **Input Validation** — image paths validated (path traversal?), search queries sanitised, metadata not injected into prompts unsafely
3. **External API Calls** — base URLs configurable (SSRF risk?), timeouts set, retries bounded (tenacity config), HTTP responses validated
4. **Prompt Injection** — tool results inserted into conversation: can a crafted search result inject system instructions? Can image metadata carry instructions?
5. **Cache Safety** — TTLCache in-memory only (no disk serialisation), no sensitive data in cache keys, eviction works correctly
6. **Dependency Security** — openai, httpx, aiohttp, PIL, minijinja versions pinned? Known CVEs?
7. **OWASP Top 10** — Injection (prompt/path), Broken Auth (API key management), Sensitive Data Exposure (logs), SSRF (configurable URLs)

### Phase 3: Validate
For each finding, provide a concrete remediation step and grep/test command to verify.

## Output Format

Top findings (max 15), rated: **Critical** (exploitable now), **High** (exploitable with effort), **Medium** (defense-in-depth gap), **Low** (best practice). Each with: path + symbol + snippet, remediation.

Stop after all Critical + top 5 High.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Don't write a long upfront plan or preamble. Start by exploring the repo, locating entrypoints, and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests.
- When unsure, stop and propose the smallest experiment to decide.

---

You are a security engineer reviewing a research framework that makes API calls with user credentials.

## Context

Radiant Harness makes external API calls and handles sensitive credentials:
- src/radiant_harness/models/openai_adapter.py — OpenAI/OpenRouter API calls with API keys
- src/radiant_harness/retrieval/web_search.py — PubMed/NCBI API calls
- src/radiant_harness/retrieval/image_search.py — Open-i API calls
- src/radiant_harness/cache.py — TTLCache stores API responses in memory
- src/radiant_harness/config.py — Configuration with API base URLs

## Task

### Phase 1: Map Attack Surfaces
Inventory all external API calls, credential sources, user-controllable inputs.

### Phase 2: Investigate Top Risks
1. **API Key Safety** — keys never logged, not in error messages, not serialised
2. **Input Validation** — image paths, search queries, metadata sanitised
3. **External API Calls** — timeouts set, retries bounded, responses validated
4. **Prompt Injection** — tool results can't inject system instructions
5. **Cache Safety** — no sensitive data in cache keys, eviction correct
6. **Dependency Security** — versions pinned, known CVEs checked

### Phase 3: Validate
For each finding, provide a remediation step and grep/test command.

## Output Format

Top findings (max 15), rated by severity. Each with: path + symbol + snippet, remediation.
```

---

### Performance & Async Correctness Review

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Claude Code mode:
- Start by producing a brief repo map and list 5-10 highest-risk areas.
- Then ask up to 5 targeted questions only if truly blocking.
- Output: (1) top findings table (max ~15 items), (2) patch set plan (3-8 PRs), (3) then implement Patch Set #1 with tests.

---

You are a performance engineer auditing a medical imaging AI framework for async correctness, API efficiency, and image processing performance.

## Context

Radiant Harness performance-sensitive paths:
- Async: asyncio event loop in base.py agentic loop, async model adapters (openai_adapter.py, huggingface_adapter.py), async search (web_search.py, image_search.py)
- Image Processing: PIL-based operations in tools/visual.py, tools/image_ops.py, tools/image_manager.py (zoom, crop, contrast, threshold on medical images — potentially large DICOM-derived files)
- API Calls: OpenAI/OpenRouter model calls (latency-critical), PubMed search (network-bound), Open-i image search
- Caching: TTLCache in cache.py (in-memory, per-process)
- Serialisation: base64 image encoding in registry.py (encode_image), JSON schema generation in tool_documenter.py
- Batch Processing: examples run over datasets (nova/data/nova_dataset.py, gemex_thinkvg/src/dataset.py)

## Task

### Phase 1: Profile Hot Paths
Identify the top 5 latency-sensitive paths (model API round-trip, image encoding, tool execution chain, search queries, batch dataset iteration).

### Phase 2: Investigate
1. **Async Correctness** — blocking calls in async context (PIL operations in async tool executors?), proper await chains, no sync HTTP calls in async functions, event loop not blocked
2. **Image Processing** — large image handling (medical images can be 2048x2048+), unnecessary copies, base64 encoding size, JPEG quality settings, memory pressure during batch runs
3. **API Efficiency** — request batching, connection pooling (httpx/aiohttp), retry overhead (tenacity config), prompt caching in openai_adapter.py
4. **Caching** — TTLCache hit rates, cache key design, memory footprint for large result sets, eviction tuning
5. **Batch Performance** — dataset iteration in examples, parallelism (asyncio.gather vs sequential), memory during large evaluation runs

### Phase 3: Validate
For each finding, provide a measurement method (benchmark command, profiling approach, memory check).

## Output Format

Top findings (max 15), prioritised by user-visible impact. Each with: path + symbol + snippet, estimated impact, measurement method, fix.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol (fn/type/module), (b) a short quoted snippet (2-4 lines) or exact search string, and (c) a concrete validation step (test/repro/benchmark/contract check).
- If you can't verify something quickly, propose the smallest experiment/log/test to confirm or refute it.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Don't write a long upfront plan or preamble. Start by exploring the repo, locating entrypoints, and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests.
- When unsure, stop and propose the smallest experiment to decide.

---

You are a performance engineer auditing a medical imaging AI framework for async correctness, API efficiency, and image processing performance.

## Context

Performance-sensitive paths:
- Async event loop in base.py, async model adapters, async search
- PIL-based image ops on potentially large medical images
- OpenAI/OpenRouter API calls (latency-critical), PubMed/Open-i search
- TTLCache, base64 encoding, batch dataset processing

## Task

### Phase 1: Profile Hot Paths
Identify top 5 latency-sensitive paths.

### Phase 2: Investigate
1. **Async Correctness** — blocking in async context, proper await chains
2. **Image Processing** — large image handling, memory pressure
3. **API Efficiency** — connection pooling, retry overhead, prompt caching
4. **Caching** — TTLCache tuning, memory footprint
5. **Batch Performance** — dataset iteration parallelism, memory during evaluation

### Phase 3: Validate
For each finding, provide a measurement method.

## Output Format

Top findings (max 15), prioritised by impact. Each with: path + symbol + snippet, measurement method, fix.
```

---

## Core Framework (src/radiant_harness/)

### base.py — AgenticProcessorBase

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~12 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the core agentic processor — the central abstraction of a medical imaging AI framework.

## Context
src/radiant_harness/base.py contains:
- `ImageInput(path, label?, width?, height?, encoded?)` — image with lazy loading and dimension tracking
- `AgenticProcessorBase(ABC)` — abstract base for multi-turn VLM analysis
  - Constructor: model_name, use_tools, use_web_search, max_turns, reasoning_enabled, reasoning_effort, enable_caching, disabled_tools, adapter_factory, config
  - Abstract methods: get_system_prompt(), get_user_message(), get_response_schema(), validate_response()
  - Core loop: analyze() → _run_analysis() → _execute_tools() → model API → parse → validate → AgenticResult
  - Confidence: calculate_confidence() (base 0.5 + tool bonus up to 0.7)
  - Tool creation: _create_tool_registry() (overrideable)

Depends on: types.py (Turn, ToolCall, ToolResult, AgenticResult), config.py, exceptions.py, tools/, models/, prompts/

Used by all 5 examples (nova, gemex_thinkvg, agentclinic_nejm, pubmedqa, vqa_rad).

## Task

### Phase 1: Map
- Trace the full agentic loop: analyze() → _run_analysis() → model call → tool execution → response parsing → validation
- List all abstract methods and verify all 5 examples implement them
- Check the "continue" field logic: how does the framework decide to stop?

### Phase 2: Investigate
1. **Loop Correctness** — max_turns enforced? Off-by-one? What if model never sets continue=false? What if model returns invalid JSON?
2. **Tool Execution** — _execute_tools() handles errors gracefully? Unknown tools raise UnknownToolError? Tool results properly formatted for conversation?
3. **Response Parsing** — JSON extraction robust (uses utils/json_extract.py)? Schema validation catches all malformed outputs?
4. **Confidence Calculation** — calculate_confidence() formula appropriate? Can it be gamed? Does it reflect actual output quality?
5. **Abstract Method Contract** — are abstract method signatures sufficient? Do examples need to work around limitations?
6. **Image Handling** — ImageInput.load() handles missing files, corrupt images, very large images?

### Phase 3: Validate
Run `uv run pytest tests/test_agentic_processor.py tests/test_base_processor.py`. Grep for bare except in base.py.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix. Include a contract satisfaction check across all 5 examples.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests.

---

You are reviewing the core agentic processor — the central abstraction of a medical imaging AI framework.

## Context
src/radiant_harness/base.py contains AgenticProcessorBase — the abstract base for all multi-turn VLM analysis.
- Core loop: analyze() → _run_analysis() → model call → tool execution → response parsing → validation
- Used by all 5 examples.
- Depends on: types.py, config.py, exceptions.py, tools/, models/, prompts/

## Task

### Phase 1: Map
- Trace the full agentic loop end-to-end
- Verify all 5 examples implement all abstract methods

### Phase 2: Investigate
1. **Loop Correctness** — max_turns, continue field, invalid JSON handling
2. **Tool Execution** — error handling, unknown tools, result formatting
3. **Response Parsing** — JSON extraction robustness, schema validation
4. **Confidence Calculation** — formula appropriateness, gamability
5. **Image Handling** — missing files, corrupt images, large images

### Phase 3: Validate
Run `uv run pytest tests/test_agentic_processor.py tests/test_base_processor.py`.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix.
```

---

### tools/ — Tool System

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Claude Code mode:
- Start by mapping all tools, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~12 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the tool system for a medical imaging AI framework.

## Context
src/radiant_harness/tools/ provides the complete tool infrastructure:
- tool.py — Tool class definition (name, description, parameters, execute, category)
- registry.py — ToolRegistry (execution, state tracking), ToolDocumenter (schema generation), EncodedImage, encode_image()
- visual.py — 7 visual tools: zoom_image, crop_image, adjust_contrast, apply_threshold, flip_image, rotate_image + reset; factory: create_visual_tools()
- search.py — 2 search tools: search_web (PubMed), search_images (Open-i); factory: create_search_tools()
- image_manager.py — ImageManager for image state, undo history, operation tracking
- image_ops.py — Low-level PIL image operations
- tool_documenter.py — OpenAI function-calling schema generation

Tools are registered with ToolRegistry and executed during the agentic loop. Visual tools modify a shared ImageManager. All visual tool functions use @beartype.

## Task

### Phase 1: Map
- List all tools with their parameters and categories
- Trace: model emits ToolCall → registry.execute_tool() → tool function → ToolResult → back to conversation
- Check schema generation: do get_tool_schemas() outputs match actual tool signatures?

### Phase 2: Investigate
1. **Visual Tool Correctness** — zoom_image bounds (0.5-4.0x), crop_image normalised coords [0,1], contrast factor range, threshold bounds. Do these preserve diagnostic information? Can parameters outside expected ranges cause exceptions?
2. **Image State Management** — ImageManager undo/reset reliable? Concurrent access safe? Memory with many operations?
3. **Schema Generation** — tool_documenter.py produces valid OpenAI function-calling JSON? Parameters match actual function signatures?
4. **Search Tools** — PubMed/Open-i integration: error handling, timeout, result formatting, image encoding for search results
5. **Tool Documentation** — generate_prompt_documentation() produces clear, non-misleading descriptions for the model?
6. **Registry Design** — execute_tool() handles unknown tools (UnknownToolError), disabled tools, execution errors

### Phase 3: Validate
Run `uv run pytest tests/test_tool_registry.py tests/test_search_tools.py`. Test visual tools with edge-case parameters (0, negative, very large).

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing the tool system for a medical imaging AI framework.

## Context
src/radiant_harness/tools/ — Tool infrastructure:
- tool.py (Tool class), registry.py (ToolRegistry, ToolDocumenter), visual.py (7 visual tools), search.py (PubMed + Open-i), image_manager.py (state management), image_ops.py (PIL ops), tool_documenter.py (schema gen)

## Task

### Phase 1: Map
- List all tools with parameters. Trace execution path from ToolCall to ToolResult.

### Phase 2: Investigate
1. **Visual Tool Correctness** — parameter bounds, diagnostic info preservation
2. **Image State** — undo/reset, memory, concurrent access
3. **Schema Generation** — valid OpenAI function-calling JSON
4. **Search Tools** — error handling, timeouts, result formatting
5. **Registry Design** — unknown tools, disabled tools, execution errors

### Phase 3: Validate
Run `uv run pytest tests/test_tool_registry.py tests/test_search_tools.py`.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix.
```

---

### models/ — Model Adapters

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations (OpenAI ↔ HuggingFace).

Claude Code mode:
- Start by mapping both adapters, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~10 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the model adapter layer for a medical imaging AI framework.

## Context
src/radiant_harness/models/ provides:
- adapter_protocol.py — AdapterProtocol (generate_chat interface), GenerationLog (token tracking)
- openai_adapter.py — OpenAIAdapter: OpenAI/OpenRouter API, reasoning model support (o1/o3), prompt caching, structured output, tool calling, streaming
- huggingface_adapter.py — HuggingFaceAdapter (base), HuggingFaceVLMAdapter (vision-language): local inference via transformers

Both must implement:
```python
async generate_chat(messages, max_tokens, temperature, tools?, response_format?, stream?) -> tuple[str, list[dict]|None, GenerationLog] | AsyncIterator[str]
```

## Task

### Phase 1: Map
- List all methods on each adapter beyond the protocol interface
- Check protocol compliance: do both adapters satisfy AdapterProtocol exactly?
- Compare feature support: tool calling, streaming, structured output, vision input

### Phase 2: Investigate
1. **Protocol Compliance** — do both adapters return identical types for identical inputs? Feature gaps (does HuggingFace support tools? structured output?)?
2. **OpenAI Adapter** — API key from env (not hardcoded?), error handling (APIError with status codes?), retry logic, reasoning model special handling, prompt caching correctness
3. **HuggingFace Adapter** — model loading (memory management, GPU/CPU placement), tokenizer handling, vision input encoding, graceful degradation for unsupported features
4. **Streaming** — both adapters handle streaming correctly? Backpressure? Error mid-stream?
5. **Token Tracking** — GenerationLog accurate for both adapters? prompt_tokens, completion_tokens correct?

### Phase 3: Validate
Run `uv run pytest tests/test_huggingface_adapter.py`. Propose a parity test: same prompt → both adapters → compare response structure (not content).

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix. Include parity assessment.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Treat "find the parity harness (or create the smallest one)" as an explicit task whenever comparing implementations (OpenAI ↔ HuggingFace).

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Prefer contract tests for adapter boundaries.

---

You are reviewing the model adapter layer for a medical imaging AI framework.

## Context
src/radiant_harness/models/ — AdapterProtocol, OpenAIAdapter, HuggingFaceAdapter/HuggingFaceVLMAdapter.

## Task

### Phase 1: Map
- Protocol compliance check for both adapters. Feature support comparison.

### Phase 2: Investigate
1. **Protocol Compliance** — identical return types? Feature gaps?
2. **OpenAI Adapter** — API key safety, error handling, retry logic, reasoning models
3. **HuggingFace Adapter** — model loading, memory, vision input, unsupported features
4. **Streaming** — both correct? Error mid-stream?
5. **Token Tracking** — GenerationLog accurate?

### Phase 3: Validate
Run tests. Propose a parity test.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

---

### retrieval/ — Search & Retrieval

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~10 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the medical literature retrieval system for a VLM framework.

## Context
src/radiant_harness/retrieval/ provides external search integration:
- web_search.py — PubMedEngine with advanced ranking:
  - SearchResult: title, url, content, snippet, source, reliability_score, publication_date, author, journal, doi, content_type, medical_relevance, extracted_entities, citation_count, open_access, ranking_score
  - RankingWeights (from config.py): medical_relevance, recency, open_access, title/content/entity match, content_type_boosts
  - Caching via TTLCache
  - search_medical_literature() convenience function
- image_search.py — Open-i medical image search with modality filtering
  - search_medical_images() function

Used by search tools in tools/search.py, which wrap these as tool executors for the agentic loop.

## Task

### Phase 1: Map
- Trace: user query → PubMedEngine.search() → NCBI API → result parsing → ranking → caching → SearchResult
- Trace: image query → search_medical_images() → Open-i API → result parsing

### Phase 2: Investigate
1. **PubMed Ranking** — RankingWeights formula appropriate for medical queries? Does reliability_score correctly weight evidence types (guidelines > case reports)? Is recency decay reasonable?
2. **API Integration** — NCBI rate limiting respected? NCBI_API_KEY/NCBI_EMAIL handled securely? Timeout and retry config? XML parsing robust?
3. **Result Quality** — SearchResult fields populated correctly from API responses? Content preview truncation preserves meaning? Extracted entities useful?
4. **Open-i Search** — modality filtering works? Error handling for unavailable service? Result format consistent with SearchResult?
5. **Caching** — TTLCache appropriate for search results? Cache invalidation? Memory with many queries?
6. **Error Handling** — SearchError provides enough context? Graceful degradation when APIs unavailable?

### Phase 3: Validate
Run `uv run pytest tests/test_web_search.py tests/test_image_search.py tests/test_search_tools.py`.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing the medical literature retrieval system for a VLM framework.

## Context
src/radiant_harness/retrieval/ — PubMedEngine (web_search.py) with ranking and caching, Open-i image search (image_search.py).

## Task

### Phase 1: Map
- Trace full search pipelines for both PubMed and Open-i.

### Phase 2: Investigate
1. **PubMed Ranking** — formula appropriate? Evidence type weighting?
2. **API Integration** — rate limiting, credentials, timeouts, XML parsing
3. **Result Quality** — fields populated correctly? Content truncation?
4. **Caching** — appropriate TTL? Memory pressure?
5. **Error Handling** — graceful degradation?

### Phase 3: Validate
Run `uv run pytest tests/test_web_search.py tests/test_image_search.py`.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

---

### verifiers/ — RL Training Integration

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to reward-hacking vectors: can a model game the reward function without improving on the actual task?

Claude Code mode:
- Start by mapping all reward functions and environments, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~12 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the RL training integration for a medical imaging AI framework.

## Context
src/radiant_harness/verifiers/ provides RL training support:
- rewards.py — Reward function hierarchy:
  - BaseRewardFunction(ABC): __call__(prompt, completion, info) -> float
  - ExactMatchReward: binary 1.0/0.0
  - TokenF1Reward: token-level precision/recall/F1
  - IoUReward: Intersection over Union for bounding boxes
  - CombinedReward: weighted combination of multiple rewards
  - extract_completion_text(completion): extracts text from various formats
- base.py — BaseMultiTurnEnv(vf.MultiTurnEnv): template for multi-turn RL environments
  - __init__: cases, dataset_path, max_turns, name, log_dir
  - Abstract: build_initial_state(), is_completed(), env_response()
- mixin.py — VerifiableProcessorMixin: adds verifiers support to AgenticProcessorBase
- adapter.py — RadiantHarnessAdapter: integration bridge

Example environments:
- examples/nova/src/rewards.py — NOVAVerifiersReward
- examples/gemex_thinkvg/src/rewards/ — Answer, Location, BBox, Combined rewards
- examples/agentclinic_nejm/src/environment.py — AgentClinicEnv
- environments/nova_brain_mri/src/nova_brain_mri/rewards.py — MedMarks NOVA rewards

## Task

### Phase 1: Map
- List all reward functions (core + examples) and what they measure
- List all environments and their completion conditions
- Trace: training loop → environment → processor → model → response → reward

### Phase 2: Investigate
1. **Reward Correctness** — ExactMatchReward handles case sensitivity, whitespace? TokenF1Reward tokenization matches expected behavior? IoUReward handles degenerate boxes (zero area, negative coords)?
2. **Reward Hacking Vectors** — IoU: can model predict full-image box for guaranteed non-zero IoU? TokenF1: can model repeat all tokens from prompt? ExactMatch: does normalisation prevent trivial gaming?
3. **extract_completion_text()** — handles all verifiers completion formats? What if format changes?
4. **BaseMultiTurnEnv** — state management correct? Max turns enforced? Logging captures enough for debugging?
5. **Example Rewards** — NOVA rewards match schema expectations? GEMeX combined weights (0.4/0.3/0.3) well-motivated? AgentClinic binary reward too sparse?
6. **Mixin Integration** — VerifiableProcessorMixin correctly wraps AgenticProcessorBase? No method resolution order issues?

### Phase 3: Validate
Run `uv run pytest tests/test_verifiers_integration.py tests/test_nova_reward_schema_alignment.py`. Test IoUReward with degenerate inputs.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix. Flag all reward-hacking vectors explicitly.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to reward-hacking vectors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing the RL training integration for a medical imaging AI framework.

## Context
src/radiant_harness/verifiers/ — rewards.py (ExactMatch, TokenF1, IoU, Combined), base.py (BaseMultiTurnEnv), mixin.py (VerifiableProcessorMixin), adapter.py. Plus example rewards in nova, gemex_thinkvg, agentclinic_nejm, environments/nova_brain_mri.

## Task

### Phase 1: Map
- List all reward functions and environments.

### Phase 2: Investigate
1. **Reward Correctness** — edge cases (case, whitespace, zero-area boxes)
2. **Reward Hacking** — full-image IoU, token repetition, trivial matching
3. **extract_completion_text()** — format robustness
4. **BaseMultiTurnEnv** — state management, max turns
5. **Example Rewards** — schema alignment, weight motivation

### Phase 3: Validate
Run `uv run pytest tests/test_verifiers_integration.py tests/test_nova_reward_schema_alignment.py`.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix.
```

---

### Remaining Core Module Prompts (config, types, exceptions, cache, prompts, utils)

For the remaining core modules, all prompts follow the same dual-version pattern. Each has identical body content with the appropriate tool header prepended.

#### Template

For any module prompt below, prepend **one** of:

**Claude Code header:**

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table, (2) patch set plan, (3) implement Patch Set #1.

---
```

**Codex CLI header:**

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.
- Make PR-sized diffs; keep refactors staged behind tests.

---
```

Then paste the body:

#### config.py + types.py + exceptions.py — Core Type System

```
You are reviewing the core type system of a medical imaging AI framework.

## Context
- src/radiant_harness/config.py — Frozen dataclasses: ImageProcessingConfig, CacheConfig, SearchConfig, RankingWeights, AgenticConfig, HarnessConfig. Thread-safe global access via get_config()/set_config().
- src/radiant_harness/types.py — Core types: TurnRole, ToolCall, ToolResult, Turn, AgenticResult (with num_turns, tool_call_count, get_tools_used properties).
- src/radiant_harness/exceptions.py — Exception hierarchy: HarnessError → ToolExecutionError, TemplateError, UnknownToolError, AgenticProcessingError, SchemaValidationError, ModelError, APIError.

These are the foundation types used by every other module.

## Task

### Phase 1: Map
- List all public types and their fields
- Verify frozen dataclass invariants (no mutation after construction)
- Check __init__.py exports match actual public API

### Phase 2: Investigate
1. **Type Design** — ToolResult: is image_base64 the right way to pass images? Turn: does the role/content/tool_calls combination cover all conversation states? AgenticResult: are computed properties (num_turns, tool_call_count) correct?
2. **Config Safety** — frozen dataclasses truly immutable? Thread-safe global config (get_config/set_config) correct under concurrent access? Default values clinically appropriate (e.g. max_turns_limit, temperature)?
3. **Exception Quality** — all exceptions carry enough context for debugging? No bare except in codebase? From/Into conversions correct? Exceptions are Send-equivalent (no mutable state)?
4. **Serialisation** — types serialise/deserialise correctly? No data loss in ToolResult → JSON → ToolResult round-trip?

### Phase 3: Validate
Run `uv run pytest tests/test_utils.py`. Run `uv run pyright src/radiant_harness/types.py src/radiant_harness/config.py src/radiant_harness/exceptions.py`.

## Output Format
Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

#### cache.py — TTL Cache

```
You are reviewing the caching layer of a medical imaging AI framework.

## Context
src/radiant_harness/cache.py — TTLCache[T]: generic in-memory cache with automatic TTL-based expiration.
- get(key), set(key, value), has(key), stats() (hits, misses, hit_rate)
- Automatic eviction when size exceeds max_cache_size (configurable evict_ratio)
- Used by PubMedEngine for search result caching

## Task

### Phase 1: Map
- Trace all cache usage (grep for TTLCache across codebase)
- Check configuration: CacheConfig defaults (max_cache_size, cache_duration_seconds, evict_ratio)

### Phase 2: Investigate
1. **Correctness** — TTL expiration accurate? Eviction removes oldest entries? Race conditions under concurrent access (asyncio)?
2. **Memory** — what's cached (SearchResult objects can be large)? Memory growth bounded? Eviction ratio prevents thrashing?
3. **Cache Key Design** — keys unique and deterministic? No sensitive data in keys?
4. **Stats** — hit_rate calculation correct? Stats useful for tuning?

### Phase 3: Validate
Run `uv run pytest tests/test_cache.py`. Propose a stress test: many concurrent sets/gets with TTL expiration.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix.
```

#### prompts/ — Template System

```
You are reviewing the prompt template system for a medical imaging AI framework.

## Context
src/radiant_harness/prompts/__init__.py — Template loading via minijinja:
- AnalysisMode(Enum): "agentic" or "single_turn"
- load_template(), load_prompt(), load_system_prompt(), load_task_prompt(), combine_prompts(), create_prompt()
- Base templates: src/radiant_harness/prompts/{agentic,single_turn}/{system,task}.jinja
- Example templates: examples/*/src/prompts/{agentic,single_turn}/{system,task}.jinja

Templates use variables: domain_expertise, tool_documentation, analysis_workflow, task_instructions, image_info, context, output_format, image_path.

## Task

### Phase 1: Map
- List all template files (base + all examples)
- List all template variables used across templates
- Check: do all examples provide all required variables?

### Phase 2: Investigate
1. **Template Safety** — can template variables inject control sequences? Is minijinja auto-escaping appropriate for this use case (prompts, not HTML)?
2. **Template Quality** — do prompts avoid encouraging model confabulation? Do they include appropriate clinical framing? Is the "continue" response format clearly explained?
3. **Error Handling** — TemplateError raised for missing templates, missing variables, render failures? Error messages actionable?
4. **Extensibility** — is it easy for new examples to provide custom templates? Is the base/override pattern clear?

### Phase 3: Validate
Run `uv run pytest tests/test_prompts.py`. Render each template with minimal context and check for missing variable errors.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix.
```

#### utils/ — Utility Functions

```
You are reviewing utility functions for a medical imaging AI framework.

## Context
src/radiant_harness/utils/:
- iou.py — compute_iou(box1, box2) -> float: IoU for [x1,y1,x2,y2] boxes in [0,1] range. Used by IoUReward and NOVA detection evaluation.
- json_extract.py — extract_json_from_text(text) -> dict|None: Extracts JSON from model output text. Handles markdown code blocks, uses JSONDecoder.raw_decode(). Used by base.py for response parsing.

These are critical-path utilities: IoU affects reward scores, JSON extraction affects every model response.

## Task

### Phase 1: Map
- List all callers of compute_iou() and extract_json_from_text()
- Check: are these the only JSON/IoU implementations or are there duplicates?

### Phase 2: Investigate
1. **IoU Correctness** — handles zero-area boxes? Negative coordinates? Boxes outside [0,1]? Non-overlapping boxes return 0.0? Identical boxes return 1.0?
2. **JSON Extraction** — handles nested JSON? Multiple JSON objects in text? Malformed JSON (missing closing brace)? Empty input? Very large text?
3. **Edge Cases** — IoU with swapped corners (x2 < x1)? JSON with special characters, unicode, escape sequences?

### Phase 3: Validate
Run `uv run pytest tests/test_json_extract.py tests/test_utils.py`. Add edge-case tests for degenerate inputs.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix + proposed test.
```

---

## Examples (examples/)

### examples/nova — NOVA Brain MRI Benchmark

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to evaluation metric correctness — these numbers will appear in papers.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~12 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing the NOVA brain MRI benchmark implementation — the flagship example of a medical imaging AI framework.

## Context
examples/nova/ implements multi-task brain MRI analysis:
- src/processor.py — NOVAAgenticProcessor: 3 tasks (caption, diagnosis, localization), subclasses AgenticProcessorBase
- src/config.py — ConfidenceConfig dataclass
- src/types.py — NOVA-specific types
- src/schemas.py — Response schema definition and validation
- src/rewards.py — NOVAVerifiersReward for RL training
- src/cli.py — CLI entry point (task, model, data-dir, output-dir, use-tools, max-turns, reasoning, batch-size)
- src/prompts/ — Jinja templates for agentic and single-turn modes
- src/evaluation/ — Evaluation metrics:
  - caption.py: BLEU, METEOR, CIDEr for radiological captions
  - diagnosis.py: accuracy, F1 for primary + differential diagnosis
  - detection.py: mAP, IoU for lesion localization
- src/data/ — Dataset handling:
  - nova_dataset.py: NOVADataset (loads from HuggingFace c-i-ber/Nova)
  - nova_ground_truth.py: Ground truth annotation handling
  - transforms.py: Image transforms
- src/visualization/ — Streamlit GUI for result inspection
- src/utils/ — confidence_calibration_utils.py, statistical_analysis.py

## Task

### Phase 1: Map
- Trace each task: caption → system prompt → model call → response schema → evaluation metric
- Check: does each task (caption, diagnosis, localization) have correct schema, evaluation, and reward alignment?

### Phase 2: Investigate
1. **Evaluation Metric Correctness** — BLEU/METEOR/CIDEr in caption.py match standard implementations (nltk, pycocoevalcap)? mAP in detection.py follows COCO convention? IoU thresholds [0.5, 0.75] standard? Diagnosis accuracy handles multi-label correctly?
2. **Schema-Reward Alignment** — NOVAVerifiersReward computes rewards on fields that match get_response_schema()? Schema validation catches missing fields before reward computation?
3. **Prompt Quality** — do agentic/system.jinja and task.jinja for each task avoid encouraging hallucination? Do localization prompts clearly specify coordinate format? Do diagnosis prompts list valid classes?
4. **Dataset Integrity** — NOVADataset loads correctly from HuggingFace? Ground truth format matches evaluation expectations? Image transforms preserve spatial correctness (important for localization)?
5. **CLI Robustness** — cli.py handles missing data, invalid task names, API errors gracefully?

### Phase 3: Validate
Run `uv run pytest examples/nova/tests/test_nova_dataset_smoke.py`. Spot-check: compute BLEU on a known reference pair.

## Output Format

Top findings (max 12), classified as: **Evaluation Integrity** (affects paper numbers), **Clinical Accuracy** (affects clinical relevance), **Code Quality** (maintainability). Each with: path + symbol + snippet, severity, fix.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to evaluation metric correctness.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing the NOVA brain MRI benchmark — the flagship example.

## Context
examples/nova/ — 3 tasks (caption, diagnosis, localization). Processor, schemas, rewards, evaluation (BLEU/METEOR/CIDEr, accuracy/F1, mAP/IoU), dataset, Jinja prompts, CLI, Streamlit visualization.

## Task

### Phase 1: Map
- Trace each task end-to-end. Check schema/evaluation/reward alignment.

### Phase 2: Investigate
1. **Evaluation Metrics** — match standard implementations? Thresholds appropriate?
2. **Schema-Reward Alignment** — rewards match schema fields?
3. **Prompt Quality** — avoid hallucination encouragement?
4. **Dataset Integrity** — loads correctly? Transforms preserve spatial info?

### Phase 3: Validate
Run `uv run pytest examples/nova/tests/`.

## Output Format

Top findings (max 12). Each with: path + symbol + snippet, severity, fix.
```

---

### examples/gemex_thinkvg — Visual Grounding with RL

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to reward-hacking vectors in the combined reward.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~10 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing a visual grounding + RL training example for chest X-ray analysis.

## Context
examples/gemex_thinkvg/ implements visual grounding on MIMIC-CXR chest X-rays:
- src/processor.py — GEMeXProcessor (subclasses AgenticProcessorBase)
- src/dataset.py — GEMeXDataset (HuggingFace GEMeX-ThinkVG with MIMIC-CXR)
- src/schemas.py — ThinkVG schemas (XML/JSON response formats)
- src/rewards/ — Multi-component reward system:
  - answer.py: AnswerReward — semantic matching of findings (exact, contains, token F1)
  - location.py: LocationReward — anatomical region matching (hierarchical, synonym-aware)
  - bbox.py: BBoxReward — IoU-based spatial accuracy
  - combined.py: CombinedReward — weighted combination (default: 0.4 answer / 0.3 location / 0.3 bbox)
- src/verifiers/environment.py — MultiTurnEnv implementation
- train.py — RL training script
- eval.py — Evaluation script

## Task

### Phase 1: Map
- Trace: dataset → processor → model → schema parsing → reward decomposition → combined score
- List all reward components and their weight contributions

### Phase 2: Investigate
1. **Combined Reward Design** — weights 0.4/0.3/0.3 well-motivated? Does answer reward dominate? Can a model score high by getting answer right but location/bbox wrong?
2. **Answer Reward** — semantic matching (exact, contains, token F1): which is used when? Can model game contains-match by including all possible answers?
3. **Location Reward** — hierarchical anatomical matching: is the hierarchy medically correct? Synonym list complete (e.g. "right lower lobe" ↔ "RLL")?
4. **BBox Reward** — IoU handles degenerate boxes? Can model predict full-image box for non-zero reward?
5. **Schema Parsing** — XML/JSON parsing handles malformed outputs? ThinkVG format enforced?
6. **Environment** — multi-turn state management correct? Completion condition robust?

### Phase 3: Validate
Run existing tests. Test combined reward with adversarial inputs (full-image box, repeated tokens, all-inclusive answer).

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix. Flag all reward-hacking vectors.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- Pay special attention to reward-hacking vectors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing visual grounding + RL training for chest X-ray analysis.

## Context
examples/gemex_thinkvg/ — GEMeXProcessor, GEMeXDataset, rewards (Answer 0.4, Location 0.3, BBox 0.3), MultiTurnEnv.

## Task

### Phase 1: Map
- Trace dataset → processor → model → rewards.

### Phase 2: Investigate
1. **Combined Reward** — weight motivation, dominance patterns
2. **Individual Rewards** — answer gaming, location hierarchy, bbox degeneracy
3. **Schema Parsing** — XML/JSON robustness
4. **Environment** — state management, completion

### Phase 3: Validate
Test rewards with adversarial inputs.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

---

### Remaining Example Prompts (agentclinic_nejm, pubmedqa, vqa_rad)

For the remaining examples, all prompts follow the same dual-version pattern. Use the template headers from the core modules section, then paste the body.

#### examples/agentclinic_nejm — Clinical Diagnostic Reasoning

```
You are reviewing a clinical diagnostic reasoning environment for RL training.

## Context
examples/agentclinic_nejm/ implements multi-turn clinical case analysis with information gathering:
- src/environment.py — AgentClinicEnv(vf.MultiTurnEnv): assistant requests HISTORY/EXAM/TESTS/IMAGE, then diagnoses
- data/download.py — Dataset download script for NEJM clinical cases
- train.py — RL training script
- eval.py — Evaluation script

Interaction pattern: assistant asks for information (HISTORY, EXAM, TESTS, IMAGE) → environment provides → assistant diagnoses. Binary reward: 1.0 if correct, 0.0 otherwise.

## Task

### Phase 1: Map
- Trace: NEJM case → environment → assistant interaction → diagnosis → reward
- List all information types the assistant can request

### Phase 2: Investigate
1. **Environment Correctness** — information request parsing robust? All case fields accessible? Image integration correct?
2. **Reward Design** — binary reward too sparse for RL? Does it reward information gathering or just final diagnosis? Can model guess without requesting info?
3. **Dataset** — download.py handles network errors, partial downloads? Case format validated?
4. **Completion Logic** — how does environment detect diagnosis vs. info request? Can model exploit ambiguity?

### Phase 3: Validate
Run `cargo clippy` equivalent (`uv run ruff check examples/agentclinic_nejm/`). Test environment with mock cases.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix.
```

#### examples/pubmedqa — Medical Q&A

```
You are reviewing a medical Q&A example (text-only, no images).

## Context
examples/pubmedqa/ demonstrates text-only analysis:
- src/processor.py — PubmedQAProcessor + reward (subclasses AgenticProcessorBase without images)
- src/dataset.py — PubmedQADataset (yes/no/maybe answers)
- src/schemas.py — Response schema
- src/cli.py — CLI entry point

This example validates that the framework works for pure text tasks without requiring images.

## Task

### Phase 1: Map
- Trace: question + context → processor → model → answer → reward
- Check: how does PubmedQAProcessor handle the image-free case? Does base class require images?

### Phase 2: Investigate
1. **Image-Free Operation** — does AgenticProcessorBase gracefully handle no images? Are tools disabled correctly? Any assumptions about image availability?
2. **Schema Design** — yes/no/maybe schema appropriate? Does it capture confidence or reasoning?
3. **Reward** — exact match on yes/no/maybe: case-insensitive? Whitespace-tolerant?
4. **Dataset** — PubmedQA loading correct? Context properly formatted?

### Phase 3: Validate
Run `uv run ruff check examples/pubmedqa/`. Test processor with a minimal case.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix.
```

#### examples/vqa_rad — Radiology VQA

```
You are reviewing a radiology visual question answering example.

## Context
examples/vqa_rad/ implements VQA on radiology images:
- src/processor.py — VQARadProcessor (subclasses AgenticProcessorBase)
- src/dataset.py — VQARadDataset
- src/schemas.py — Response schema
- src/evaluation.py — Evaluation metrics (accuracy, BLEU, CIDEr)
- src/cli.py — CLI entry point

## Task

### Phase 1: Map
- Trace: image + question → processor → model → answer → evaluation
- Check: evaluation metrics appropriate for VQA (open-ended vs. closed-ended questions)?

### Phase 2: Investigate
1. **Evaluation** — accuracy metric handles both open and closed questions? BLEU/CIDEr appropriate for short answers?
2. **Schema** — response format captures question type (yes/no, what, where, how many)?
3. **Dataset** — VQA-RAD loading, image-question pairing correct?
4. **Processor** — prompt templates guide model appropriately for different question types?

### Phase 3: Validate
Run `uv run ruff check examples/vqa_rad/`. Test evaluation metrics with known answer pairs.

## Output Format
Top findings (max 8). Each with: path + symbol + snippet, severity, fix.
```

---

## Environments (environments/)

### environments/nova_brain_mri — MedMarks Environment

#### Claude Code

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.
- This is a standalone package for external evaluation — correctness is critical.

Claude Code mode:
- Start by mapping, then ask up to 5 questions if blocking.
- Output: (1) top findings table (max ~10 items), (2) patch set plan, (3) then implement Patch Set #1 with tests.

---

You are reviewing a standalone MedMarks evaluation environment for brain MRI analysis.

## Context
environments/nova_brain_mri/ is a standalone Python package for the MedMarks leaderboard:
- src/nova_brain_mri/__init__.py — load() factory, NOVABrainMRIEnv (subclasses vf.MultiTurnEnv), NOVAEnvConfig
- src/nova_brain_mri/__main__.py — Entry point
- src/nova_brain_mri/cli.py — CLI
- src/nova_brain_mri/rewards.py — NOVA reward functions for MedMarks scoring
- pyproject.toml — Standalone package configuration (separate from main repo)

This package must work independently of the main radiant_harness package. It's consumed by the MedMarks evaluation framework.

## Task

### Phase 1: Map
- Check independence: does this package import from radiant_harness? (It shouldn't — it's standalone)
- Trace: MedMarks → load() → NOVABrainMRIEnv → multi-turn interaction → rewards → scores
- Compare rewards.py here vs examples/nova/src/rewards.py — are they consistent?

### Phase 2: Investigate
1. **Independence** — no imports from src/radiant_harness/? Own pyproject.toml with correct deps?
2. **Reward Consistency** — rewards.py matches examples/nova/src/rewards.py in logic? Any drift?
3. **Environment Contract** — NOVABrainMRIEnv satisfies vf.MultiTurnEnv interface? load() factory returns correct type?
4. **Config** — NOVAEnvConfig appropriate defaults? All required fields?
5. **Edge Cases** — handles missing data, malformed model responses, timeout gracefully?

### Phase 3: Validate
Check pyproject.toml deps. Diff rewards.py against examples/nova/src/rewards.py.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix. Include a reward consistency assessment.
```

#### Codex CLI

```
Important (evidence + anti-reward-hacking):
- Prefer falsifiable checks over opinions. Every claim must include: (a) path + symbol, (b) a short quoted snippet, and (c) a concrete validation step.
- Do not "pass" by weakening checks, deleting validation, loosening types, or silencing errors.

Codex CLI mode:
- Start by exploring the repo and building an evidence-backed issue list.
- Work in tight iterations: investigate → change → run the narrowest checks → summarise results.

---

You are reviewing a standalone MedMarks evaluation environment.

## Context
environments/nova_brain_mri/ — standalone package: NOVABrainMRIEnv, load() factory, rewards.py, separate pyproject.toml.

## Task

### Phase 1: Map
- Verify independence from radiant_harness. Compare rewards vs examples/nova/.

### Phase 2: Investigate
1. **Independence** — no cross-imports?
2. **Reward Consistency** — logic matches examples/nova?
3. **Environment Contract** — satisfies vf.MultiTurnEnv?
4. **Edge Cases** — missing data, malformed responses?

### Phase 3: Validate
Diff rewards files. Check pyproject.toml.

## Output Format

Top findings (max 10). Each with: path + symbol + snippet, severity, fix.
```

---

## Cross-Cutting Concerns

Cross-cutting prompts follow the same dual pattern. Use the template headers from the core modules section, then paste the body.

### Test Coverage Audit

```
You are auditing test coverage for a medical imaging AI framework.

## Context
- Core tests: tests/ (16 modules covering agentic processor, tools, cache, prompts, search, verifiers, utils, evaluation metrics, JSON extraction, HuggingFace adapter, reward schema alignment)
- Example tests: examples/nova/tests/test_nova_dataset_smoke.py
- Config: pytest in pyproject.toml, testpaths=["tests"], markers=[slow, integration, unit], coverage fail_under=60
- Dev deps: pytest, pytest-cov, pytest-asyncio, pytest-mock, pytest-benchmark

## Task

### Phase 1: Map
- Count test files and test functions per module
- Cross-reference: for each src/radiant_harness/*.py, is there a corresponding test file?
- Identify the top 5 most critical untested paths

### Phase 2: Investigate
1. **Core Coverage** — base.py agentic loop fully tested? All error paths? Max turns? Invalid JSON? Tool execution failures?
2. **Tool Coverage** — visual tools tested with edge-case parameters? Search tools tested with mocked APIs? Registry tested with unknown tools?
3. **Adapter Coverage** — OpenAI adapter tested with mock API? HuggingFace adapter tested with mock model? Streaming tested?
4. **Example Coverage** — NOVA evaluation metrics tested against known values? Other examples have no tests — critical gap?
5. **Async Testing** — pytest-asyncio used correctly? All async functions tested? Event loop handling?
6. **Integration** — cross-module integration tests? End-to-end with mock model?

### Phase 3: Propose
For each critical gap, propose the smallest test to close it.

## Output Format
| Module | Test File | Test Count | Critical Gap | Proposed Test |

Stop after top 10 gaps.
```

### Documentation Audit

```
You are auditing documentation completeness for a medical imaging AI framework.

## Context
- docs/verifiers_integration.md — RL training integration guide
- docs/MEDMARKS_INTEGRATION.md — MedMarks leaderboard guide
- README.md — Project overview
- CLAUDE.md — Claude Code project guide
- CONTRIBUTING.md — Contribution guidelines
- CHANGELOG.md — Version history
- AUDIT_LOG.md — Audit tracking
- examples/nova/docs/ — NOVA-specific docs (index.md, usage.md, agentic_workflow.md)
- examples/nova/README.md, examples/gemex_thinkvg/README.md, examples/agentclinic_nejm/README.md
- Inline: docstrings, type hints

## Task

### Phase 1: Map
- List all doc files and their last-modified dates
- Check: does each example have a README? Does each core module have docstrings on public functions?

### Phase 2: Investigate
1. **API Documentation** — docstrings on AgenticProcessorBase, ToolRegistry, AdapterProtocol, all public functions? Docstrings explain "why" not just "what"?
2. **Usage Documentation** — README.md accurate for current state? Installation instructions work? API key setup clear?
3. **Example Documentation** — each example has setup, usage, expected output? NOVA docs reflect current code?
4. **Integration Guides** — verifiers_integration.md and MEDMARKS_INTEGRATION.md accurate? Code snippets match current API?
5. **Research Context** — paper/ directory: is research context preserved? Does documentation support reproducibility?

## Output Format
| Area | Status (Complete/Partial/Missing) | Specific Issue | Fix |

Stop after 12 items.
```

### Dependency Audit

```
You are auditing dependencies for a medical imaging AI framework.

## Context
- pyproject.toml — Core deps: loguru, beartype, pillow, numpy, openai, httpx, requests, aiohttp, minijinja>=2.12.0, verifiers, tenacity>=9.1.2
- Optional: [nova] torch/torchvision/datasets, [medmarks] verifiers/prime/datasets
- Dev: pytest, ruff, pyright, pre-commit, bandit, safety
- environments/nova_brain_mri/pyproject.toml — Standalone deps

## Task

### Phase 1: Map
- List all direct deps and their version constraints
- Check: are all deps actually used? Any unused?
- Compare main pyproject.toml vs environments/nova_brain_mri/pyproject.toml for version drift

### Phase 2: Investigate
1. **Version Pinning** — are versions pinned appropriately (not too loose, not too tight)?
2. **Security** — known CVEs in current versions? bandit/safety configured?
3. **Redundancy** — httpx, requests, AND aiohttp all needed? Can consolidate?
4. **Compatibility** — verifiers package compatibility? minijinja version constraint justified?
5. **Dev Deps** — are dev deps in correct group? Any dev deps leaking into runtime?

## Output Format
| Dependency | Version | Issue | Severity | Fix |

Stop after 12 items.
```

### Research Reproducibility Review

```
You are reviewing a medical imaging AI framework for research reproducibility — the ability for other researchers to reproduce published results.

## Context
This framework supports benchmarking VLMs on medical imaging tasks. Results may appear in publications. Key reproducibility surfaces:
- src/radiant_harness/config.py — all configuration with defaults
- src/radiant_harness/models/openai_adapter.py — model API calls (non-deterministic without seed)
- examples/nova/src/cli.py — CLI with all parameters
- examples/nova/src/evaluation/ — evaluation metrics
- examples/*/src/rewards/ — reward functions
- pyproject.toml — dependency versions

## Task

### Phase 1: Map
- List all sources of non-determinism (model API, random seeds, dataset shuffling, cache state)
- Check: can a researcher reproduce results from CLI args + config alone?

### Phase 2: Investigate
1. **Configuration Completeness** — are ALL parameters that affect results captured in config? Model temperature, max_tokens, reasoning_effort, tool availability, search settings?
2. **Random Seeds** — is there a seed parameter? Does it propagate to all random operations (model sampling, dataset shuffling)?
3. **Model API** — OpenAI API has a `seed` parameter for reproducibility: is it used? Temperature=0 for deterministic outputs?
4. **Evaluation Determinism** — BLEU/METEOR/CIDEr implementations deterministic? IoU computation deterministic?
5. **Version Locking** — can a researcher install exact same deps? uv.lock committed?
6. **Output Logging** — are model responses, tool calls, intermediate results logged for post-hoc analysis?

### Phase 3: Validate
Check: does running the same CLI command twice produce identical results? If not, identify the source of variation.

## Output Format

Top findings (max 10), classified as: **Reproducibility Blocker** (results not reproducible), **Reproducibility Risk** (results may vary), **Best Practice** (improve confidence). Each with: path + symbol + snippet, fix.
```

### Code Quality & Style Audit

```
You are auditing code quality for a Python research framework.

## Context
- Linting: ruff (rules: E, W, F, I, B, C4, UP, N, S, T20, SIM, ARG, PTH, ERA, PL, PERF), line-length=100
- Type checking: pyright (basic mode, Python 3.10)
- Runtime validation: beartype
- Testing: pytest, coverage fail_under=60
- Pre-commit: pre-commit hooks configured
- CI: make check runs ruff + pyright + pytest

## Task

### Phase 1: Run Checks
- Run `uv run ruff check src/ examples/ environments/` — report violations
- Run `uv run pyright src/` — report type errors
- Grep for TODO/FIXME/HACK/XXX across entire codebase
- Grep for bare `except:` or `except Exception:` (should use specific exceptions)

### Phase 2: Investigate
1. **beartype Coverage** — @beartype on all public functions in src/radiant_harness/? Missing decorators?
2. **Type Completeness** — pyright errors? `Any` types that should be specific? Missing return type annotations?
3. **Import Organisation** — circular imports? Lazy imports where not needed?
4. **Naming Conventions** — consistent naming across modules? Any "improved_", "enhanced_", "unified_" naming debt?
5. **Dead Code** — unused imports, unreachable code, commented-out blocks?

### Phase 3: Fix
For each finding, provide a one-line fix or propose the minimal change.

## Output Format
4 buckets: **Type Safety**, **Linting**, **Naming**, **Dead Code**. Max 12 items. Each with: path + symbol + snippet, fix.
```

---

## Usage Guide

1. Find the prompt you need in the table of contents
2. Copy the **Claude Code** or **Codex CLI** version (everything inside the code block)
3. Paste directly into your tool — no assembly needed, preamble is included
4. For prompts in the "template" sections (remaining core modules, examples), copy the header + body and combine
5. Follow up with specific questions about findings
6. Run tests after implementing recommended changes: `make check`

## Customisation

- Combine related prompts for broader reviews (e.g., Medical Safety + Reward Correctness)
- Add `Focus particularly on files changed in the last week` for targeted reviews
- Add `Only report Critical and Important findings` to reduce noise
- Prepend `Before starting, read CLAUDE.md for project conventions` for convention-aware reviews
- For agent teams: assign disjoint module scopes per teammate (e.g. one on tools/, one on models/, one on verifiers/)

## Version

Prompts Version: 1.0 (2026-02-12)

### Changelog

- **1.0** — Initial structured prompts tailored for Radiant Harness. Covers: full architecture, medical safety, security, performance, all core modules (base, tools, models, retrieval, verifiers, config/types/exceptions, cache, prompts, utils), all 5 examples (nova, gemex_thinkvg, agentclinic_nejm, pubmedqa, vqa_rad), MedMarks environment, and cross-cutting concerns (test coverage, documentation, dependencies, reproducibility, code quality). Dual Claude Code / Codex CLI variants for all prompts. (2026-02-12)
