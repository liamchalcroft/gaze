# Radiant Harness Audit Prompts

Unified audit prompts for both Claude Code and Codex.

Each section below contains one self-contained prompt. The prompts are written for open-ended audits rather than one-off bug hunts, so they emphasize current-repo inspection, evidence-backed findings, durable risks, and narrow validation steps.

## How To Use

- Pick the narrowest prompt that matches the review you want.
- Paste the entire code block into your coding agent.
- If you want analysis only, say so. If you want code changes too, ask for fixes after the audit or instruct the agent to implement the first bounded patch it recommends.
- Treat the live repository as the source of truth. These prompts name the current structure, but if files move or new modules appear, the agent should follow the code it finds.

## Current Repo Map

- `src/radiant_harness/`: core package with `base.py`, `config.py`, `types.py`, `exceptions.py`, `cache.py`, `models/`, `tools/`, `retrieval/`, `prompts/`, `verifiers/`, and `utils/`
- `examples/nova/`: full benchmark package with processor, config, prompts, schemas, rewards, data loading, evaluation, visualization, experiments, docs, and tests
- `examples/gemex_thinkvg/`: processor, dataset, schemas, reward stack, verifiers environment, and train/eval scripts
- `examples/agentclinic_nejm/`: multi-turn clinical environment plus train/eval and dataset download tooling
- `examples/pubmedqa/`: text-only biomedical QA processor, CLI, dataset, evaluation, schema, and reward
- `examples/vqa_rad/`: radiology VQA processor, CLI, dataset, evaluation, schema, and reward
- `environments/nova_brain_mri/`: standalone MedMarks/verifiers environment with its own utilities, packaging, and tests
- `tests/`: broad regression suite covering prompts, adapters, tools, retrieval, security, clinical safety, performance, verifiers, and example alignment
- `docs/`, `paper/`, `README.md`, `CLAUDE.md`, `PROMPTS.md`, `Makefile`, `pyproject.toml`, `scripts/`

## Audit Index

### Repository-Wide

1. Full Repository Architecture & Risk Audit
2. Clinical Safety & Evaluation Integrity Audit
3. Security, Privacy & External I/O Audit
4. Performance, Concurrency & Memory Audit
5. Public API, Config & Type Contracts Audit

### Core Package

6. `src/radiant_harness/base.py` Agentic Loop Audit
7. `src/radiant_harness/tools/` Tool System & Image State Audit
8. `src/radiant_harness/models/` Adapter Parity Audit
9. `src/radiant_harness/retrieval/` Search & Evidence Audit
10. `src/radiant_harness/prompts/` Runtime Prompt Template Audit
11. `src/radiant_harness/verifiers/` RL Integration Audit

### Examples & Environment

12. `examples/nova/` Benchmark Audit
13. `examples/gemex_thinkvg/` Reward and Environment Audit
14. `examples/agentclinic_nejm/` Clinical Environment Audit
15. `examples/pubmedqa/` Text-Only QA Audit
16. `examples/vqa_rad/` VQA Audit
17. `environments/nova_brain_mri/` Standalone Environment Audit

### Cross-Cutting

18. Test Suite & Regression Harness Audit
19. Documentation, Prompt Catalog & Repo Instruction Audit
20. Dependency, Packaging & Developer Workflow Audit
21. Reproducibility, Benchmarking & Paper Artifact Audit
22. LM Studio End-to-End Example Baseline Audit

## Repository-Wide

### 1. Full Repository Architecture & Risk Audit

```text
You are auditing the Radiant Harness repository, a Python framework for agentic vision-language analysis in medical imaging.

Operating rules:
- Inspect the live repo before judging it. Use the paths below as anchors, not proof.
- Treat this as an open-ended architecture audit, not a hunt for one known bug. Prioritize durable risks: abstraction leaks, contract drift, duplicated logic, unsafe defaults, evaluation gaps, and missing regression protection.
- Every substantive finding must include: severity, path + symbol, a short quoted snippet or exact search string, why it matters, and the smallest concrete validation step.
- Prefer falsifiable checks over opinions. When unsure, propose the narrowest experiment or test that would decide the issue.
- Do not suggest weakening schemas, hiding exceptions, loosening tests, or deleting safeguards to make problems disappear.
- Ask questions only if truly blocked. If fixes were explicitly requested, propose a minimal patch sequence after the audit and implement only the first bounded change with narrow checks.

Audit scope:
- Core package: `src/radiant_harness/`
- Example packages: `examples/nova/`, `examples/gemex_thinkvg/`, `examples/agentclinic_nejm/`, `examples/pubmedqa/`, `examples/vqa_rad/`
- Standalone environment: `environments/nova_brain_mri/`
- Quality surfaces: `tests/`, `docs/`, `README.md`, `CLAUDE.md`, `PROMPTS.md`, `pyproject.toml`, `Makefile`, `scripts/`

Tasks:
1. Build a current repo map and dependency sketch from the live code.
2. Trace the main dataflow:
   image and metadata input -> processor -> tool registry and image manager -> model adapter -> JSON extraction and validation -> agentic result -> optional verifiers environment and rewards.
3. Identify the highest-risk boundaries across:
   - coupling and abstraction leaks
   - duplicated logic or parity gaps between core, examples, environment, and docs
   - response-schema lifecycle and prompt alignment
   - failure handling and exception translation
   - clinical/evaluation integrity hotspots
   - test coverage on the riskiest paths
4. Explicitly call out places where documentation or prompt assumptions have drifted enough to create engineering risk.
5. For the top issues, name the narrowest validating test, repro, benchmark, or parity harness.

Output format:
1. Repo map
2. Top findings table, ranked `Critical`, `Important`, or `Suggestion` (max 15 items)
3. Dependency and dataflow sketch
4. Recommended patch sequence (only if fixes were requested)
5. Open questions or assumptions
```

### 2. Clinical Safety & Evaluation Integrity Audit

```text
You are auditing Radiant Harness for clinical safety, benchmark integrity, and reward alignment.

Operating rules:
- Start from the live repo, not from assumptions in docs.
- Treat this as a reusable safety and evaluation audit. Look for durable risks that would still matter after today's obvious bugs are fixed.
- Every finding must include: category, path + symbol, a short quoted snippet or exact search string, impact, and a concrete validation step.
- Prefer parity checks when the same clinical logic appears in multiple places.
- Do not accept weaker checks, broader normalization, or prompt-only fixes when the underlying metric, schema, or reward contract is wrong.
- Ask questions only if blocked. If the user asked for fixes, stop after the findings unless they also asked for implementation.

Audit scope:
- Core processor and prompt loop: `src/radiant_harness/base.py`, `src/radiant_harness/prompts/`
- Visual and search tools: `src/radiant_harness/tools/`, `src/radiant_harness/retrieval/`
- Reward and verifier logic: `src/radiant_harness/verifiers/`
- Example evaluation and rewards: `examples/nova/src/evaluation/`, `examples/nova/src/rewards.py`, `examples/gemex_thinkvg/src/rewards/`, `examples/pubmedqa/src/evaluation.py`, `examples/vqa_rad/src/evaluation.py`
- Example schemas and processors: `examples/nova/src/schemas.py`, `examples/gemex_thinkvg/src/schemas.py`, `examples/pubmedqa/src/schemas.py`, `examples/vqa_rad/src/schemas.py`
- Standalone benchmark environment: `environments/nova_brain_mri/`
- Relevant tests: `tests/test_clinical_safety.py`, `tests/test_clinical_audit_fixes.py`, `tests/test_evaluation_integrity.py`, `tests/test_evaluation_metrics.py`, example/env tests

Tasks:
1. Map every place where clinical content is generated, transformed, evaluated, or used as an RL reward.
2. Audit reward correctness and reward-hacking resistance:
   - `src/radiant_harness/verifiers/rewards.py`
   - `examples/nova/src/rewards.py`
   - `examples/gemex_thinkvg/src/rewards/`
   - `environments/nova_brain_mri/src/nova_brain_mri/rewards.py`
3. Audit evaluation integrity:
   - metric formulas
   - IoU thresholds and box conventions
   - normalization behavior
   - parity between reward functions and reported metrics
4. Audit schema and prompt alignment:
   - required fields
   - confidence bounds
   - `continue` semantics
   - malformed JSON handling
   - free-text bypasses
5. Audit tool safety and evidence-grounding quality:
   - destructive image transforms
   - coordinate-space changes
   - PubMed/Open-i result formatting and attribution
   - anti-confabulation instructions in runtime prompts
6. When logic is duplicated across core, examples, and the environment, require a parity harness or a concrete diff check.

Output format:
1. Findings ranked as `Patient Safety Risk`, `Evaluation Integrity`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Short list of the highest-priority parity checks to add if coverage is missing
```

### 3. Security, Privacy & External I/O Audit

```text
You are auditing Radiant Harness for security, privacy, prompt-injection resistance, and external I/O safety.

Operating rules:
- Inspect the live codepaths for network access, file access, temp files, secrets, and untrusted content handling.
- Focus on durable attack surfaces: SSRF, credential leakage, prompt injection via tool results, unsafe local URLs, path traversal, error scrubbing failures, and overly permissive config.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, exploit or failure mode, and the smallest concrete repro or validation step.
- Do not treat a warning comment or docstring as a mitigation unless the code enforces it.
- Do not recommend hiding or truncating evidence in a way that would weaken safety reviews.
- Ask questions only if blocked. If fixes are requested, propose the smallest defensive patch set first.

Audit scope:
- Processor/tool-result injection boundaries: `src/radiant_harness/base.py`
- Config and URL validation: `src/radiant_harness/config.py`
- Model adapters and error translation: `src/radiant_harness/models/openai_adapter.py`, `src/radiant_harness/models/lmstudio_adapter.py`, `src/radiant_harness/models/huggingface_adapter.py`
- Retrieval stack: `src/radiant_harness/retrieval/base.py`, `src/radiant_harness/retrieval/web_search.py`, `src/radiant_harness/retrieval/image_search.py`
- Tool state and image handling: `src/radiant_harness/tools/registry.py`, `src/radiant_harness/tools/image_manager.py`, `src/radiant_harness/tools/search.py`
- CLIs/docs that expose secrets or unsafe commands: `README.md`, `docs/`, example CLIs
- Relevant tests: `tests/test_security.py`, `tests/test_image_download_ssrf.py`, `tests/test_credential_scrubbing.py`, `tests/test_honest_user_agent.py`

Tasks:
1. Map all external trust boundaries:
   - model API calls
   - PubMed/Open-i requests
   - LM Studio local server usage
   - image downloads
   - temp directory creation and cleanup
   - user-provided paths or metadata
2. Inspect the code for:
   - SSRF and DNS-rebinding defenses
   - unsafe HTTP allowances
   - prompt injection from tool results or external abstracts
   - credential leakage in errors, logs, or docs
   - path traversal or unsafe file resolution
   - malformed-JSON or exception paths that accidentally expose sensitive content
3. Check whether docs, examples, and CLIs encourage insecure defaults or omit required warnings.
4. Prioritize real exploitability over theoretical lint-style concerns.

Output format:
1. Findings ranked `Critical`, `High`, or `Medium`
2. For each finding: exploit path or failure mode, impact, and smallest validation step
3. Residual risks that are not yet exploitable but deserve hardening
```

### 4. Performance, Concurrency & Memory Audit

```text
You are auditing Radiant Harness for performance, async correctness, scalability, and memory behavior.

Operating rules:
- Inspect the live hot paths before proposing optimizations.
- Focus on enduring performance risks: event-loop blocking, redundant encoding/copies, repeated network work, excessive prompt construction, cache misuse, avoidable image transforms, and memory retention.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, why it is costly, and the narrowest benchmark, profile, or test that would validate it.
- Prefer changes that preserve correctness and observability; do not trade away validation or safety for speed without explicit justification.
- Ask questions only if blocked. If implementation was requested, keep changes small and benchmark the narrow path they affect.

Audit scope:
- Processor loop and image loading: `src/radiant_harness/base.py`
- Tool/image state: `src/radiant_harness/tools/image_manager.py`, `src/radiant_harness/tools/registry.py`, `src/radiant_harness/tools/visual.py`
- Model adapters: `src/radiant_harness/models/`
- Retrieval stack: `src/radiant_harness/retrieval/`
- Prompt rendering: `src/radiant_harness/prompts/`
- NOVA experiments and analysis helpers: `examples/nova/experiments/`, `examples/nova/src/utils/`, `examples/nova/src/cli.py`
- Relevant tests: `tests/test_performance_fixes.py`, `tests/test_perf_ps1_batch_and_vectorize.py`, `tests/test_perf_audit_ps1_ps2.py`, `tests/test_ps1_perf_fixes.py`, `tests/test_ps2_cow_image_manager.py`

Tasks:
1. Trace the highest-cost paths for:
   - loading/encoding images
   - repeated tool transforms
   - network requests and retries
   - prompt/template rendering
   - result aggregation and evaluation
2. Check async correctness:
   - blocking work on the event loop
   - unnecessary sequential awaits
   - hidden synchronization bottlenecks
   - cleanup paths that leak temp files or in-memory images
3. Check data movement and caching:
   - repeated base64 encoding
   - large string concatenations
   - prompt/tool result copies
   - cache invalidation and eviction behavior
4. Tie each recommendation to a concrete measurement plan or existing benchmark gap.

Output format:
1. Findings ranked `Blocking`, `Material`, or `Improvement`
2. For each finding: cost explanation and smallest benchmark or validation step
3. Suggested benchmark additions if the current suite does not protect the hotspot
```

### 5. Public API, Config & Type Contracts Audit

```text
You are auditing the public API and core contracts of Radiant Harness.

Operating rules:
- Start from exported symbols and work inward.
- Focus on durable contract risks: breaking exports, inconsistent lazy imports, mutable data leaking through frozen types, config isolation bugs, exception taxonomy gaps, and docs that promise APIs the code does not actually support.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, contract impact, and the smallest validation step.
- Treat tests and docs as part of the contract when they shape user expectations.
- Do not recommend weakening invariants, freezing less state, or broadening exception handling just to make the API easier to use.
- Ask questions only if blocked.

Audit scope:
- Public exports: `src/radiant_harness/__init__.py`, `src/radiant_harness/models/__init__.py`, `src/radiant_harness/verifiers/__init__.py`
- Core contracts: `src/radiant_harness/config.py`, `src/radiant_harness/types.py`, `src/radiant_harness/exceptions.py`, `src/radiant_harness/cache.py`, `src/radiant_harness/utils/`
- CLI surface: `src/radiant_harness/__main__.py`
- Example export surface: `examples/nova/src/__init__.py`
- Relevant tests: `tests/test_immutability.py`, `tests/test_config_isolation.py`, `tests/test_cache.py`, `tests/test_verifiers_lazy_imports.py`

Tasks:
1. Inventory the public symbols users are expected to import and the contracts they imply.
2. Audit:
   - lazy import behavior and optional dependency boundaries
   - config dataclass validation and thread-local/global isolation
   - immutability guarantees for result types and metadata
   - exception hierarchy completeness and specificity
   - cache correctness and eviction behavior
   - small utility contracts such as IoU and JSON extraction
3. Compare exported APIs against README/docs/examples for drift.
4. Flag any contract that is surprising, under-tested, or internally inconsistent.

Output format:
1. Findings ranked `Breaking Contract`, `Weak Contract`, or `Improvement`
2. For each finding: contract affected, evidence, and smallest validation step
3. Missing regression tests to lock down the public API
```

## Core Package

### 6. `src/radiant_harness/base.py` Agentic Loop Audit

```text
You are auditing `src/radiant_harness/base.py`, which owns the core agentic processing loop.

Operating rules:
- Read the live implementation and its tests before judging the design.
- Focus on durable correctness risks: turn-loop termination, tool-call parsing, coordinate-space drift, response parsing, image lifecycle, adapter initialization, and error translation.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Prefer tracing one real end-to-end path through `analyze()` over isolated opinions about individual helpers.
- Do not recommend papering over bad outputs by making validation weaker.
- Ask questions only if blocked.

Audit scope:
- Primary file: `src/radiant_harness/base.py`
- Supporting contracts: `src/radiant_harness/types.py`, `src/radiant_harness/config.py`, `src/radiant_harness/tools/`, `src/radiant_harness/models/`
- Relevant tests: `tests/test_base_processor.py`, `tests/test_agentic_processor.py`, `tests/test_agentic_tool_messages.py`, `tests/test_coord_space_tracking.py`

Tasks:
1. Trace the full control flow for:
   - image normalization and loading
   - model adapter initialization
   - prompt assembly and turn creation
   - tool-call extraction and execution
   - tool-result reinjection and sanitization
   - forced finalization and idle-turn handling
   - final JSON extraction, validation, and confidence calculation
2. Audit high-risk behaviors:
   - malformed tool calls
   - model outputs that ignore the schema
   - coordinate-changing tool usage before localization outputs
   - image cleanup and memory retention
   - exception wrapping that loses too much context
3. Check whether the tests cover the real failure modes or only happy paths.

Output format:
1. Findings ranked `Correctness Bug`, `Safety or Robustness Gap`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. The single most important additional regression test if coverage is missing
```

### 7. `src/radiant_harness/tools/` Tool System & Image State Audit

```text
You are auditing the Radiant Harness tool system and image state handling.

Operating rules:
- Inspect the live registry, image manager, and tool implementations together.
- Focus on durable risks: state corruption, coordinate confusion, unsafe defaults, schema drift, and tool documentation that no longer matches runtime behavior.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat coordinate-space semantics as a first-class contract, not as an implementation detail.
- Do not solve correctness problems by making the tool interface vaguer.
- Ask questions only if blocked.

Audit scope:
- `src/radiant_harness/tools/tool.py`
- `src/radiant_harness/tools/registry.py`
- `src/radiant_harness/tools/image_manager.py`
- `src/radiant_harness/tools/visual.py`
- `src/radiant_harness/tools/search.py`
- Supporting interaction in `src/radiant_harness/base.py`
- Relevant tests: `tests/test_tool_registry.py`, `tests/test_visual_tools.py`, `tests/test_tool_system_audit.py`, `tests/test_tool_review_fixes.py`, `tests/test_image_manager.py`

Tasks:
1. Map the tool lifecycle:
   registration -> documentation -> parameter schema -> execution -> state mutation -> formatted tool result.
2. Audit the image state model:
   - original vs current image
   - reset behavior
   - encoded image reuse
   - coordinate normalization
   - pixel vs normalized argument interpretation
3. Audit visual tool correctness and safety:
   - destructive threshold/window settings
   - flip/rotate/crop/zoom coordinate consequences
   - intensity/measurement utilities
   - edge cases for small images and bounds
4. Audit search tool wrappers for error handling, formatting, and disabled-tool behavior.

Output format:
1. Findings ranked `Correctness`, `Safety`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Any parity harness needed between tool docs, schemas, and actual runtime behavior
```

### 8. `src/radiant_harness/models/` Adapter Parity Audit

```text
You are auditing the model adapter layer for contract consistency and parity.

Operating rules:
- Compare adapters against the shared protocol and against each other.
- Focus on durable risks: mismatched message contracts, inconsistent tool support, structured-output drift, lazy import bugs, error translation gaps, and unsafe base URL handling.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- When behavior is duplicated or intentionally divergent, demand an explicit parity check or rationale.
- Do not accept adapter-specific prompt hacks as a replacement for a broken contract unless the prompt is the explicit contract.
- Ask questions only if blocked.

Audit scope:
- `src/radiant_harness/models/adapter_protocol.py`
- `src/radiant_harness/models/openai_adapter.py`
- `src/radiant_harness/models/huggingface_adapter.py`
- `src/radiant_harness/models/lmstudio_adapter.py`
- `src/radiant_harness/models/__init__.py`
- Relevant tests: `tests/test_openai_adapter.py`, `tests/test_huggingface_adapter.py`, `tests/test_lmstudio_adapter.py`

Tasks:
1. Inventory the adapter contract exposed by `AdapterProtocol` and `GenerationLog`.
2. Compare adapter behavior for:
   - chat generation
   - streaming
   - tool calling
   - response-format and JSON handling
   - reasoning support
   - vision payload handling
   - base URL and credential validation
   - error mapping into repository exceptions
3. Check lazy import boundaries and optional dependencies.
4. Identify places where examples or docs assume parity that the adapters do not really provide.

Output format:
1. Findings ranked `Contract Break`, `Parity Gap`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Highest-value adapter parity tests to add if gaps exist
```

### 9. `src/radiant_harness/retrieval/` Search & Evidence Audit

```text
You are auditing the retrieval layer that powers PubMed and Open-i evidence gathering.

Operating rules:
- Inspect live query construction, ranking, sanitization, download, and formatting code.
- Focus on durable risks: unsafe network behavior, incorrect evidence scoring, attribution drift, brittle parsing, cache mistakes, and poor failure handling.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat evidence formatting as part of the correctness contract because it directly affects model behavior.
- Do not paper over retrieval bugs with prompt instructions alone.
- Ask questions only if blocked.

Audit scope:
- `src/radiant_harness/retrieval/base.py`
- `src/radiant_harness/retrieval/web_search.py`
- `src/radiant_harness/retrieval/image_search.py`
- `src/radiant_harness/tools/search.py`
- Relevant tests: `tests/test_web_search.py`, `tests/test_image_search.py`, `tests/test_search_tools.py`, `tests/test_search_engine_base.py`, `tests/test_honest_user_agent.py`

Tasks:
1. Trace how search queries are issued, retried, ranked, cached, and formatted for LLM consumption.
2. Audit:
   - URL validation and host restrictions
   - timeout and retry behavior
   - user-agent and API compliance
   - snippet and content truncation
   - evidence scoring and reliability weighting
   - image download validation and temp directory cleanup
   - exception scrubbing and logging hygiene
3. Check whether retrieval results preserve enough provenance for downstream reasoning and audits.

Output format:
1. Findings ranked `Safety or Correctness Bug`, `Evidence Quality Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Missing parity or integration tests between retrieval code and search tools
```

### 10. `src/radiant_harness/prompts/` Runtime Prompt Template Audit

```text
You are auditing the runtime prompt template system used by the processors.

Operating rules:
- Inspect the live template loader, template files, and the processors that feed them.
- Focus on durable prompt-system risks: template/context drift, silent omissions, agentic vs single-turn inconsistency, schema mismatch, and prompt instructions that undermine safe tool use or output validation.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat prompt-template contracts as code contracts: missing context, incorrect guard usage, and stale fields all count as bugs if they change behavior.
- Do not reduce strictness just to make templates easier to render.
- Ask questions only if blocked.

Audit scope:
- `src/radiant_harness/prompts/__init__.py`
- `src/radiant_harness/prompts/agentic/`
- `src/radiant_harness/prompts/single_turn/`
- `examples/nova/src/prompts/`
- Prompt construction in processors such as `examples/nova/src/processor.py`
- Relevant tests: `tests/test_prompts.py`

Tasks:
1. Map the prompt-loading contract:
   template lookup -> strict rendering -> prompt combination -> processor usage.
2. Audit:
   - strict undefined handling
   - `is defined` guard coverage
   - context keys required by each template
   - agentic vs single-turn instruction consistency
   - tool documentation and search guidance injection
   - schema/prompt alignment, especially around `continue` and required fields
3. Check whether runtime prompts still match the live tool surface and evaluation expectations.

Output format:
1. Findings ranked `Contract Mismatch`, `Prompt Quality Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. The most useful prompt-rendering regression tests to add if gaps exist
```

### 11. `src/radiant_harness/verifiers/` RL Integration Audit

```text
You are auditing the RL and verifiers integration layer in Radiant Harness.

Operating rules:
- Read the live verifier code, its lazy import boundaries, and its tests before judging it.
- Focus on durable training risks: reward extraction bugs, reward hacking surfaces, environment conversion drift, data-path resolution mistakes, and mismatches between RL rewards and reported metrics.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Demand parity checks when reward logic is duplicated across environments or examples.
- Do not accept broader normalization or looser parsing as a substitute for a correct reward contract.
- Ask questions only if blocked.

Audit scope:
- `src/radiant_harness/verifiers/base.py`
- `src/radiant_harness/verifiers/adapter.py`
- `src/radiant_harness/verifiers/mixin.py`
- `src/radiant_harness/verifiers/rewards.py`
- `src/radiant_harness/verifiers/__init__.py`
- Relevant tests: `tests/test_verifiers_integration.py`, `tests/test_verifiers_lazy_imports.py`

Tasks:
1. Inventory the verifier-facing contracts:
   - environment base class
   - processor mixin
   - adapter bridge
   - reward interfaces
2. Audit:
   - lazy import behavior and optional dependencies
   - message conversion and completion extraction
   - image-path resolution and safety
   - reward composition and weighting
   - parity between RL reward logic and example evaluation code
3. Flag reward-hackable behavior or environment contracts that are under-specified.

Output format:
1. Findings ranked `Training Blocker`, `Reward Alignment Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Highest-value parity tests to add across core and example reward paths
```

## Examples & Environment

### 12. `examples/nova/` Benchmark Audit

```text
You are auditing the NOVA example, which is the repository's largest end-to-end benchmark package.

Operating rules:
- Inspect the live example package, not just the README.
- Treat this as a reusable benchmark audit. Focus on durable risks: metric integrity, dataset mapping, prompt/schema/reward drift, CLI workflow correctness, local-model support, experiment reproducibility, and visualization or analysis helpers that can mislead users.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Compare example logic against core-library contracts and against the standalone environment when behavior overlaps.
- Ask questions only if blocked.

Audit scope:
- Processor/config/schema/reward core: `examples/nova/src/processor.py`, `config.py`, `schemas.py`, `rewards.py`, `types.py`
- Data and evaluation: `examples/nova/src/data/`, `examples/nova/src/evaluation/`
- Prompts: `examples/nova/src/prompts/`
- Visualization and analysis: `examples/nova/src/visualization/`, `examples/nova/src/utils/`
- CLI and experiments: `examples/nova/src/cli.py`, `examples/nova/experiments/`
- Docs/tests: `examples/nova/docs/`, `examples/nova/README.md`, `examples/nova/tests/`

Tasks:
1. Map the end-to-end NOVA flow:
   dataset -> processor -> prompt/schema -> model/tool loop -> output files -> evaluation -> analysis/plots.
2. Audit:
   - task-specific prompt/schema/reward alignment
   - evaluation formulas and thresholds
   - dataset parsing and bbox conversion
   - CLI behavior and output serialization
   - LM Studio/OpenAI adapter usage paths
   - experiment aggregation and analysis helpers
   - doc drift against the live implementation
3. Compare shared logic with `environments/nova_brain_mri/` and demand parity evidence where behavior should match.

Output format:
1. Findings ranked `Benchmark Blocker`, `Evaluation Integrity Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. The highest-value parity or smoke tests to add
```

### 13. `examples/gemex_thinkvg/` Reward and Environment Audit

```text
You are auditing the GEMeX-ThinkVG example for reward integrity, environment behavior, and training/evaluation parity.

Operating rules:
- Inspect the live reward stack, processor, dataset loader, and verifiers environment together.
- Focus on durable risks: schema drift, bbox math errors, location hierarchy mistakes, environment/processor mismatch, and reward weighting that no longer reflects the intended task.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Prefer parity checks whenever evaluation and reward logic touch the same concept.
- Ask questions only if blocked.

Audit scope:
- `examples/gemex_thinkvg/src/processor.py`
- `examples/gemex_thinkvg/src/dataset.py`
- `examples/gemex_thinkvg/src/schemas.py`
- `examples/gemex_thinkvg/src/rewards/answer.py`
- `examples/gemex_thinkvg/src/rewards/location.py`
- `examples/gemex_thinkvg/src/rewards/bbox.py`
- `examples/gemex_thinkvg/src/rewards/combined.py`
- `examples/gemex_thinkvg/src/verifiers/environment.py`
- `examples/gemex_thinkvg/train.py`, `examples/gemex_thinkvg/eval.py`, `examples/gemex_thinkvg/README.md`
- Relevant tests under `tests/` that target GEMeX behavior

Tasks:
1. Map the end-to-end GEMeX flow from dataset case to final reward.
2. Audit:
   - answer normalization
   - anatomical hierarchy logic
   - bbox validation and IoU/GIoU behavior
   - combined reward weighting and scaling
   - ThinkVG response parsing
   - environment-tool interaction loop
   - train/eval assumptions and doc drift
3. Identify any places where the environment and processor encode different task contracts.

Output format:
1. Findings ranked `Training or Eval Blocker`, `Reward or Design Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Missing parity harnesses between reward components and environment behavior
```

### 14. `examples/agentclinic_nejm/` Clinical Environment Audit

```text
You are auditing the AgentClinic NEJM example for multi-turn environment correctness and clinical-case evaluation integrity.

Operating rules:
- Read the live environment, train/eval scripts, and download tooling before judging the package.
- Focus on durable risks: information-request loop bugs, completion criteria mistakes, normalization drift, image or data-path handling errors, and dataset/download instructions that no longer reproduce the expected setup.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat prompt and completion format as part of the environment contract.
- Ask questions only if blocked.

Audit scope:
- `examples/agentclinic_nejm/src/environment.py`
- `examples/agentclinic_nejm/train.py`
- `examples/agentclinic_nejm/eval.py`
- `examples/agentclinic_nejm/data/download.py`
- `examples/agentclinic_nejm/README.md`

Tasks:
1. Map the episode lifecycle:
   case loading -> prompt construction -> information request loop -> completion detection -> reward calculation.
2. Audit:
   - request verbs and completion criteria
   - answer normalization and brace handling
   - image or auxiliary evidence handling
   - logging/debug output
   - dataset download assumptions and reproducibility
   - README accuracy
3. Identify failure modes that would silently inflate accuracy or prematurely terminate episodes.

Output format:
1. Findings ranked `Environment Blocker`, `Evaluation Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. The most important regression tests or smoke checks to add
```

### 15. `examples/pubmedqa/` Text-Only QA Audit

```text
You are auditing the PubMedQA example, which exercises the harness in a text-only biomedical QA setting.

Operating rules:
- Inspect the live processor, schema, evaluation, and CLI code rather than assuming it behaves like the image-based examples.
- Focus on durable risks: answer-extraction fallbacks, reward/evaluation mismatch, search grounding issues, schema drift, and CLI behavior that hides malformed outputs.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat yes/no/maybe normalization as a contract, not just a convenience.
- Ask questions only if blocked.

Audit scope:
- `examples/pubmedqa/src/processor.py`
- `examples/pubmedqa/src/schemas.py`
- `examples/pubmedqa/src/evaluation.py`
- `examples/pubmedqa/src/dataset.py`
- `examples/pubmedqa/src/cli.py`

Tasks:
1. Map the full flow from dataset item to final answer and reward.
2. Audit:
   - fallback extraction when JSON is missing
   - answer normalization and schema validation
   - evaluation vs reward parity
   - web-search usage and grounding expectations
   - text-only processor assumptions inside the generic agentic framework
   - CLI/reporting behavior
3. Identify failure modes that would make the benchmark accept malformed or weakly grounded answers.

Output format:
1. Findings ranked `Correctness Bug`, `Evidence Alignment Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. The highest-value tests to add if answer extraction or grounding is under-protected
```

### 16. `examples/vqa_rad/` VQA Audit

```text
You are auditing the VQA-RAD example for question handling, prompt/schema alignment, and reward/evaluation parity.

Operating rules:
- Inspect the live processor, evaluation, schema, and CLI code together.
- Focus on durable risks: closed-vs-open question branching bugs, prompt/tool mismatch, answer normalization drift, and output handling that diverges between evaluation and RL reward.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat question metadata as part of the task contract.
- Ask questions only if blocked.

Audit scope:
- `examples/vqa_rad/src/processor.py`
- `examples/vqa_rad/src/schemas.py`
- `examples/vqa_rad/src/evaluation.py`
- `examples/vqa_rad/src/dataset.py`
- `examples/vqa_rad/src/cli.py`

Tasks:
1. Map the flow from dataset sample to processor response, evaluation, and reward.
2. Audit:
   - closed/open question handling
   - prompt expectations for tool use and search
   - schema validation
   - normalization parity between evaluation and reward
   - CLI output/reporting
3. Identify cases where incorrect metadata or malformed JSON would be scored too generously.

Output format:
1. Findings ranked `Correctness Bug`, `Metric Parity Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Missing tests that would lock down closed/open branching behavior
```

### 17. `environments/nova_brain_mri/` Standalone Environment Audit

```text
You are auditing the standalone NOVA Brain MRI environment used for MedMarks/verifiers integration.

Operating rules:
- Inspect the live environment package, not just its README.
- Focus on durable risks: independence from `radiant_harness`, utility parity drift, reward correctness, schema drift, CLI mismatch, and packaging assumptions that break external use.
- Every finding must include: severity, path + symbol, a short quoted snippet or exact search string, impact, and the smallest validation step.
- When local utility copies duplicate core-library behavior, demand an explicit parity check or the smallest parity harness.
- Ask questions only if blocked.

Audit scope:
- `environments/nova_brain_mri/src/nova_brain_mri/__init__.py`
- `environments/nova_brain_mri/src/nova_brain_mri/rewards.py`
- `environments/nova_brain_mri/src/nova_brain_mri/_utils.py`
- `environments/nova_brain_mri/src/nova_brain_mri/cli.py`
- `environments/nova_brain_mri/pyproject.toml`
- `environments/nova_brain_mri/README.md`
- `environments/nova_brain_mri/tests/test_rewards.py`

Tasks:
1. Map the standalone environment contract:
   dataset -> prompt/schema -> reward -> CLI/API usage.
2. Audit:
   - reward fidelity and thresholds
   - duplicated utility parity with `radiant_harness`
   - independence from `radiant_harness` imports where that independence is required
   - CLI flags and schema output
   - packaging/install behavior
   - README accuracy
3. Identify any drift between this environment and `examples/nova/` that would confuse benchmark comparisons.

Output format:
1. Findings ranked `Integration Blocker`, `Parity or Integrity Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Highest-value parity tests between the standalone environment and the core/example implementation
```

## Cross-Cutting

### 18. Test Suite & Regression Harness Audit

```text
You are auditing the repository's tests and regression harnesses.

Operating rules:
- Start from the highest-risk behaviors in the live code and then inspect whether the tests really protect them.
- Focus on durable testing risks: missing guardrails, brittle assertions, overfit regression tests that encode a buggy implementation, weak parity coverage, and documentation/examples that are never exercised.
- Every finding must include: severity, path + symbol or test name, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat missing tests for high-risk contracts as findings even if the underlying code currently looks correct.
- Ask questions only if blocked.

Audit scope:
- Root tests in `tests/`
- Example tests in `examples/nova/tests/`
- Environment tests in `environments/nova_brain_mri/tests/`
- Coverage and quality config in `pyproject.toml`, `pytest.ini`, `coverage.json` if needed

Tasks:
1. Map the highest-risk contracts in the repository:
   adapters, tools, retrieval, prompt rendering, clinical safety, reward parity, config isolation, performance, and docs-to-code drift.
2. Check whether tests cover:
   - core happy paths
   - failure paths
   - security boundaries
   - parity between duplicated implementations
   - benchmark and evaluation invariants
3. Flag brittle or misleading tests:
   - assertions that lock in incidental behavior
   - tests that silently rely on generated artefacts
   - missing smoke tests for important CLIs or docs examples
4. Recommend only the most leverage-heavy additions.

Output format:
1. Findings ranked `Missing Guardrail`, `Weak Guardrail`, or `Hygiene`
2. For each finding: evidence, impact, and smallest validation step
3. Top 5 tests or parity harnesses worth adding next
```

### 19. Documentation, Prompt Catalog & Repo Instruction Audit

```text
You are auditing the repository's human-facing instructions and audit prompts for drift, correctness, and long-term usefulness.

Operating rules:
- Inspect the live code and compare it directly to the docs and prompts.
- Focus on durable drift: outdated repo maps, wrong commands, incorrect capability claims, prompt catalogs that overfit to a past issue, and instructions that are too tool-specific to stay reusable.
- Every finding must include: severity, path + section, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Treat `PROMPTS.md` as a product artifact: prompts should be reusable, detailed enough to produce high-quality audits, and not anchored to one transient bug.
- Ask questions only if blocked.

Audit scope:
- `README.md`
- `CLAUDE.md`
- `PROMPTS.md`
- `docs/`
- Example READMEs under `examples/`
- `environments/nova_brain_mri/README.md`
- `paper/PAPER_GUIDE.md`
- `paper/RESEARCH_PROMPT.md`

Tasks:
1. Compare docs and prompts against the live repo structure and package surface.
2. Audit:
   - wrong commands or dependency instructions
   - outdated example descriptions
   - missing mention of important modules or workflows
   - prompt quality, reusability, and model-agnostic behavior
   - duplicated or contradictory instructions across docs
3. Flag generated or archival files only if they materially confuse the documented workflow.

Output format:
1. Findings ranked `Incorrect`, `Misleading`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Highest-priority doc or prompt updates if the user wants edits next
```

### 20. Dependency, Packaging & Developer Workflow Audit

```text
You are auditing the repository's packaging, dependency strategy, and developer workflow.

Operating rules:
- Read the live packaging and tooling files before judging the workflow.
- Focus on durable risks: broken install paths, optional dependency confusion, stale coverage/type-check exclusions, environment-specific packaging drift, and commands that are documented but unsupported.
- Every finding must include: severity, path + section, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Prefer concrete build or install failure modes over style commentary.
- Ask questions only if blocked.

Audit scope:
- Root packaging/tooling: `pyproject.toml`, `uv.lock`, `Makefile`, `Dockerfile`, `pytest.ini`, `.pre-commit-config.yaml`
- Scripts: `scripts/`
- Environment packaging: `environments/nova_brain_mri/pyproject.toml`
- Docs that define the workflow: `README.md`, `CONTRIBUTING.md`, `docs/`

Tasks:
1. Map the supported install, lint, type-check, and test flows.
2. Audit:
   - dependency bounds and optional groups
   - lazy-import assumptions vs declared dependencies
   - coverage and pyright exclusion strategy
   - packaging boundaries between root project and standalone environment
   - Makefile/scripts accuracy
   - developer ergonomics that create avoidable setup failures
3. Prioritize findings that block reproducible local development or release hygiene.

Output format:
1. Findings ranked `Build or Release Risk`, `Workflow Friction`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Minimal workflow hardening steps if the user wants fixes
```

### 21. Reproducibility, Benchmarking & Paper Artifact Audit

```text
You are auditing the repository's reproducibility story, benchmark outputs, and research artifacts.

Operating rules:
- Compare code, experiment helpers, docs, and paper artifacts against each other.
- Focus on durable research risks: result aggregation mistakes, seed/control gaps, undocumented benchmark assumptions, figure-generation drift, and paper or research prompts that no longer match the implementation.
- Every finding must include: severity, path + symbol or section, a short quoted snippet or exact search string, impact, and the smallest validation step.
- Ignore generated LaTeX build files unless they materially affect source-control hygiene or reproducibility.
- Ask questions only if blocked.

Audit scope:
- Experiment helpers: `examples/nova/experiments/`
- Evaluation and statistical utilities: `examples/nova/src/evaluation/`, `examples/nova/src/utils/`
- Analysis/plotting: `examples/nova/src/visualization/`
- Research artifacts: `paper/main.tex`, `paper/references.bib`, `paper/PAPER_GUIDE.md`, `paper/RESEARCH_PROMPT.md`, `paper/research_results.md`
- Supporting docs: `examples/nova/docs/`, `docs/`

Tasks:
1. Map how benchmark results are produced, aggregated, analyzed, and described.
2. Audit:
   - experiment configuration coverage
   - statistical-analysis assumptions
   - plotting/aggregation consistency with evaluation outputs
   - claim traceability from code to paper/docs
   - research prompts that are stale, over-specified, or disconnected from the current repo
3. Prioritize issues that would cause irreproducible figures, misleading comparisons, or paper claims that the code cannot substantiate.

Output format:
1. Findings ranked `Reproducibility Blocker`, `Benchmark Integrity Risk`, or `Improvement`
2. For each finding: evidence, impact, and smallest validation step
3. Highest-value parity or reproducibility checks to add next
```

### 22. LM Studio End-to-End Example Baseline Audit

```text
You are auditing Radiant Harness as a practical benchmarking baseline by running, or preparing to run, the example suites against a local model served through LM Studio.

Operating rules:
- This is an empirical baseline audit, not a purely static review. Prefer real example runs over speculation whenever the environment and datasets make them possible.
- Start by inspecting the live repo to determine what is actually runnable today for each example, what supports LM Studio already, and what only supports OpenAI-style remote usage.
- Focus on durable competitiveness gaps: runability, structured-output reliability, single-turn vs agentic behavior, tool-use quality, evaluation integrity, CLI ergonomics, and differences between core-harness issues and example-specific issues.
- Every substantive finding must include: severity, path + symbol or command surface, a short quoted snippet or exact search string, what happened or is likely to happen, and the smallest concrete validation step or rerun plan.
- Record exact commands, model identifiers, LM Studio base URL assumptions, dataset/split scope, failure counts, and metric outputs. If a full run is blocked, say exactly why and identify the smallest code or setup change needed to unblock it.
- Do not shrink scope silently. If a true full run is infeasible, state the blocker and use the largest defensible slice only as a temporary diagnostic, not as a substitute for the intended benchmark.
- Ask questions only if blocked by missing local assets or hardware facts.

Audit scope:
- LM Studio integration: `src/radiant_harness/models/lmstudio_adapter.py`, `src/radiant_harness/models/__init__.py`
- Example run surfaces:
  - `examples/nova/src/cli.py`, `examples/nova/run_local.sh`, `examples/nova/run_medgemma.sh`, `examples/nova/src/processor.py`
  - `examples/pubmedqa/src/cli.py`, `examples/pubmedqa/src/processor.py`
  - `examples/vqa_rad/src/cli.py`, `examples/vqa_rad/src/processor.py`
  - `examples/gemex_thinkvg/` train/eval scripts, processor, and verifiers environment
  - `examples/agentclinic_nejm/` train/eval scripts and environment
- Core harness surfaces that commonly affect local baselines:
  - `src/radiant_harness/base.py`
  - `src/radiant_harness/prompts/`
  - `src/radiant_harness/tools/`
  - `src/radiant_harness/verifiers/`
- Evaluation outputs and benchmark helpers under each example

Tasks:
1. Build a runability matrix for every example:
   - how to run it today
   - whether it already supports LM Studio or needs adapter plumbing
   - whether it supports `single_turn`, `agentic`, or only one of them
   - what datasets, splits, and local assets are required
2. For each example, attempt a full evaluation plan using an LM Studio-hosted local model such as Qwen 3.5 A3B:
   - use the native example CLI or environment when possible
   - run both `single_turn` and `agentic` modes where the example supports both
   - if an example lacks one of those modes, treat that as a design finding to evaluate rather than an assumption to ignore
3. Capture execution results:
   - exact command/config
   - model name and LM Studio endpoint assumptions
   - number of completed vs failed samples
   - primary benchmark metrics
   - token, latency, and tool-use behavior when available
   - structured-output failures, retries, parse failures, tool dead-ends, and mode-specific regressions
4. Convert the results into a competitiveness diagnosis:
   - core-harness problems to fix once for all examples
   - example-specific prompt/schema/eval/tooling changes
   - workflow/documentation changes needed so outside users can reproduce the baseline easily
5. Prioritize the smallest set of changes that would materially improve the repo as a public benchmark baseline for local and open-weight models.

Output format:
1. Runability matrix by example and mode
2. Execution summary with commands, metrics, failures, and blockers
3. Findings ranked `Execution Blocker`, `Baseline Weakness`, or `Improvement`
4. Split improvement plan:
   - core harness changes
   - example-specific changes
   - docs/workflow changes
5. If code edits were requested, implement only the first high-leverage unblocker and rerun the narrowest proving slice
```

## Notes

- Ignore generated artefacts such as `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`, `runs/`, and LaTeX build byproducts unless the task is specifically about checked-in artefact hygiene.
- Treat `paper-old/` as archival unless the user explicitly asks for historical comparison.
- When code is duplicated across core, examples, and the standalone environment, a parity harness is usually better than relying on visual inspection alone.

## Version

Refreshed on 2026-03-06.

Key changes in this edition:
- Unified each audit target into one prompt that works for both Claude Code and Codex.
- Updated the repo map to match the current structure, including `lmstudio_adapter`, expanded `examples/nova/`, standalone `examples/pubmedqa` and `examples/vqa_rad` processors/CLIs, `paper/`, and the broader regression suite.
- Added cross-cutting prompts for prompt-catalog drift, packaging/workflow, and research artifact reproducibility.
- Added an empirical LM Studio baseline prompt aimed at running end-to-end example evaluations on local models and converting the results into competitiveness improvements.
