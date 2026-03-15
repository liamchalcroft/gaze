# Development Prompts

Reusable prompts for coding agents (Claude Code, Codex, or similar). Copy-paste any prompt below into your agent. Each is designed to be run repeatedly as the codebase evolves — they scan first, then act.

Every prompt follows the same structure: SCOPE (what to look at), PLAN (scan and present findings, wait for approval), ACTION (make changes), VERIFY (run checks), COMMIT (conventional commit message).

**Evidence standard:** Every finding must include (a) file path + symbol, (b) a short quoted snippet or search string, and (c) a concrete validation step. Do not "fix" by weakening checks, deleting validation, loosening types, or silencing errors.

## Contents

### General Maintenance
- [Code Quality & Dead Code Removal](#code-quality--dead-code-removal)
- [Type Safety & Strictness](#type-safety--strictness)
- [Test Coverage & Quality](#test-coverage--quality)
- [Dependency Hygiene](#dependency-hygiene)
- [Error Handling & Resilience](#error-handling--resilience)
- [API & Interface Consistency](#api--interface-consistency)
- [Documentation & Developer Experience](#documentation--developer-experience)
- [Build & CI Health](#build--ci-health)
- [Simplification & Debt Reduction](#simplification--debt-reduction)
- [Performance & Efficiency](#performance--efficiency)

### Domain-Specific
- [Medical Safety & Clinical Accuracy](#medical-safety--clinical-accuracy)
- [Reward Correctness & Anti-Hacking](#reward-correctness--anti-hacking)
- [Research Reproducibility](#research-reproducibility)
- [Security & Credential Safety](#security--credential-safety)
- [LM Studio Local Model Baseline](#lm-studio-local-model-baseline)

### Module-Targeted
- [Agentic Loop (base.py)](#agentic-loop-basepy)
- [Tool System (tools/)](#tool-system-tools)
- [Model Adapters (models/)](#model-adapters-models)
- [Retrieval (retrieval/)](#retrieval-retrieval)
- [Example Processor Parity](#example-processor-parity)

---

## Code Quality & Dead Code Removal

*Find and remove lint violations, unused imports, dead code paths, and unreachable branches across the core package.*

```text
SCOPE: src/radiant_harness/ and tests/. Excludes examples/ (ruff extend-exclude).

PLAN: Run `uv run ruff check src/ tests/ --select E,W,F,ERA,ARG`. List every violation with file path, line number, and rule code. Separately, search for any functions, classes, or module-level variables in src/radiant_harness/ that have zero references outside their own module (check imports in __init__.py, tests/, and examples/). Present both lists as numbered findings. Wait for approval before making changes.

ACTION: For each ruff violation, apply the fix directly — remove unused imports, delete dead code, simplify unreachable branches. For unreferenced symbols not in __all__, delete them entirely. Do not add backwards-compatibility re-exports or TODO comments. Do not introduce new abstractions. Keep each fix minimal and local to the finding.

VERIFY:
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright src/
uv run pytest tests/ -x --tb=short

COMMIT: chore: remove dead code and fix lint violations in core package
```

---

## Type Safety & Strictness

*Tighten loose types across the core package: missing annotations, Any usage, unsafe casts.*

```text
SCOPE: src/radiant_harness/ only. Respect existing pyright exclusions in pyproject.toml (huggingface_adapter, openai_adapter, verifiers/).

PLAN: Run `uv run pyright src/ --outputjson` and parse the output. Search src/radiant_harness/ for Any type annotations (excluding TYPE_CHECKING blocks and existing exclusions). Search for bare cast() calls. Present findings as a numbered list with file, line, and the current loose type. Wait for approval.

ACTION: Replace Any with the narrowest concrete type that fits the usage. Add return-type annotations to un-annotated public functions. Replace cast() with runtime checks or isinstance guards where the cast is not provably safe. Do not add generic type parameters that are not used. Do not weaken existing types to fix pyright errors — instead fix the underlying type mismatch.

VERIFY:
uv run pyright src/
uv run ruff check src/
uv run pytest tests/ -x --tb=short

COMMIT: refactor: tighten type annotations in core package
```

---

## Test Coverage & Quality

*Identify untested code paths and audit existing tests for weak assertions.*

```text
SCOPE: src/radiant_harness/ (code under test) and tests/ (test files).

PLAN: Run `uv run pytest tests/ --cov=radiant_harness --cov-report=term-missing -q` and identify files below 70% coverage. Then grep tests/ for assertions that are trivially satisfiable: assert True, assert result is not None without further checks, assert isinstance(x, dict) without checking contents. Present two lists: (1) under-covered files with the specific uncovered line ranges, (2) weak assertions with file and line number. Wait for approval.

ACTION: Write tests for the top 5 under-covered files, targeting the uncovered line ranges. Each test must assert on specific values, not just that code runs. Replace weak assertions with specific value/shape/type checks. Tests must validate actual behavior and edge cases. Do not mock anything that can be instantiated directly. Do not write tests that a degenerate always-true implementation would pass.

VERIFY:
uv run pytest tests/ -x --tb=short
uv run pytest tests/ --cov=radiant_harness --cov-report=term-missing -q

COMMIT: test: improve coverage and assertion quality in core tests
```

---

## Dependency Hygiene

*Find unused, duplicate, or replaceable dependencies in pyproject.toml.*

```text
SCOPE: pyproject.toml — [project.dependencies], [project.optional-dependencies], and [dependency-groups].

PLAN: For each dependency in [project.dependencies], search src/radiant_harness/ for actual import usage. For each dev dependency, search tests/, scripts/, and config files. List any packages with zero imports, packages that duplicate stdlib functionality, and packages pinned to unnecessarily narrow ranges. Present as a numbered list. Wait for approval.

ACTION: Remove unused dependencies from pyproject.toml. Widen overly narrow version pins where the lower bound is the only meaningful constraint. Do not add new dependencies. Do not change optional dependency groups unless a package is provably unused. Run `uv lock` after changes to regenerate the lockfile.

VERIFY:
uv lock --check
uv sync
uv run pytest tests/ -x --tb=short

COMMIT: chore: remove unused dependencies and clean pyproject.toml
```

---

## Error Handling & Resilience

*Find swallowed errors, generic catch-all handlers, and silent failure modes.*

```text
SCOPE: src/radiant_harness/ — all .py files.

PLAN: Search for except Exception, except BaseException, bare except:, and any except block that contains only pass, continue, or a bare return None. Search for logger.warning or logger.error calls that swallow the exception without re-raising. Present findings as a numbered list with file, line, and the problematic pattern. Wait for approval.

ACTION: Replace broad exception handlers with the specific exception type from radiant_harness.exceptions that matches the failure mode (HarnessError hierarchy: ToolExecutionError, TemplateError, UnknownToolError, AgenticProcessingError, SchemaValidationError, ModelError, APIError). If an exception is caught and logged, re-raise it unless the function's contract explicitly requires graceful degradation (document why inline). Remove bare pass in except blocks — either handle or propagate. Do not add new exception classes unless an existing one genuinely does not fit.

VERIFY:
uv run ruff check src/
uv run pyright src/
uv run pytest tests/ -x --tb=short

COMMIT: fix: replace broad exception handlers with specific types
```

---

## API & Interface Consistency

*Review public API exports and processor interfaces for naming inconsistencies and leaky abstractions.*

```text
SCOPE: src/radiant_harness/__init__.py (public API), src/radiant_harness/base.py (processor interface), src/radiant_harness/models/adapter_protocol.py (adapter protocol).

PLAN: Compare __all__ in __init__.py against actual imports — find any symbols exported but not imported, or imported but not exported. Check that all AgenticProcessorBase abstract methods (get_system_prompt, get_user_message, get_response_schema, validate_response) have consistent parameter naming across the 5 example processors in examples/*/src/processor.py. Check AdapterProtocol implementors (OpenAIAdapter, LMStudioAdapter, HuggingFaceAdapter) for method signature drift. Present findings. Wait for approval.

ACTION: Fix __all__ to match actual imports exactly. Align parameter names in example processors to match the base class. Fix any adapter method signatures that drift from the protocol. Do not rename public API symbols that downstream code depends on without checking all call sites.

VERIFY:
uv run pyright src/
uv run pytest tests/ -x --tb=short

COMMIT: refactor: align public API exports and interface consistency
```

---

## Documentation & Developer Experience

*Remove stale comments, verify README setup instructions work, and fix dead doc references.*

```text
SCOPE: README.md, CONTRIBUTING.md, docs/, and inline comments in src/radiant_harness/.

PLAN: Run every shell command in README.md and flag any that fail. Search docs/ for references to files or symbols that no longer exist. Search src/radiant_harness/ for TODO/FIXME/HACK/XXX comments and stale docstrings that reference removed parameters or old behavior. Present findings. Wait for approval.

ACTION: Fix broken commands in README.md. Delete or update stale doc references. Remove TODO comments by either doing the work or deleting the comment if the TODO is no longer relevant. Do not add new documentation unless the code's intent is genuinely non-obvious. Do not add docstrings to private helper functions.

VERIFY:
uv run ruff check src/
uv run pytest tests/ -x --tb=short

COMMIT: docs: fix stale references and broken setup instructions
```

---

## Build & CI Health

*Audit CI pipeline, Makefile, and pre-commit config for drift and missing steps.*

```text
SCOPE: .github/workflows/ci.yml, Makefile, .pre-commit-config.yaml, pyproject.toml (tool config sections).

PLAN: Compare the CI pipeline steps against make check — identify any step present in one but missing from the other. Check pre-commit hook versions against current releases. Check if CI caches uv or .venv (it should). Check if CI runs uv lock --check. Check if the integration job actually works (the current command -v torch check is wrong). Present findings. Wait for approval.

ACTION: Align CI and Makefile so both run the same checks in the same order. Add uv caching to CI if missing. Fix the broken integration job torch detection. Update pre-commit hook revs to latest stable. Do not add new CI jobs unless a current gap is identified. Keep CI fast — no redundant steps.

VERIFY:
make check

COMMIT: ci: align CI pipeline with Makefile and fix integration job
```

---

## Simplification & Debt Reduction

*Find and delete over-engineered abstractions, dead feature paths, and unnecessary complexity.*

```text
SCOPE: src/radiant_harness/ — all modules.

PLAN: Identify: (1) classes with only one concrete subclass or implementor, (2) utility functions called from exactly one site, (3) configuration options that are never varied from their defaults in any example or test, (4) compatibility shims or adapter wrappers that bridge nothing, (5) any module that re-exports everything from another module without adding value. Present findings with evidence (call sites, config usage). Wait for approval.

ACTION: Inline single-use utilities at their call site. Remove unused config options and their associated plumbing. Delete wrapper classes that add no behavior. Merge re-export-only modules into their source. Do not add migration layers or deprecation warnings — just delete. Keep diffs minimal: one finding per change.

VERIFY:
uv run ruff check src/
uv run pyright src/
uv run pytest tests/ -x --tb=short

COMMIT: refactor: simplify core package by removing unnecessary abstractions
```

---

## Performance & Efficiency

*Identify hot paths with unnecessary allocations, redundant computation, or missing caching.*

```text
SCOPE: src/radiant_harness/base.py (agentic loop), src/radiant_harness/tools/ (tool execution), src/radiant_harness/cache.py, src/radiant_harness/tools/image_manager.py.

PLAN: Trace the per-turn hot path in base.py: message construction → tool dispatch → JSON parsing → image encoding. Identify: (1) allocations inside loops that could be hoisted, (2) repeated JSON serialization/deserialization of the same data, (3) image re-encoding on every turn when the image hasn't changed (check _downscale_image and base64 encoding), (4) tool registry lookups that could be cached. Check if TTLCache eviction runs on every access or is amortized. Check for blocking PIL operations inside async tool executors. Present findings with approximate impact. Wait for approval.

ACTION: Hoist loop-invariant allocations. Cache repeated computations. Use the existing TTLCache where appropriate rather than adding new caching. Do not add premature micro-optimizations — focus on O(n) improvements and allocation reduction. Do not change public API signatures.

VERIFY:
uv run pytest tests/ -x --tb=short
uv run pytest tests/ -x --tb=short -m performance

COMMIT: perf: optimize hot paths in agentic loop and tool execution
```

---

## Medical Safety & Clinical Accuracy

*Audit every point where clinical content is generated, evaluated, or used for training. Prioritise by patient risk.*

```text
SCOPE: src/radiant_harness/base.py (agentic loop), src/radiant_harness/tools/visual.py (image manipulation), src/radiant_harness/retrieval/ (literature search), examples/*/src/processor.py (task-specific prompts and validation), examples/*/src/evaluation/ (metrics that may appear in publications).

PLAN: Map the clinical content pipeline: image input → visual tools → model generation → response parsing → schema validation → evaluation metric. For each stage, check:
1. Visual tools — can aggressive thresholding or contrast adjustment destroy diagnostically relevant features? Are parameter bounds in ImageProcessingConfig clinically appropriate (e.g., threshold range, contrast factor range)?
2. Schema validation — do response schemas (get_response_schema()) enforce clinically necessary fields? Can a model bypass validation with well-formatted but clinically nonsensical output?
3. Prompts — do system/task prompts in examples/*/src/prompts/ avoid encouraging confabulation? Do localization prompts specify coordinate format unambiguously? Do diagnosis prompts constrain to valid diagnostic categories?
4. PubMed ranking — does RankingWeights in config.py appropriately weight evidence types (systematic reviews > case reports)? Does reliability_score reflect evidence quality?
5. Evaluation metrics — do NOVA caption metrics (BLEU/METEOR/CIDEr) match reference implementations? Does mAP in detection.py follow COCO convention? Are IoU thresholds [0.5, 0.75] standard for the modality?
Present findings classified as Patient Safety Risk, Evaluation Integrity, or Improvement. Wait for approval.

ACTION: Fix parameter bounds that could destroy diagnostic information. Add schema constraints for clinically required fields. Fix prompt language that encourages confabulation. Correct evaluation metric implementations that diverge from reference. Do not add disclaimers or boilerplate — fix the actual clinical risk.

VERIFY:
uv run pytest tests/test_clinical_safety.py -x --tb=short
uv run pytest tests/ -x --tb=short
uv run pyright src/

COMMIT: fix: address clinical safety findings in tools and validation
```

---

## Reward Correctness & Anti-Hacking

*Verify reward functions produce correct scores and cannot be gamed by a model that doesn't actually solve the task.*

```text
SCOPE: src/radiant_harness/verifiers/rewards.py (core rewards), examples/*/src/rewards.py (task-specific rewards), examples/gemex_thinkvg/src/rewards/ (multi-component rewards), environments/nova_brain_mri/src/nova_brain_mri/rewards.py (MedMarks rewards).

PLAN: For each reward function, check:
1. ExactMatchReward — case sensitivity, whitespace handling, normalisation. Can a model game it by echoing the prompt?
2. TokenF1Reward — tokenisation matches expected behavior. Can a model score high by repeating all tokens from the prompt context?
3. IoUReward — handles degenerate boxes (zero area, negative coords, coords outside [0,1]). Can a model predict a full-image box [0,0,1,1] for guaranteed non-zero IoU against any ground truth?
4. CombinedReward — do weights sum correctly? Can one component dominate (e.g., GEMeX 0.4/0.3/0.3 answer/location/bbox — can high answer reward mask zero bbox)?
5. extract_completion_text() — handles all verifiers completion formats. What happens if the format changes?
6. NOVA rewards vs MedMarks rewards — are they consistent? Any drift in scoring logic?
7. GEMeX LocationReward — is the anatomical hierarchy medically correct? Synonym list complete (e.g., "right lower lobe" ↔ "RLL")?
Present each finding with: the specific gaming vector, a concrete adversarial input that exploits it, and the expected vs actual reward score. Wait for approval.

ACTION: Add guards against degenerate inputs (e.g., reject full-image boxes in IoU, cap token repetition benefit in F1). Fix normalisation issues. Align reward implementations across examples/ and environments/. Do not change reward semantics without noting the impact on existing training runs.

VERIFY:
uv run pytest tests/test_nova_audit_fixes.py tests/test_verifiers_integration.py -x --tb=short
uv run pytest tests/ -x --tb=short

COMMIT: fix: harden reward functions against gaming vectors
```

---

## Research Reproducibility

*Identify sources of non-determinism and missing configuration that prevent result reproduction.*

```text
SCOPE: src/radiant_harness/config.py (all config), src/radiant_harness/models/openai_adapter.py (model API calls), examples/*/src/cli.py (CLI parameters), examples/*/src/evaluation/ (metrics).

PLAN: Identify every source of non-determinism:
1. Model API — is the OpenAI `seed` parameter exposed and used? Is temperature configurable and logged? Are reasoning_effort and max_tokens captured in output?
2. Dataset ordering — is dataset shuffling seeded? Does batch processing order affect results?
3. Tool execution — do visual tools produce identical output for identical inputs (PIL determinism)?
4. Search results — PubMed/Open-i results vary over time. Are search results cached or logged for reproduction?
5. Configuration completeness — can a researcher reproduce results from CLI args + config alone? Are ALL parameters that affect results captured in AgenticResult or output logs?
6. Dependency versions — is uv.lock committed? Can exact environment be recreated?
7. Evaluation determinism — BLEU/METEOR/CIDEr implementations deterministic? No random sampling in metrics?
Present findings classified as Reproducibility Blocker (results not reproducible), Risk (results may vary), or Best Practice. Wait for approval.

ACTION: Expose and propagate seed parameters. Log all configuration that affects results in AgenticResult metadata. Ensure evaluation metrics are deterministic. Do not add complexity — prefer logging config over adding new parameters.

VERIFY:
uv run pytest tests/ -x --tb=short

COMMIT: fix: improve research reproducibility with seed propagation and config logging
```

---

## Security & Credential Safety

*Scan for hardcoded secrets, injection vectors, SSRF risks, and insecure defaults.*

```text
SCOPE: All tracked files. Focus on src/radiant_harness/models/ (API keys), src/radiant_harness/retrieval/ (external API calls), src/radiant_harness/tools/ (user-controllable inputs), examples/*/run_local.sh (scripts with URLs).

PLAN: Check:
1. API key safety — keys (OPENROUTER_API_KEY, OPENAI_API_KEY, NCBI_API_KEY) never appear in logs (grep loguru calls near API usage), not in error messages, not serialised in AgenticResult, not in cache keys.
2. SSRF vectors — configurable base_url in adapters and retrieval reaches network calls. Can a user-supplied URL reach internal services? Check URL validation in LMStudioAdapter (allows HTTP by design — document why).
3. Prompt injection — tool results (PubMed abstracts, Open-i results) are inserted into conversation history. Can a crafted search result inject system-level instructions?
4. Input validation — image paths validated against path traversal? Search queries sanitised before NCBI API calls?
5. Secrets in repo — .env in .gitignore? No API keys in run_local.sh scripts or README examples?
6. HTTP in production — http:// URLs in non-test code (LMStudioAdapter allows HTTP intentionally for local inference; flag any other http:// usage).
Present findings rated Critical (exploitable now), High (exploitable with effort), Medium (defense-in-depth gap), Low (best practice). Wait for approval.

ACTION: Remove any hardcoded secrets. Add URL validation at trust boundaries. Ensure credential scrubbing in error paths. Do not add security theater (obfuscation, unnecessary encryption of non-sensitive data). Fix real injection vectors with input validation.

VERIFY:
uv run ruff check src/ --select S
uv run pytest tests/test_clinical_safety.py tests/test_web_search.py -x --tb=short

COMMIT: fix: address security findings and credential safety
```

---

## LM Studio Local Model Baseline

*Run example suites against a local LM Studio model end-to-end. Identify runability blockers, structured-output failures, and competitiveness gaps. Iterate on fixes and rerun until baselines improve.*

```text
SCOPE: src/radiant_harness/models/lmstudio_adapter.py (adapter), examples/*/src/cli.py and examples/*/run_local.sh (run surfaces), examples/*/src/processor.py (processors), src/radiant_harness/base.py (agentic loop), src/radiant_harness/tools/ (tool system), src/radiant_harness/prompts/ (templates).

PLAN: This is an empirical baseline audit — prefer real example runs over speculation.
1. Build a runability matrix for every example:
   - how to run it today (CLI command, run_local.sh)
   - whether it supports LM Studio (--base-url, LMStudioAdapter)
   - whether it supports single_turn, agentic, or both
   - what datasets/splits/local assets are required
   - known context window constraints (GLM-4.6V ~4K ctx, tool docs + base64 image can exceed this)
2. For each runnable example, execute a small evaluation (3-5 samples) using an LM Studio-hosted model:
   - use the native CLI: `uv run python -m examples.<name>.src.cli --model <model> --base-url <endpoint> --mode <mode> --max-tokens 8192 --max-image-dim 256 --max-samples 3`
   - run both single_turn and agentic modes where supported
   - record: exact command, model name, completed/failed counts, primary metrics, token usage, coercion events, tool-use behavior
3. Capture failure modes:
   - context overflow (ModelError with "context size"/"n_ctx")
   - schema validation failures (SchemaValidationError — which fields?)
   - truncated responses (single-turn salvage via extract_json_from_text)
   - confidence out of range (clamp_confidence fired?)
   - missing continue field (injection fired?)
   - thinking model content-empty fallback (reasoning_content used?)
   - tool dead-ends in agentic mode
4. Convert results into a competitiveness diagnosis:
   - core harness problems (fix once, all examples benefit)
   - example-specific prompt/schema/eval changes
   - model-specific workarounds (context limits, output format quirks)
Do not shrink scope silently — if a full run is blocked, state exactly why and identify the smallest change to unblock it. Present findings ranked as Execution Blocker, Baseline Weakness, or Improvement. Wait for approval.

ACTION: Implement the highest-leverage unblocker or baseline improvement. After each fix, rerun the narrowest proving slice to verify the improvement. Iterate: fix → rerun → measure → fix next issue. Focus on changes that improve structured-output success rate and evaluation metrics across examples, not cosmetic issues.

Common fix patterns from prior audits:
- Context overflow → reduce --max-image-dim, trim tool descriptions, or use single_turn mode
- Schema failures → check coerce_json_types() coverage, add missing coercions for the model's output patterns
- Truncation → increase --max-tokens (≥8192 for thinking models), check _try_wrap_inner_schema() for nested schemas
- Confidence out of range → verify clamp_confidence() is called in the example's validate_response()
- Missing continue → verify base.py injection fires, check if model omits or misspells the key
- Tool-use failures → check if tool schemas fit in context, verify tool result formatting

VERIFY:
uv run pytest tests/ -x --tb=short
# Then rerun the specific example that was fixed:
uv run python -m examples.<name>.src.cli --model <model> --base-url <endpoint> --mode <mode> --max-tokens 8192 --max-image-dim 256 --max-samples 3 -v

COMMIT: fix: improve local model baseline for <example> (<specific improvement>)
```

---

## Agentic Loop (base.py)

*Deep review of the central abstraction: multi-turn loop correctness, response parsing, tool dispatch, and edge cases.*

```text
SCOPE: src/radiant_harness/base.py, src/radiant_harness/types.py (Turn, ToolCall, ToolResult, AgenticResult).

PLAN: Trace the full agentic loop: analyze() → _run_analysis() → model call → tool execution → response parsing → validation. Check:
1. Loop correctness — max_turns enforced? Off-by-one? What happens when model never sets continue=false? (base.py injects continue:false when key missing — verify this works)
2. Single-turn vs multi-turn — base.py skips POLICY injection when max_turns==1. Does single-turn truncation + extract_json_from_text salvage work correctly?
3. Tool execution — _execute_tools() handles errors gracefully? Unknown tools raise UnknownToolError? Tool results properly formatted for conversation?
4. Response parsing — JSON extraction via utils/json_extract.py robust? Schema validation via validate_response() catches malformed outputs? coerce_json_types() handles type mismatches from local models?
5. Thinking model support — GLM-4.6V and Qwen 3.5 put content in reasoning_content. Does the content-empty fallback fire correctly? Does max_tokens >= 4096 enforcement work for thinking models?
6. Image handling — _downscale_image() uses PIL thumbnail + Lanczos. Does max_encode_dimension propagate correctly? Memory with large medical images?
7. Nudge messages — do nudge messages include missing field names from schema? Do they help local models recover?
8. _try_wrap_inner_schema() — detects truncated inner-object JSON and wraps under correct parent key. Test with malformed inputs.
9. Confidence — calculate_confidence() formula (base 0.5 + tool bonus up to 0.7). Can it be gamed? Does it reflect actual output quality?
Present findings as numbered list with severity. Wait for approval.

ACTION: Fix loop edge cases, parsing failures, and confidence calculation issues. Do not refactor the loop structure — fix specific bugs. Add regression tests for any fixed edge case.

VERIFY:
uv run pytest tests/ -x --tb=short
uv run pyright src/radiant_harness/base.py

COMMIT: fix: address agentic loop edge cases in base.py
```

---

## Tool System (tools/)

*Review tool infrastructure: visual tool correctness, schema generation, image state management, search integration.*

```text
SCOPE: src/radiant_harness/tools/ — tool.py, registry.py, visual.py, search.py, image_manager.py, image_ops.py.

PLAN: Check:
1. Visual tool bounds — zoom_image (0.5-4.0x), crop_image (normalised coords [0,1]), contrast factor range, threshold bounds. Do these preserve diagnostic information in medical images? Can parameters outside expected ranges cause exceptions or silent data corruption?
2. Image state — ImageManager undo/reset reliable? Memory growth with many operations? Are operations tracked correctly for AgenticResult?
3. Schema generation — do tool schemas (OpenAI function-calling JSON) match actual function signatures? Are parameter descriptions accurate for model consumption?
4. Search tools — PubMed/Open-i wrappers in search.py handle API errors, timeouts, empty results? Result formatting consistent?
5. Registry — execute_tool() handles unknown tools (UnknownToolError), disabled tools, execution errors. Tool arguments frozen via deep_freeze — does deep_thaw work correctly before execution?
6. encode_image() — base64 encoding of medical images. Memory with large images? JPEG quality settings appropriate?
Present findings. Wait for approval.

ACTION: Fix parameter validation gaps, schema mismatches, and error handling. Do not change tool semantics or parameter ranges without clinical justification.

VERIFY:
uv run pytest tests/ -x --tb=short -k "tool"
uv run pyright src/radiant_harness/tools/

COMMIT: fix: tool system correctness and schema alignment
```

---

## Model Adapters (models/)

*Review adapter parity, protocol compliance, and adapter-specific edge cases.*

```text
SCOPE: src/radiant_harness/models/ — adapter_protocol.py, openai_adapter.py, lmstudio_adapter.py, huggingface_adapter.py.

PLAN: Check:
1. Protocol compliance — do all adapters satisfy AdapterProtocol.generate_chat() with identical return types? Feature gaps (LMStudio: no response_format, no retries; HuggingFace: lazy-imported)?
2. LMStudio-specific — allows HTTP (not HTTPS), 300s timeout, context overflow detection (400 + "context size"/"n_ctx" in error body). Does supports_multipart_tool_content=False propagate correctly? Does _create_completion_with_retry actually skip retries?
3. OpenAI adapter — API key from env (not hardcoded), error handling with status codes, retry logic (tenacity config), reasoning model special handling (o1/o3), prompt caching.
4. HuggingFace adapter — lazy import via __getattr__ in __init__.py works? Model loading memory management? Vision input encoding?
5. Token tracking — GenerationLog (prompt_tokens, completion_tokens) accurate for all adapters?
6. Streaming — handled correctly across adapters? Error mid-stream?
Present findings with a parity assessment: same prompt → all adapters → same response structure. Wait for approval.

ACTION: Fix protocol compliance gaps, error handling, and token tracking. Do not change adapter interfaces without updating all implementors.

VERIFY:
uv run pytest tests/ -x --tb=short -k "adapter or model"
uv run pyright src/radiant_harness/models/adapter_protocol.py

COMMIT: fix: model adapter parity and protocol compliance
```

---

## Retrieval (retrieval/)

*Review PubMed and Open-i search: ranking quality, API integration, caching, error handling.*

```text
SCOPE: src/radiant_harness/retrieval/web_search.py (PubMedEngine, ranking), src/radiant_harness/retrieval/image_search.py (Open-i), src/radiant_harness/tools/search.py (tool wrappers).

PLAN: Trace search pipelines:
1. PubMed — user query → PubMedEngine.search() → NCBI E-utilities API → XML parsing → result construction → ranking → caching → SearchResult. Check: NCBI rate limiting respected? NCBI_API_KEY/NCBI_EMAIL handled securely (not logged)? XML parsing handles malformed responses? Timeout and retry config?
2. Ranking — RankingWeights formula: does medical_relevance appropriately weight evidence types (guidelines > RCTs > case reports)? Is recency decay reasonable for medical literature (older landmark papers still relevant)? Do content_type_boosts match clinical evidence hierarchy?
3. Open-i — image query → search_medical_images() → Open-i API → result parsing. Modality filtering works? Error handling for unavailable service?
4. Caching — TTLCache appropriate for search results (medical literature doesn't change hourly)? Cache invalidation? Memory with many queries?
5. Result quality — SearchResult fields (reliability_score, medical_relevance, extracted_entities) populated correctly? Content preview truncation preserves clinical meaning?
Present findings. Wait for approval.

ACTION: Fix ranking formula issues, API integration bugs, and caching gaps. Do not change SearchResult schema without updating all consumers.

VERIFY:
uv run pytest tests/test_web_search.py -x --tb=short
uv run pytest tests/ -x --tb=short

COMMIT: fix: retrieval search quality and API integration
```

---

## Example Processor Parity

*Verify all 5 example processors correctly implement the base class contract and handle their domain-specific concerns.*

```text
SCOPE: examples/*/src/processor.py, examples/*/src/schemas.py, examples/*/src/cli.py. Cross-reference against src/radiant_harness/base.py abstract methods.

PLAN: For each of the 5 examples (nova, gemex_thinkvg, agentclinic_nejm, pubmedqa, vqa_rad):
1. Abstract method contract — implements get_system_prompt(), get_user_message(), get_response_schema(), validate_response() with correct signatures?
2. Schema-validator alignment — does validate_response() validate exactly the fields in get_response_schema()? Does it use coerce_json_types() for local model compatibility? Does it use clamp_confidence() for confidence fields?
3. CLI consistency — all CLIs accept --base-url, --mode, --max-tokens? LM Studio preflight (require_lmstudio_model) present? --max-image-dim for vision examples?
4. Domain-specific correctness:
   - NOVA: 3 tasks (caption, diagnosis, localization) — schema matches evaluation metric expectations?
   - GEMeX: bbox validation accepts float and coerces to int? ROI normalises alternative keys?
   - AgentClinic: _requested_information() keyword list complete for clinical info gathering? Binary reward too sparse?
   - PubmedQA: image-free operation — tools disabled? Base class doesn't require images?
   - VQA-RAD: handles both open-ended and closed-ended questions? ROI normalisation for alternative keys?
5. LM Studio compatibility — all processors work with LMStudioAdapter? max_tokens >= 4096 for thinking models?
Present findings as a comparison table. Wait for approval.

ACTION: Fix contract violations, schema-validator misalignment, and missing CLI options. Align all processors to use the same patterns for common concerns (confidence clamping, type coercion, LM Studio preflight).

VERIFY:
uv run pytest tests/ -x --tb=short
uv run pytest examples/nova/tests/ -x --tb=short

COMMIT: fix: align example processors with base class contract
```

---

## Usage

1. Find the prompt you need in the table of contents
2. Copy everything inside the code block
3. Paste directly into your coding agent — no assembly needed
4. Follow up with specific questions about findings
5. Run `make check` after implementing recommended changes

**Composing prompts:** For broader reviews, combine related prompts (e.g., Medical Safety + Reward Correctness for a pre-publication audit). Add `Focus particularly on files changed since [commit]` for targeted reviews.

**Agent teams:** Assign disjoint module scopes per teammate (e.g., one on tools/, one on models/, one on retrieval/). Use the module-targeted prompts for this.
