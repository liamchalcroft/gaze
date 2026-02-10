# AUDIT REPORT

Date: 2026-02-10  
Repo: `nova_retrieval_vlm`

## Scope

Audit focus areas requested:
1. Core correctness and cohesion (`src/radiant_harness`)
2. Benchmark integrity/reproducibility (`examples/nova`)
3. Tool/retrieval safety and determinism
4. Adapter reliability
5. Verifiers/RL integration alignment (`environments/nova_brain_mri`)

## Core Contract Map (`radiant_harness`)

### Public Surface (stable contract candidates)

- `src/radiant_harness/base.py:269` `AgenticProcessorBase.analyze(...)`
- `src/radiant_harness/base.py:187` `get_system_prompt(images, metadata)`
- `src/radiant_harness/base.py:204` `get_user_message(images, metadata)`
- `src/radiant_harness/base.py:221` `get_response_schema()`
- `src/radiant_harness/base.py:230` `validate_response(response)`
- `src/radiant_harness/tools/registry.py:323` `ToolRegistry.get_tool_schemas()`
- `src/radiant_harness/tools/registry.py:328` `ToolRegistry.execute(tool_name, **kwargs)`
- `src/radiant_harness/models/adapter_protocol.py:28` `AdapterProtocol.generate_chat(...)`

### Key Types

| Type | Path | Contract |
|---|---|---|
| `ToolCall` | `src/radiant_harness/types.py:18` | `{id, name, arguments}` where `arguments` is `dict | str` |
| `ToolResult` | `src/radiant_harness/types.py:33` | `{tool_name, description, error?, image_base64?, image_mime_type?, metadata}` |
| `Turn` | `src/radiant_harness/types.py:71` | `role in {"user","assistant","tool_result"}` plus content/tool data |
| `AgenticResult` | `src/radiant_harness/types.py:90` | `{final_response, turns, total_tokens, confidence}` |
| `GenerationLog` | `src/radiant_harness/models/adapter_protocol.py:11` | `{prompt_tokens, completion_tokens, finish_reason}` |

### Error Types

- `AgenticProcessingError` (`src/radiant_harness/exceptions.py:54`)
- `ToolExecutionError` (`src/radiant_harness/exceptions.py:13`)
- `UnknownToolError` (`src/radiant_harness/exceptions.py:38`)
- `ModelError` / `APIError` (`src/radiant_harness/exceptions.py:92`, `:100`)

### `analyze(...)` flow + termination/JSON/tool-call paths

1. Normalize and load images, create `ToolRegistry` (`base.py:289-295`).
2. Build `messages = [system, user(+images)]` (`base.py:386-389`).
3. Turn loop (`base.py:399`), call adapter with:
   - `tools=current_tools` (disabled on last turn)
   - `response_format=response_schema` (always passed)
4. If `tool_calls` present:
   - Validate each has `id,name,arguments` (`base.py:423`)
   - Append assistant tool call message (`base.py:457-465`)
   - Execute tools and append `role="tool"` messages (`base.py:468-498`)
   - Append `Turn(role="tool_result", ...)` and `continue` to next turn (`base.py:500-506`)
5. If no tool calls:
   - `json.loads(response_text)` required (`base.py:507-515`)
   - Must be JSON object (`base.py:517-523`)
   - `continue` must be boolean (`base.py:526-532`)
   - `continue=false` => final response, break (`base.py:534-537`)
   - On last turn and `continue=true`, force `continue=false` and break (`base.py:539-544`)
6. Final contract check is only `self.validate_response(final_response)` (`base.py:554`).

## Findings

| Issue | Severity | Evidence (path:symbol + snippet) | Exploit/Failure mode | Fix | Validation/KPI |
|---|---|---|---|---|---|
| HF adapters incompatible with core schema flow | High | `src/radiant_harness/base.py:_run_analysis` `response_format=response_schema`; `src/radiant_harness/models/huggingface_adapter.py:generate_chat` `"response_format is not supported"` | Any processor returning schema fails on HF adapters before generation | Add adapter capability negotiation or one-time fallback in core when adapter rejects `response_format` | Repro run: `ModelError response_format is not supported for HuggingFace adapters` |
| Mixed tool-call + JSON content ambiguity | High | `src/radiant_harness/base.py:_run_analysis` `if typed_tool_calls ... continue` | Final JSON content is ignored whenever tool calls exist; on terminal turn this yields opaque final-validation failure | Explicitly reject mixed mode on same turn or parse/validate content before tool branch | Repro script returned `AgenticProcessingError Final response failed schema validation` |
| NOVA response validation is shallow | High | `examples/nova/src/schemas.py:validate_nova_response` `return all(field in response for field in required)` | Empty nested objects pass as “valid” if provider-side schema is bypassed | Enforce nested JSON schema validation (`jsonschema`/Pydantic) in `validate_response` | Repro: minimal `{caption:{}, diagnosis:{}, localization:{}, continue:false}` returns `True` |
| Search wrapper cache is not reused across tool calls | Medium | `src/radiant_harness/retrieval/web_search.py:775` `async with WebSearchManager(...)`; `image_search.py:643` same pattern | TTL cache mostly ineffective in real tool usage; repeated identical queries hit network | Keep a long-lived manager per processor/session or inject singleton manager into tools | Instrumentation: two identical calls created two manager instances (`manager_instances=2`) |
| Retrieval prompt injection passthrough | High | `src/radiant_harness/tools/search.py:_execute_search_web` appends raw `result.title`/`result.content` into `formatted_results` | External text can inject instructions into next model turn | Quote+sanitize untrusted retrieval text, prepend “untrusted source” envelope, strip tool-control patterns | Mock repro showed `contains_injection=True` for attacker string |
| Image download has no content-size cap or decode verification | High | `src/radiant_harness/retrieval/image_search.py:_do_download` reads full body and writes bytes directly | Memory/disk DoS via oversized response; mislabeled payload accepted if `Content-Type: image/*` | Add `max_download_bytes`, streaming read cap, and PIL verify/decode before save | Add test with mocked large response; KPI: `% downloads rejected for oversize` |
| Visual zoom can grow image unboundedly across turns | High | `src/radiant_harness/tools/visual.py:zoom_image` only per-step factor bounds, no absolute max dimensions/pixels | Repeated zoom causes runaway memory and huge base64 payloads in conversation | Add max dimensions/pixel budget in config and reject over-limit transforms | Add stress test: repeated zoom should fail fast at configured cap |
| Diagnosis benchmark metric depends on remote LLM by default | High | `examples/nova/src/evaluation/diagnosis.py:16-19` default model; `:190-195` live API semantic match | Non-deterministic scoring; vendor/model updates can shift leaderboard | Add offline deterministic mode, cache semantic decisions, pin model/version for official runs | KPI: rerun variance of diagnosis metrics across fixed predictions |
| Diagnosis metric is gameable (`top5` unbounded, `coverage` unclamped) | High | `examples/nova/src/evaluation/diagnosis.py:266-291` iterates all predictions; `:306-308` `coverage = uniq_preds/uniq_refs` | Model can emit very long diagnosis lists to inflate top-k hit chance and coverage >1 | Hard-cap predictions to 5 for top5; clamp coverage to `[0,1]`; penalize overlong candidate lists | Add adversarial eval test with 100 guesses; KPI: exploit success rate |
| Environment advertises tools/search but does not execute tools | High | `environments/nova_brain_mri/src/nova_brain_mri/__init__.py:343-397` `env_response` only updates turn/completion | “Tool-using” benchmark can be solved without tool behavior fidelity | Implement via `vf.ToolEnv` or `RadiantHarnessAdapter`; parse and execute calls, record tool traces | Repro: tool-like assistant text still yields `tool_uses=0`, no env messages |
| Example vs packaged environment diagnosis semantics drift | High | `examples/nova/src/evaluation/diagnosis.py:95-137` semantic equivalence rules; `environments/nova_brain_mri/src/nova_brain_mri/rewards.py:161-227` normalization-only exact matching | Same prediction can score differently between benchmark and leaderboard env | Share one canonical diagnosis scorer between example and environment | Repro: `acoustic neuroma` vs `vestibular schwannoma` => env `0.0`, example exact semantic `True` |
| Environment CLI permits weak schema and swallows model errors | Medium | `environments/nova_brain_mri/src/nova_brain_mri/cli.py:243-247` `json_object`; `:258-261` catch+break | Partial/malformed runs still produce rewards without strict schema contracts | Use strict `json_schema`; count/propagate failures in aggregate output | KPI: `schema_parse_failure_rate`, `episode_abort_rate` |
| Example smoke test wiring is broken (not CI-covered) | Medium | `examples/nova/tests/test_nova_dataset_smoke.py:15` inserts `examples/nova/src`; imports `src.data...`; root pytest only `tests/` (`pyproject.toml:97`) | Dataset integrity regressions can ship silently | Fix import path (`examples/nova`), include example tests in CI matrix | Repro: `uv run pytest examples/nova/tests/test_nova_dataset_smoke.py -q` fails import |
| Docs drift against actual API/implementation | Low | `examples/nova/docs/agentic_workflow.md:84` `analyze(image_path=...)`; `:225` `search_results` field not in `AgenticResult`; `environments/nova_brain_mri/README.md:154` claims Hungarian algorithm | Misleads contributors and downstream integrations | Minimal doc corrections for signature, result fields, and matching algorithm description | Repro: `analyze(image_path=...)` raises `TypeError` |

## Patch Set #1 (Implemented)

### Goal
Fix one high-ROI correctness/alignment bug in verifiers reward integration.

### Change
Schema-aligned NOVA reward extraction and localization strictness in `examples/nova/src/rewards.py`:

- Accept and prioritize schema keys:
  - caption: `description` (fallback `text`)
  - diagnosis: `primary_diagnosis` (fallbacks retained)
  - localization: `bounding_box` and `bbox`
- Accept integer coordinates safely and cast to float for IoU.
- Fix false-positive loophole in localization reward:
  - `prediction != []` with `reference == []` now scores `0.0` (previously could score `1.0`).
- Added fallback parsing of reference boxes from `info.localizations` / `gold_localizations`.

### Tests Added

- `tests/test_nova_reward_schema_alignment.py`
  - `test_nova_rewards_use_schema_keys`
  - `test_localization_reward_penalizes_false_positive_without_reference`

### Commands Run

1. `uv run pytest tests/test_nova_reward_schema_alignment.py -q`  
Pass criteria: both new tests pass.

2. `uv run pytest tests/test_verifiers_integration.py tests/test_agentic_processor.py tests/test_agentic_tool_messages.py tests/test_huggingface_adapter.py -q`  
Pass criteria: no regressions in adjacent behavior.

Observed: both commands passed.

## Reward-Hack Vectors (>=10) + Mitigations

1. Minimal top-level JSON objects pass NOVA validator.  
Mitigation: nested schema validation in `validate_response`.
2. Diagnosis top5 accepts arbitrary-length lists.  
Mitigation: enforce max 5 predictions.
3. Diagnosis coverage can exceed 1.0.  
Mitigation: clamp coverage and/or redefine denominator.
4. Keyword stuffing boosts token-set caption F1.  
Mitigation: add precision-heavy metrics and hallucination penalties.
5. Environment tool flags are non-binding (`use_tools`/`use_web_search` no execution).  
Mitigation: actual tool execution trace + tool-use compliance rubric.
6. Retrieval prompt injection text is forwarded verbatim.  
Mitigation: sanitize/quote untrusted snippets and strip instruction-like patterns.
7. Oversized image downloads can consume resources.  
Mitigation: byte caps + decode verification.
8. Repeated zoom can blow up image size exponentially.  
Mitigation: transformation budget and max pixels.
9. Diagnosis semantic matching relies on mutable external model behavior.  
Mitigation: pinned evaluator model + cached decisions.
10. Mixed tool-call/content turn ambiguity can force failure or bypass intent.  
Mitigation: enforce one output mode per assistant turn.
11. Environment CLI catches model errors and continues scoring partial episodes.  
Mitigation: explicit failure accounting and abort thresholds.
12. Search cache reset per call increases stochastic network effects.  
Mitigation: session-level cache persistence.

## Benchmark Integrity Notes (NOVA)

- Bounding boxes in `NovaGroundTruth` are converted from `(x,y,width,height)` to `(x1,y1,x2,y2)` (`examples/nova/src/data/nova_ground_truth.py:122-134`).
- `examples/nova/src/evaluation/detection.py` assumes `xyxy` and uses `torchmetrics` mAP.
- Diagnosis evaluation currently mixes heuristic exact matching and remote LLM semantic matching, reducing strict reproducibility.
- Example tests for data loading are currently outside default pytest collection and one smoke test import path is broken.

### Three concrete “cheat” routes in current metrics

1. Diagnosis: emit many guesses to increase top-k hit probability.  
Fix: strict cap + over-generation penalty.
2. Caption: emit broad medical keyword soup to maximize set-overlap F1.  
Fix: add contradiction penalty and weighted precision terms.
3. Localization: emit coarse large boxes at low IoU threshold to grab partial F1.  
Fix: include tighter IoU slices, area penalty, and FP-sensitive components.

## Environment Drift Notes (`environments/nova_brain_mri`)

- Reward semantics diverge from example benchmark diagnosis matching.
- Environment prompt claims tool support, but environment loop does not execute or return tool calls/results.
- CLI uses `json_object` rather than strict task schema.
- README claims Hungarian matching; implementation uses greedy best-IoU matching.

## Recommended KPIs

1. `strict_schema_pass_rate` (nested, not top-level only)
2. `adapter_schema_compat_rate` (OpenAI/HF parity)
3. `tool_execution_fidelity` (requested vs actually executed/recorded)
4. `retrieval_cache_hit_rate` (within benchmark session)
5. `retrieval_prompt_injection_leak_rate` (sanitizer misses)
6. `tool_payload_oversize_rate` (messages exceeding configured cap)
7. `diagnosis_eval_variance` (stddev across repeated fixed-score runs)
8. `localization_false_positive_rate_on_negative_cases`
9. `episode_abort_rate` (model/env failures)
10. `benchmark_run_determinism_rate` (identical outputs across repeated runs with fixed seed/config)

## Constraints / Not Fully Executed

- Full end-to-end benchmark run was not executed (dataset/API/GPU dependencies not guaranteed in this workspace).
- Several findings are validated via focused repro scripts and static path-level evidence rather than full benchmark jobs.
