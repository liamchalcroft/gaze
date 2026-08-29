# Changelog

All notable changes to GAZE are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org).

## [0.2.0] - 2026-08-29

### Removed

Breaking. All of these are source-incompatible for anyone pinning 0.1.x.

- ``compute_iou`` no longer reorders transposed corners by default. A box with
  ``x2 <= x1`` or ``y2 <= y1`` scores 0.0; pass ``lenient=True`` for the old
  behaviour. **NOVA localization, detection, and mAP numbers move**, together
  with the confidence-ordered matching and GT-less mAP changes below: results
  computed under 0.1.x must be regenerated before reuse.
- ``ToolRegistry.history``, the ``max_history`` constructor argument, and the
  execution history they backed.
- ``EncodedImage._data_url`` is no longer a dataclass field (the data URL is
  computed on demand by ``to_data_url()``).
- ``aiohttp`` is no longer a dependency. Code catching ``aiohttp`` exceptions
  around GAZE retrieval must catch ``httpx`` equivalents instead.
- ``HuggingFaceAdapter`` and ``HuggingFaceVLMAdapter`` are no longer in
  ``gaze.__all__``, so ``from gaze import *`` no longer binds them. Importing
  them by name is unchanged.
- Three ``WINDOW_PRESETS`` aliases (``mri_brain``, ``mri_flair``, ``mri_t2``)
  that duplicated ``brain``, ``flair``, and ``t2`` exactly.
- ``NovaGroundTruth.get_ground_truth_by_subject_id`` (unused, and misaligned
  predictions against ground truth).

### Fixed
- ``ImageInput`` leaked a file handle whenever ``img.load()`` failed after the
  header check passed (a truncated file), and ``_downscale_image`` never
  released the full-resolution source it superseded.
- ``analyze(pil_image, max_encode_dimension=N)`` encoded at full resolution and
  then discarded it. ``from_pil`` now defers encoding to ``load()``, so a
  2000x2000 input encodes 65k pixels instead of 4M.
- The idle-tool guard required zero tool calls for the entire run, so a single
  early call disabled it and a stalled model burned every remaining turn. It
  now counts consecutive tool-free turns, and force-accepts only a response
  that actually validates.
- An unrecognized ``continue`` value (``1.0``, ``"maybe"``) ended the whole run
  by exception on turn 1, while every neighbouring malformed-output case is
  nudge-recoverable. Numeric values coerce like ints; anything else is treated
  as ``False``, exactly as a missing flag is.
- ``adaptive_equalize`` gave the last tile row/column the entire remainder (16
  rows instead of 12 on a 100px image at ``tile_size=8``) while interpolating
  as though every tile were uniform, leaving a seam along the bottom and right
  edges. Tiles now spread the remainder evenly and interpolate against their
  real centres; the worst boundary artefact drops from 21x the median local
  curvature to 3x.
- Grid cell labels ran off the end of the alphabet past column 25
  (``chr(65 + col)`` yields ``[``, ``\``, ``]``); columns now continue AA, AB.
- The NOVA environment sent ``file://`` image URLs to a remote, OpenAI-compatible
  model, which has no filesystem access, so the caption and localization tasks
  scored against an image the model never saw. Images are now inlined as base64
  ``data:`` URLs.
- NOVA detection matching consumed ground-truth boxes in the order the model
  listed its findings rather than by confidence, so recall and precision moved
  with output ordering. A searched example scored 2 TP / 0 FP one way and
  1 TP / 1 FP / 1 FN the other. Predictions are now matched in confidence order.
- Images with no ground truth contributed 1.0 to mAP when the model correctly
  predicted nothing, inflating the mean exactly where labels are missing. Such
  images are now excluded, as COCO does.
- ``compute_iou`` silently reordered transposed corners, awarding full credit to
  a malformed box. Inverted boxes now score 0.0; ``lenient=True`` restores the
  old behaviour. ``compute_iou`` also accepts int coordinates now.
- ``IoUReward`` returned 0.0 for a *correct* pixel-space box whenever
  ``image_area`` was absent. The area penalty is skipped when the area is
  unknown (and still applied whenever it is known, including via
  ``image_width``/``image_height``).
- Search retries covered neither connection errors nor 5xx, while the OpenAI
  client is built with ``max_retries=0``: one 503 or DNS blip aborted a whole
  multi-turn run. Retries now cover those and honour ``Retry-After``.
- Non-retryable HTTP statuses (4xx other than 429) were retried three times with
  back-off before failing; they now fail immediately.
- Malformed PubMed and Open-i payloads raised ``AttributeError``/``TypeError``
  out of tool execution, where neither the retry wrapper nor the tool boundary
  catches them. Non-object records and non-string fields are now skipped.
- ``asyncio.gather`` propagated the first fetch error without cancelling its
  sibling, leaving an orphaned request against a possibly-closed session.
- ``measure_distance`` scaled normalized points by width rather than width-1,
  reporting 362 px across a 256 px image and disagreeing with the line
  ``compute_intensity_profile`` samples.
- ``compute_intensity_profile`` truncated sample positions, biasing every
  off-grid sample toward the start point by up to a full pixel.
- ``crop`` reported the requested area percentage, not the (truncated) area it
  actually produced.
- The dead ``get_ground_truth_by_subject_id`` looked ground truth up by CSV
  insertion order while predictions are indexed by parquet row order; its
  docstring claimed the two matched. Removed.
- ``CITATION.cff`` still declared 0.1.0; the release workflow now checks it.
- ``OPENAI_BASE_URL`` bypassed the adapter's base-URL allowlist entirely.
  ``AsyncOpenAI`` reads that variable itself, so the key was sent to whatever
  host it named (in plaintext over HTTP) without ever reaching
  ``_validate_base_url`` or the ``GAZE_ALLOW_CUSTOM_BASE_URL`` opt-in. The
  variable is now resolved and validated before the client is built.
- ``GazeAdapter`` paired tool results with tool calls by position in two
  independently flattened lists, so a single turn whose call and result counts
  differed mislabelled the ``tool_call_id`` of every later result. Pairing now
  happens per turn, and the assistant message declares its ``tool_calls`` so
  the ``role: "tool"`` messages are no longer orphaned.
- ``ExactMatchReward`` raised ``AttributeError`` on a list-valued gold answer,
  and ``normalize=True`` never collapsed whitespace unless ``strip_braces`` was
  also set (the documented behaviour).
- ``BaseMultiTurnEnv`` defaulted its debug-log directory to a path inside the
  installed package (``site-packages/gaze/logs``); it now defaults to
  ``./logs``.
- The agentic loop reset its nudge budget on any parsed JSON object, so a model
  returning well-formed but schema-invalid output was pinned at "nudge 1 of 2"
  forever: the force-finalize escalation never fired and every turn was spent on
  the same soft nudge. The budget now refills only on a response that passes
  ``validate_response``.
- ``analyze()`` closed a caller-supplied ``PIL.Image``. Ownership is now tracked
  on ``ImageInput``, so only pixels GAZE decoded are closed.
- Reusing a processor across an image run and a text-only run advertised the
  image run's 23 visual tools on the text-only call, which the registry then
  rejected as unknown. The schema and prompt-doc caches are keyed on whether
  images were supplied.
- High-bit-depth medical images (``I``, ``I;16``, ``F``, i.e. DICOM-converted PNGs)
  were clipped at 255 rather than windowed, so a 12-bit slice reached the model
  as solid white above 255 and measured as if its maximum were 255. A new
  ``to_uint8_grayscale`` windows by actual range; used by ``encode_image`` and
  every intensity tool.
- ``morphological_op(..., iterations=n)`` was a silent no-op for ``open`` and
  ``close``: repeating an idempotent composite does nothing, while the model was
  told "x3". Iterations now scale the structuring element.
- NOVA ``compute_diagnosis_reward`` gave full credit when both prediction and
  reference were empty, so a sample with a missing diagnosis label paid out 0.34
  of the weighted ``task="all"`` reward for saying nothing. The standalone
  environment already guarded this; the example now matches.
- `environments/nova_brain_mri` was uninstallable: it pinned `pillow>=10,<12`
  while depending on `gaze-vlm`, which requires `pillow>=12.2`. Dropped the
  unused `pillow` and `numpy` pins, moved `datasets` to `>=4.8.5`, and relaxed
  `requires-python` to `>=3.10` to match the framework.
- NOVA diagnosis evaluation: `normalize_diagnosis_string()` did not strip
  punctuation, so a trailing period made an otherwise identical diagnosis fail
  exact match (`"Glioblastoma."` vs `"glioblastoma"`). Punctuation other than
  hyphens is now removed. This makes exact-match slightly more permissive, so
  previously reported NOVA diagnosis numbers should be regenerated.

### Changed
- The four response-recovery paths in the agentic loop (per turn, both salvage
  paths, and the final check) are one ``_recover_response``. They had grown
  four different step orders, so a response accepted on one path could be
  rejected on another purely by which branch produced it. Inner-schema
  wrapping now also applies per turn, so a model returning a bare inner object
  recovers on turn 1 instead of after the loop ends.
- Grid labels and the ``cell_labels`` metadata come from one function rather
  than the same expression written twice.
- The normalize-then-range-check pair behind box and point arguments is one
  helper instead of five copies.
- Removed ``_get_current_image``, a single-call-site duplicate of
  ``_require_image``.
- ``examples/aiih2026_paper/`` moved out of the repository, along with the
  three carve-outs that existed only for it (gitignore entry, ruff exclusion,
  and the CI/Makefile ``--ignore``). Three tests that could only ever skip in a
  clean checkout are gone with it.
- NCBI requests now pass through one shared rate limiter per engine at the
  documented limits (3/s, or 10/s with an API key). Sleeping between calls did
  not bound concurrent callers: esummary and efetch fired simultaneously.
- Replaced ``aiohttp`` with ``httpx``, which the OpenAI client already pulls in.
  This removes ``aiohttp``, ``yarl``, and ``multidict`` from the runtime
  dependency closure. Retrieval tests now use ``httpx.MockTransport``, which
  exercises the real client path instead of hand-built async context managers.
- Tool results are downscaled before encoding (``max_tool_encode_dimension``,
  default 1024). Two zooms produced a ~1 MB base64 payload re-sent every turn;
  it is now ~17 KB. The manager keeps full resolution for measurements.
- ``EncodedImage`` builds its data URL lazily; it was eagerly keeping a second
  ~500 KB copy of every tool result that nothing read.
- Removed ``ToolRegistry`` execution history, which retained up to 100 encoded
  images for no in-repo consumer.
- ``morphological_op`` iterations now scale the structuring element.
- Removed three ``WINDOW_PRESETS`` aliases that duplicated existing entries
  exactly and reached the model three times per request.
- NOVA statistical helpers no longer label descriptive relative-change bands as
  "significant", and the results key that held no test is named
  ``descriptive_comparisons``.
- ``docs/verifiers_integration.md`` documented three ``BaseMultiTurnEnv`` hooks
  that do not exist (``build_initial_state``, ``is_completed``, and a
  two-tuple ``env_response``), and omitted ``IoUReward``'s ``continuous`` and
  ``area_penalty_start`` parameters along with its fail-closed behaviour on
  pixel coordinates.
- `TTLCache` no longer calls `close()` on cached values. Both real caches store
  plain lists; the close-handling was exercised only by mocks.
- `OpenAIAdapter.aclose()` awaits `AsyncOpenAI.close()` directly instead of
  duck-typing `close`/`aclose` and inspecting awaitability.
- The five near-identical recovery-nudge blocks in the agentic loop are now one
  `_nudge()` helper; per-failure guidance text is unchanged.
- `_maybe_normalize_box` and `_maybe_normalize_point` are one
  `_maybe_normalize_coords` handling both 2- and 4-element coordinate lists.
- Removed the `F722`/`F821` ruff suppressions, which were justified by jaxtyping
  annotations the project does not use and were masking undefined names.

## [0.1.1] - 2026-06-03

### Changed
- Relaxed `requires-python` from `>=3.10,<3.13` to `>=3.10`, so `pip install
  gaze-vlm` works on Python 3.13 and 3.14. The CI matrix now covers 3.10 through
  3.14 on Linux and 3.14 on macOS.

### Security
- Bumped pinned development and CI dependencies (cryptography, pyjwt, pyarrow,
  requests, starlette, and others) to clear advisories flagged by `pip-audit`.
  These are build and test dependencies, not runtime dependencies of the
  published package.

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
