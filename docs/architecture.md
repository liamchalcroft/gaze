# Architecture

## Overview

GAZE is organized around a multi-turn agentic loop where a VLM reasons over images using tools and structured JSON output.

```
                       +------------------------+
                       |  AgenticProcessorBase  |
                       |     (your subclass)    |
                       +-----------+------------+
                                   |
                       +-----------v------------+
                       |      Agentic loop      |
                       |   (multi-turn, JSON)   |
                       +-----------+------------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
   +--------v---------+   +--------v---------+   +--------v---------+
   |  Model adapter   |   |  Tool registry   |   |     Prompts      |
   | (OpenAI, LM      |   |     (visual,     |   |    (Jinja2)      |
   |  Studio, HF)     |   |      search)     |   |                  |
   +------------------+   +--------+---------+   +------------------+
                                   |
                       +-----------+-----------+
                       |                       |
              +--------v---------+   +---------v--------+
              |  Image manager   |   |    Retrieval     |
              |    (PIL ops)     |   |    (PubMed,      |
              |                  |   |     Open-i)      |
              +------------------+   +------------------+
```

## Core components

### AgenticProcessorBase (`base.py`)

The abstract base class that drives the agentic loop. Subclasses implement:

- `get_system_prompt()` -- system message for the model
- `get_user_message()` -- user message with task context
- `get_response_schema()` -- JSON schema for structured output
- `validate_response()` -- validates the parsed response

The loop runs until the model returns `"continue": false` or `max_turns` is reached.

### Types (`types.py`)

All data types are frozen (immutable):

- **ToolCall** -- a tool invocation with name and arguments
- **ToolResult** -- tool execution output
- **Turn** -- one round of model response + tool calls
- **AgenticResult** -- the complete multi-turn result

### Configuration (`config.py`)

Frozen dataclasses for all configuration:

- **GazeConfig** -- top-level config
- **SearchConfig** -- PubMed/Open-i search parameters
- **ImageProcessingConfig** -- image manipulation settings

Use `config_context()` for task-scoped overrides via ContextVar.

### Model adapters (`models/`)

All adapters implement `AdapterProtocol`:

- **OpenAIAdapter** -- OpenAI API and OpenRouter
- **LMStudioAdapter** -- local models via LM Studio (subclasses OpenAIAdapter, no retries, HTTP allowed)
- **HuggingFaceAdapter** -- direct Transformers inference (lazy-imported to avoid torch dependency)

### Tool system (`tools/`)

- **ToolRegistry** -- manages tool lifecycle and execution
- **Visual tools** (23) -- image manipulation via PIL
- **Search tools** (2) -- PubMed and Open-i literature search

### Retrieval (`retrieval/`)

- **PubMed search** -- NCBI E-utilities API
- **Open-i image search** -- medical image retrieval

### Verifiers integration (`verifiers/`)

For RL training with verifiable rewards:

- **BaseMultiTurnEnv** -- multi-turn RL environment
- **Reward functions** -- ExactMatchReward, TokenF1Reward, IoUReward, CombinedReward
- **GazeAdapter** -- bridges processors and verifiers

## Design principles

- **Frozen data** -- all types and configs are immutable. Use `deep_freeze()`/`deep_thaw()` for nested structures.
- **beartype validation** -- runtime type checking on all public APIs.
- **Specific exceptions** -- `GazeError` hierarchy, never bare `except:`.
- **Async-first** -- all I/O operations are async.
- **Config isolation** -- `config_context()` uses ContextVar for thread/task-safe overrides.

## How the agentic loop works

The loop lives in `AgenticProcessorBase._run_analysis()` (`base.py`). The
description below tracks that implementation; the constants named in
parentheses are module-level defaults in `base.py`.

### Turn lifecycle

Each `analyze()` call normalizes and loads the input images, optionally
downscales them (`max_encode_dimension`), builds a tool registry (multi-turn
only), then iterates up to `max_turns` times. On every turn GAZE:

1. Sends the running message list to the adapter via `generate_chat()`.
2. Parses the response into either tool calls or a JSON object.
3. Executes any tool calls and appends their results, or accepts the JSON as
   the final answer when `continue` is false.

The default turn budget is 10 (`_DEFAULT_MAX_TURNS`) and the hard ceiling is
30 (`_MAX_TURNS_LIMIT`); larger values are clamped with a warning.

### The `continue` flag

In multi-turn mode the system prompt instructs the model to return a boolean
`continue` field every turn: `true` to request more tools or analysis,
`false` to finalize. Local models often omit the field when they mean
"false", so GAZE injects `continue: false` whenever the key is
absent. It also coerces non-boolean values: `null` becomes `false`, `0`/`1`
become booleans, and the strings `"true"`/`"false"`/`"yes"`/`"no"` are
normalized. A value that cannot be coerced raises `AgenticProcessingError`.

### Single-turn vs multi-turn

When `max_turns == 1` the behaviour differs in several ways:

- No `POLICY` block is appended to the system prompt (the multi-turn turn
  budget and `continue` instructions are skipped).
- No tool registry is created and no tools are offered, because the single
  turn is by definition the final turn and the final turn always withholds
  tools.
- The response schema is injected into the system prompt as a JSON skeleton
  with field descriptions, so models that ignore `response_format` (notably
  local models) still see the expected output shape.

In multi-turn mode the schema skeleton is instead used for recovery nudges
and the force-finalize message.

### Tool-result handling

Tool calls and `response_format` are not requested together: many providers
return empty content when both are present, so `response_format` is only
enforced on turns where tools are withheld (including the final turn). Tool
results are sanitized before being fed back to the model: control characters
are stripped, content is truncated to 8000 characters (`_MAX_TOOL_CONTENT_CHARS`),
and each result is wrapped in a randomized untrusted-content boundary to limit
prompt injection from external data such as PubMed abstracts.

Image-mutating tools (`requires_image=True`) run sequentially to preserve the
shared `ImageManager` state, while independent tools (such as search) run
concurrently alongside them. Adapters that cannot accept image payloads in
tool messages (`supports_multipart_tool_content == False`, e.g.
`LMStudioAdapter`) receive a text description plus a note that the image
could not be displayed.

GAZE tracks whether coordinate-modifying or intensity-modifying tools
have run (see the `_COORD_MODIFYING_TOOLS` and `_INTENSITY_MODIFYING_TOOLS`
sets, documented in [Tools](tools.md)). On the final turn it re-attaches the
original image and warns the model that bounding boxes from transformed views
are invalid and that intensity measurements from modified images do not
reflect original tissue. A successful `reset` clears both flags.

### Error recovery

The loop is built to keep going when a model misbehaves rather than crashing:

- **Nudges.** Empty responses, non-JSON text, non-object JSON, truncated
  output, and responses that fail `validate_response()` each trigger a
  corrective user message that restates the required structure. After
  `_MAX_CONSECUTIVE_NUDGES` (2) consecutive failures GAZE escalates to
  a stricter force-finalize message. If nudges still do not recover after
  `_MAX_RECOVERY_NUDGES` total attempts, it raises `AgenticProcessingError`.
  A turn with valid JSON or successful tool calls resets the nudge counter.
- **Idle-tool force-finalize.** In multi-turn mode, if tools are available
  but the model has made zero tool calls by `_IDLE_TOOL_TURNS_LIMIT` (3)
  turns, GAZE force-finalizes to avoid wasting tokens. It nudges once;
  if the model still does not use tools, it accepts the current response.
- **Final-turn tool stripping.** The last turn never offers tools. If the
  model still emits tool calls on that turn, GAZE first tries to
  salvage a valid JSON answer from the accompanying text; failing that, it
  raises `AgenticProcessingError`.
- **Truncation salvage.** When the final turn is cut off (`finish_reason ==
  "length"`), GAZE attempts to extract partial JSON from the text and,
  if the salvaged keys match a sub-schema rather than the top level, wraps
  them under the correct parent key before validating.
- **Stale image stripping.** On turns after the first, base64 image data from
  earlier rounds is replaced with text placeholders to reduce payload size on
  subsequent API calls.
