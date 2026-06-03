# Troubleshooting

Common issues when running GAZE, especially against local models.

## Thinking models need a larger token budget

Models that emit reasoning tokens (for example Qwen 3.5) count those tokens against `max_tokens`. Set `--max-tokens 4096` or higher (the run scripts default to 8192) so the model has room to reason and still produce the final JSON.

## LM Studio runs one model at a time

On typical hardware LM Studio serves a single model. A preflight health check that names a different model can trigger a swap (unload then reload). Point `require_lmstudio_model()` and the CLI at the model you actually have loaded.

## Empty model content

Some models (Qwen 3.5, GLM-4.6V) place their output in `reasoning_content` rather than `content`. GAZE detects an empty `content` and falls back to `reasoning_content` automatically, so no action is needed.

## Context overflow on local models

`LMStudioAdapter` detects context-size errors (HTTP 400 with "context size", "n_ctx", or "n_keep" in the message) and raises a clear `ModelError`. If you hit this, reduce `--max-image-dim`, lower `--max-turns`, or load a model with a larger context window. Agentic mode (system prompt plus tool schemas) needs more context than single-turn mode.

## Sending API keys to a custom host

For safety, `OpenAIAdapter` only sends credentials to an allowlisted set of hosts over HTTPS. To target a different cloud endpoint, set `GAZE_ALLOW_CUSTOM_BASE_URL=1`. Local LM Studio servers use `LMStudioAdapter`, which permits `http://` and needs no real key, so this flag is not required for local inference.
