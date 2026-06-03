# LM Studio adapter

`LMStudioAdapter` targets a local LM Studio server. It subclasses the OpenAI
adapter but permits `http://` URLs, drops `response_format`, and detects context
overflow for clearer errors. For how to supply it via the `adapter_factory`
pattern, see [Getting started](../../getting-started.md).

::: gaze.models.lmstudio_adapter
