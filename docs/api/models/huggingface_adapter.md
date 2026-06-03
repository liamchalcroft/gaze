# HuggingFace adapter

`HuggingFaceAdapter` and `HuggingFaceVLMAdapter` run local HuggingFace models
(text and vision-language) as an alternative to the OpenAI and LM Studio adapters.
They are imported lazily so the core install stays torch-free; install `torch` and
`transformers` (for example via the `nova` extra) before use. Supply one through
the `adapter_factory` argument, as shown in
[Getting started](../../getting-started.md).

::: gaze.models.huggingface_adapter
