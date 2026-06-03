# Adapter protocol

`AdapterProtocol` is the interface every model adapter implements, centred on
`generate_chat()`. Processors talk to models only through this protocol, so any
OpenAI-compatible backend can be wired in. For how to pass an adapter via the
`adapter_factory` pattern, see [Getting started](../../getting-started.md).

::: gaze.models.adapter_protocol
