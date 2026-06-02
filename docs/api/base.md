# Base processor

`AgenticProcessorBase` drives the multi-turn agentic loop: it prompts the VLM,
runs any requested tools, and collects structured JSON output across turns. You
subclass it and implement four methods to define a task. For how the loop fits
together with adapters, the tool registry, and prompts, see
[Architecture](../architecture.md).

::: gaze.base
