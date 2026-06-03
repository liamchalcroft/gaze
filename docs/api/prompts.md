# Prompts

Utilities for loading and rendering Jinja prompt templates with strict
undefined-variable behaviour: a reference to a missing context variable raises
`TemplateError` rather than rendering a silently incomplete prompt. `AnalysisMode`
selects the template subdirectory (`agentic` or `single_turn`). See
[Architecture](../architecture.md) for how processors assemble prompts.

::: gaze.prompts
