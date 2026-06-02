# Getting Started

## Installation

```bash
pip install gaze-vlm
```

For development:

```bash
git clone https://github.com/liamchalcroft/gaze.git
cd gaze
uv sync
```

## Your First Processor

The core abstraction is `AgenticProcessorBase`. Subclass it and implement four methods:

```python
import asyncio
from pathlib import Path
from gaze import AgenticProcessorBase

class XRayProcessor(AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a radiologist. Describe findings in the provided chest X-ray."

    def get_user_message(self, images, metadata):
        return f"Patient history: {metadata.get('history', 'None provided')}"

    def get_response_schema(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "xray_report",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "findings": {"type": "string"},
                        "impression": {"type": "string"},
                        "continue": {"type": "boolean"},
                    },
                    "required": ["findings", "impression", "continue"],
                    "additionalProperties": False,
                },
            },
        }

    def validate_response(self, response):
        return "findings" in response and "impression" in response
```

## Running the Processor

```python
async def main():
    # With OpenRouter (cloud)
    processor = XRayProcessor(
        model_name="openai/gpt-4o",
        use_tools=True,
        max_turns=3,
    )
    result = await processor.analyze(
        images=Path("chest_xray.jpg"),
        metadata={"modality": "XR", "history": "Cough for 2 weeks"},
    )
    print(result.final_response)

asyncio.run(main())
```

## Using Local Models

`AgenticProcessorBase` has no `base_url` constructor argument. To target a
local OpenAI-compatible server (such as LM Studio), pass an `adapter_factory`:
a zero-argument callable that returns a configured adapter. The base class
calls it lazily on the first `analyze()` and reuses the resulting adapter.

```python
from gaze import LMStudioAdapter

processor = XRayProcessor(
    model_name="qwen3.5-a3b",
    use_tools=True,
    adapter_factory=lambda: LMStudioAdapter(
        model_name="qwen3.5-a3b",
        base_url="http://localhost:1234/v1",
    ),
)
```

`LMStudioAdapter` permits `http://` URLs (unlike `OpenAIAdapter`, which
requires HTTPS), needs no real API key, and uses a 300s timeout suited to
local inference. When `base_url` is omitted it falls back to the
`LMSTUDIO_BASE_URL` environment variable, then to `http://localhost:1234/v1`.

Before launching a run you can fail fast if the model is not loaded:

```python
from gaze import require_lmstudio_model

await require_lmstudio_model(
    model_name="qwen3.5-a3b",
    base_url="http://localhost:1234/v1",
)
```

## Enabling Tools

When `use_tools=True`, the model can call built-in visual tools during
reasoning. Tools are only offered in multi-turn mode (`max_turns > 1`); the
final turn always withholds them so the model produces a clean answer. A
representative selection (see [Tools](tools.md) for all 25):

- **zoom** / **crop** -- magnify or extract a region of interest
- **adjust_contrast** / **adjust_brightness** / **adjust_sharpness** -- enhance visibility
- **window_level** -- apply clinical window/level presets or explicit center/width
- **threshold** / **detect_edges** -- highlight intensity ranges or boundaries

Web and image search are gated separately by `use_web_search=True`, which
enables two more tools:

- **search_web** -- query PubMed for medical literature
- **search_images** -- query NIH Open-i for reference images

The agentic loop continues until the model sets `"continue": false` or the turn limit is reached.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Yes (cloud) | API access |
| `NCBI_API_KEY` | No | Higher PubMed rate limits |
| `NCBI_EMAIL` | No | PubMed API compliance |

## Next Steps

- [Architecture](architecture.md) -- understand the framework design
- [Examples](examples.md) -- see complete applications
- [API Reference](api/index.md) -- full module documentation
