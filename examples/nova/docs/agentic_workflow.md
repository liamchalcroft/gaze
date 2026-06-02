# Agentic Workflow

Multi-turn reasoning with visual tools and web search for medical image analysis.

## Architecture

```mermaid
flowchart TD
    subgraph Input
        IMG[Brain MRI Image]
        META[Metadata]
        TASK[Task: localization/diagnosis]
    end

    subgraph AgenticProcessor
        INIT[Initialize Components]
        TR[Tool Registry]

        subgraph Tools[Visual Tools]
            ZOOM[Zoom]
            CROP[Crop]
            CONTRAST[Contrast]
            THRESHOLD[Threshold]
            FLIP[Flip H/V]
            ROTATE[Rotate]
            RESET[Reset]
            SEARCH[Search Web/PubMed]
        end
    end

    subgraph MultiTurnLoop[Multi-Turn Analysis Loop]
        T1[Turn 1: Initial Analysis]
        T2[Turn 2: Refinement]
        TN[Turn N: Final Response]
        CHECK{continue == false?}
    end

    subgraph Output
        RESP[Final Response]
        EVAL[Evaluation Metrics]
    end

    IMG --> INIT
    META --> INIT
    TASK --> INIT

    INIT --> TR
    TR --> Tools
    T1 --> CHECK
    CHECK -->|No| T2
    T2 --> CHECK
    CHECK -->|No, turns < max| TN
    CHECK -->|Yes| RESP
    TN --> RESP

    Tools -.->|Model requests| T1
    Tools -.->|Model requests| T2

    RESP --> EVAL
```

## NOVAAgenticProcessor

```python
from examples.nova.src.processor import NOVAAgenticProcessor

processor = NOVAAgenticProcessor(
    model_name="openai/gpt-4o",
    use_tools=True,
    max_turns=10,
    reasoning_enabled=False,
)

result = await processor.analyze(
    images=Path("scan.png"),
    metadata={"modality": "MRI", "plane": "axial"},
)
```

`analyze()` accepts:
- `images` -- `Path`, `PIL.Image.Image`, list of either, or `None` for text-only
- `metadata` -- dict with clinical context
- `image_labels` -- optional labels for each image (e.g., `["T1", "T2-FLAIR"]`)

Returns `AgenticResult`.

## Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `zoom` | `factor: float` | Magnify image (0.5--4.0x) |
| `crop` | `box: [x1,y1,x2,y2]` | Extract region (normalized 0--1) |
| `adjust_contrast` | `factor: float` | Contrast enhancement (0.5--3.0) |
| `adjust_brightness` | `factor: float` | Brightness adjustment (0.5--3.0) |
| `adjust_sharpness` | `factor: float` | Sharpness adjustment (0.0--3.0) |
| `threshold` | `lower, upper` | Intensity thresholding (0--255) |
| `window_level` | `center, width` or `preset` | Clinical windowing (brain, subdural, bone, etc.) |
| `equalize_histogram` | -- | Global histogram equalization (grayscale) |
| `adaptive_equalize` | `clip_limit, tile_size` | CLAHE local contrast enhancement |
| `detect_edges` | `method: sobel\|laplacian` | Edge detection for boundary delineation |
| `denoise` | `sigma: float` | Gaussian noise reduction |
| `morphological` | `operation, iterations` | Erode/dilate/open/close for mask cleanup |
| `get_intensity_stats` | `box` (optional) | Compute intensity statistics over region |
| `intensity_profile` | `point1, point2` | Sample intensities along a line |
| `symmetry_diff` | -- | Left-right symmetry difference map |
| `invert` | -- | Invert pixel intensities (negative) |
| `annotate_region` | `box, color, label` | Draw bounding box overlay |
| `flip_horizontal` | -- | Mirror left-right |
| `flip_vertical` | -- | Mirror top-bottom |
| `rotate` | `clockwise: bool` | Rotate 90 degrees |
| `show_grid` | `divisions: int` | Overlay labeled spatial grid |
| `measure` | `point1, point2` | Measure distance between two points |
| `reset` | -- | Restore original image |
| `search_web` | `query, search_type` | PubMed literature search |
| `search_images` | `query, modality, body_part` | Open-i image search |

```python
from gaze import ToolRegistry, create_visual_tools, create_search_tools

tools = create_visual_tools() + create_search_tools()
registry = ToolRegistry(image_path=image_path, tools=tools)

result = await registry.execute_tool("zoom", factor=2.0)
# ToolResult(success=True, image_base64="...", description="Zoomed 2.0x")
```

## Multi-Turn Flow

```mermaid
sequenceDiagram
    participant U as User
    participant AP as AgenticProcessor
    participant TR as Tool Registry
    participant M as VLM Model

    U->>AP: analyze(images, metadata)

    loop Turn 1..N (max_turns)
        AP->>M: Generate response
        M-->>AP: JSON + tool calls (optional)

        alt Model requests tool
            AP->>TR: Execute tool
            TR-->>AP: ToolResult (image or text)
            AP->>M: Continue with result
        end

        alt continue == false
            AP-->>U: AgenticResult
        end
    end

    AP-->>U: AgenticResult (forced after max_turns)
```

Each turn, the model returns JSON with `"continue": true` or `"continue": false`. When `continue` is false (or max turns reached), the response is finalized.

## Result Types

```python
@dataclass(frozen=True)
class AgenticResult:
    final_response: dict[str, Any]   # Complete JSON from the model
    turns: list[Turn]                # All conversation turns
    total_tokens: int                # Tokens consumed across all turns
    confidence: float                # 0.0--1.0

@dataclass(frozen=True)
class Turn:
    role: TurnRole                   # "user", "assistant", or "tool_result"
    content: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    image_base64: str | None
```

## Web Search

PubMed search via NCBI E-utilities with reliability scoring. Search is invoked through the tool system:

```python
from gaze import ToolRegistry, create_visual_tools, create_search_tools

tools = create_visual_tools() + create_search_tools()
registry = ToolRegistry(image_path=image_path, tools=tools)

result = await registry.execute_tool(
    "search_web",
    query="brain MRI lesion differential diagnosis",
    search_type="diagnosis",
)
# result.metadata contains formatted_results, sources, avg_reliability
```

Reliability scores are based on source authority: PubMed articles (~0.95), government/academic domains (~0.80), major publishers (~0.85).

## CLI

```bash
# Visual tools
uv run python -m src.cli \
    --task localization \
    --model openai/gpt-4o \
    --use-tools

# Tools + web search
uv run python -m src.cli \
    --task diagnosis \
    --model openai/gpt-4o \
    --use-tools \
    --use-web-search \
    --max-turns 10
```
