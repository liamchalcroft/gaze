# Agentic Workflow

The agentic processing module enables multi-turn reasoning with visual tools and web search integration for medical image analysis.

## Architecture Overview

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
        WS[Web Search]

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
        CHECK{Confidence >= Threshold?}
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

    INIT --> WS

    WS --> T1
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

## Components

### NOVAAgenticProcessor

The NOVA-specific processor that orchestrates multi-turn analysis using the generic harness:

```python
from src.processor import NOVAAgenticProcessor

processor = NOVAAgenticProcessor(
    model_name="x-ai/grok-4.1-fast:free",
    use_tools=True,             # Enable visual manipulation tools
    max_turns=10,               # Maximum reasoning turns (max 20)
    reasoning_enabled=False,    # Enable reasoning for supported models
)

result = await processor.analyze(
    image_path=Path("scan.png"),
    metadata={"modality": "MRI", "plane": "axial"},
)
```

### Tool Registry

Visual manipulation tools the model can request during analysis:

| Tool | Parameters | Description |
|------|------------|-------------|
| `zoom` | `factor: float` | Magnify image (0.5-4.0x) |
| `crop` | `x1, y1, x2, y2` | Extract region of interest (normalized 0-1) |
| `adjust_contrast` | `factor: float` | Enhance/reduce contrast (0.5-3.0) |
| `threshold` | `lower, upper` | Intensity thresholding (0-255) |
| `flip_horizontal` | - | Mirror image left-right |
| `flip_vertical` | - | Mirror image top-bottom |
| `rotate` | `clockwise: bool` | Rotate 90 degrees |
| `reset` | - | Restore original image |
| `search_web` | `query: str, search_type: str` | Search medical literature (PubMed/Radiopaedia) with reliability scoring |

```python
from radiant_harness import ToolRegistry, create_visual_tools, create_search_tools

# Create registry with visual and search tools
tools = create_visual_tools() + create_search_tools()
registry = ToolRegistry(image_path=image_path, tools=tools)

# Model requests a tool
result = await registry.execute("zoom", factor=2.0)
# Returns: ToolResult(success=True, image_base64="...", description="Zoomed 2.0x")

# Orientation tools
result = await registry.execute("flip_horizontal")
result = await registry.execute("rotate", clockwise=True)

# Web search tool
result = await registry.execute("search_web", query="glioblastoma MRI", search_type="pubmed")
# Returns: ToolResult(success=True, metadata={...}, description="Found 5 reliable sources")
```

### Web Search Integration

Enhanced web search with medical reliability scoring:

```mermaid
flowchart LR
    QUERY[Search Query] --> WSM[WebSearchManager]
    WSM --> PUBMED[PubMed Search]
    WSM --> RADIOPAEDIA[Radiopaedia Search]
    WSM --> GENERAL[General Web Search]
    PUBMED --> SCORE[Reliability Scoring]
    RADIOPAEDIA --> SCORE
    GENERAL --> SCORE
    SCORE --> RESULTS[Ranked Results]
```

```python
from src.retrieval.enhanced_web_search import search_medical_literature_sync

# Search with automatic medical source prioritization
results = search_medical_literature_sync(
    query="brain MRI lesion differential diagnosis",
    max_results=5,
    search_type="pubmed"  # pubmed, radiopaedia, or general
)

# Each result includes:
for result in results:
    print(f"Title: {result.title}")
    print(f"Reliability: {result.reliability_score:.2f}")
    print(f"Medical relevance: {result.medical_relevance:.2f}")
    print(f"Key entities: {result.extracted_entities}")
```

#### Search Result Structure

```python
@dataclass
class SearchResult:
    title: str                    # Article/paper title
    url: str                      # Source URL
    content: str                  # Extracted content
    reliability_score: float      # 0.0-1.0 based on source authority
    medical_relevance: float      # 0.0-1.0 based on medical content
    extracted_entities: list[str] # Medical terms/concepts
```

## Multi-Turn Analysis Flow

```mermaid
sequenceDiagram
    participant U as User
    participant AP as AgenticProcessor
    participant TR as Tool Registry
    participant WS as Web Search
    participant M as VLM Model

    U->>AP: analyze(image, task, metadata)

    loop Turn 1..N (max_turns)
        AP->>M: Generate response with context
        M-->>AP: Response + tool calls (optional)

        alt Model requests web search
            AP->>WS: search_web(query, search_type)
            WS-->>AP: Reliable medical sources
            AP->>M: Continue with search results
        end

        alt Model requests visual tool
            AP->>TR: Execute tool (zoom, crop, flip, etc.)
            TR-->>AP: Modified image
            AP->>M: Continue with modified image
        end

        alt Has final answer
            AP-->>U: Final response
        end
    end

    AP-->>U: Final response after max_turns
```

### Turn Structure

Each turn produces:

```python
@dataclass
class Turn:
    role: str  # 'assistant' or 'tool_result'
    content: str
    tool_calls: list[dict]  # Tools requested by model
    tool_results: list[ToolResult]
```

### Result Structure

```python
@dataclass
class AgenticResult:
    final_response: dict[str, Any]
    turns: list[Turn]
    total_tokens: int
    search_results: list[SearchResult]  # Web search results
    confidence: float
```

## Task-Specific Processors

### Using NOVAAgenticProcessor for Different Tasks

The NOVAAgenticProcessor handles all tasks (captioning, diagnosis, localization) in a unified analysis:

```python
from src.processor import NOVAAgenticProcessor

processor = NOVAAgenticProcessor(
    model_name="openai/gpt-4o",
    use_tools=True,
    use_web_search=True,
    max_turns=10,
)

result = await processor.analyze(
    image_path=Path("brain_mri.png"),
    metadata={"history": "Headache for 2 weeks"},
)

# Access unified response
print(result.final_response["caption"])
print(result.final_response["diagnosis"])
print(result.final_response["localization"])
```

## CLI Usage

```bash
# Enable agentic processing with Grok model
python -m src.cli task=localization \
    model.name=x-ai/grok-4.1-fast:free \
    agentic.enabled=true

# Configure agentic options with reasoning
python -m src.cli \
    task=diagnosis \
    model.name=x-ai/grok-4.1-fast:free \
    agentic.enabled=true \
    agentic.use_tools=true \
    agentic.max_turns=10 \
    agentic.confidence_threshold=0.8 \
    model.reasoning_enabled=true

# Example with web search enabled
python -m src.cli \
    task=diagnosis \
    model.name=x-ai/grok-4.1-fast:free \
    agentic.enabled=true \
    agentic.use_tools=true \
    agentic.max_turns=15
```

## Configuration

```python
class AgenticConfig(BaseModel):
    enabled: bool = False
    use_tools: bool = True
    max_turns: int = 10         # 1-20 (increased from 10)
    confidence_threshold: float = 0.7  # 0.0-1.0
    reasoning_enabled: bool = False    # For models that support reasoning
```

## When to Use Agentic Mode

| Scenario | Recommended |
|----------|-------------|
| Simple, clear scans | Standard processor (faster) |
| Complex findings | Agentic with tools |
| Need literature search | Agentic with search_web (PubMed) |
| Need comparison images | Agentic with search_web (Radiopaedia) |
| Ambiguous cases | Agentic with web search + tools |
| Research/benchmarking | Standard for reproducibility |
| Clinical decision support | Agentic with all features |

## Web Search Best Practices

The enhanced `search_web` tool follows LLM agent best practices:

### 1. **Reliability Scoring**
- PubMed articles: 0.9-1.0 (peer-reviewed)
- Radiopaedia: 0.8-0.9 (expert-curated)
- Medical journals: 0.7-0.9 (varies by impact factor)
- General web: 0.3-0.7 (varies by source)

### 2. **LLM-Friendly Formatting**
```python
# Search results are automatically formatted for LLM consumption:
formatted_results = format_for_llm(results)
# Returns structured text with reliability indicators and medical entities
```

### 3. **Medical Entity Extraction**
- Automatic extraction of medical terms, conditions, and procedures
- Helps models understand relevance and context
- Supports differential diagnosis reasoning

### 4. **Error Handling & Retry Logic**
- Graceful degradation when sources are unavailable
- Automatic retry with exponential backoff
- Fallback to general web search if specialized sources fail

### Usage Examples

```python
# Search for specific conditions
registry.execute("search_web",
    query="glioblastoma MRI findings differential diagnosis",
    search_type="pubmed")

# Search for imaging examples
registry.execute("search_web",
    query="meningioma MRI T1 with contrast radiopaedia",
    search_type="radiopaedia")

# General medical information
registry.execute("search_web",
    query="brain tumor types MRI characteristics",
    search_type="general")
```

## Architecture Benefits

1. **Fully Agentic**: No static knowledge base - the model actively searches
2. **Up-to-Date**: Always accesses latest medical literature
3. **Reliable**: Source prioritization and scoring ensure quality
4. **Flexible**: Supports multiple search types (PubMed, Radiopaedia, general)
5. **LLM-Optimized**: Results formatted for effective LLM reasoning
