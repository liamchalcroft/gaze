# System Prompts

System prompts for the NOVA medical image analysis system. Each mode has a specialized prompt optimized for its analysis approach.

## Overview

System prompts define:
- AI role and capabilities
- Medical analysis guidelines
- Output format requirements (JSON)
- Mode-specific behaviors

## Available Modes

### 1. Baseline Mode (`baseline`)
**Purpose**: Standard medical image analysis for captioning, diagnosis, and localization.

**Key Features**:
- Core medical image analysis capabilities
- Clinical diagnosis and differential diagnosis
- Anatomical localization
- Evidence-based reasoning
- JSON output format enforcement

**Best For**: Standard medical image analysis tasks with single-turn processing.

### 2. Multi-Turn Mode (`multiturn`)
**Purpose**: Iterative, multi-step analysis with progressive reasoning refinement.

**Key Features**:
- Iterative analysis through multiple reasoning steps
- Evidence gathering and hypothesis formation
- Confidence calibration
- Retrieval-augmented reasoning
- Systematic reasoning process

**Best For**: Complex cases requiring step-by-step analysis and reasoning.

### 3. Visual Multi-Turn Mode (`visual`)
**Purpose**: Advanced visual analysis with image manipulation and web search integration.

**Key Features**:
- Visual operations (zoom, crop, contrast adjustment)
- Iterative visual examination
- Web search integration for medical information
- Multi-modal reasoning
- Region-specific focus

**Best For**: Cases requiring detailed visual examination and external information gathering.

### 4. Retrieval Mode (`retrieval`)
**Purpose**: Evidence-based analysis enhanced by medical knowledge retrieval.

**Key Features**:
- Medical knowledge retrieval from guidelines and research
- Evidence-based analysis
- Clinical guideline integration
- Differential diagnosis support
- Anatomical knowledge enhancement

**Best For**: Cases requiring access to current medical guidelines and research.

### 5. Web Search Mode (`web_search`)
**Purpose**: Real-time analysis with current medical information from web sources.

**Key Features**:
- Real-time medical information access
- Research integration
- Clinical guideline access
- Source evaluation and attribution
- Evidence-based practice

**Best For**: Cases requiring current medical information and research findings.

### 6. Comprehensive Mode (`comprehensive`)
**Purpose**: Full integration of all capabilities for maximum analysis depth.

**Key Features**:
- All visual, retrieval, and web search capabilities
- Multi-turn reasoning
- Evidence synthesis
- Clinical correlation
- Comprehensive assessment

**Best For**: Complex cases requiring the most thorough analysis possible.

## Usage

### Basic Usage

```python
from nova_retrieval_vlm.prompts.system_prompts import get_system_prompt

# Get a system prompt for a specific mode
system_prompt = get_system_prompt("baseline")
```

### Enhanced Prompt Creation

```python
from nova_retrieval_vlm.prompts.prompt_loader import create_enhanced_prompt

# Create an enhanced prompt with system prompt integration
enhanced_prompt = create_enhanced_prompt(
    template_name="baseline/diagnosis.jinja",
    image_path=Path("/path/to/image.png"),
    passages=["Medical guideline passage 1", "Research finding 2"],
    metadata={
        "image_id": 0,
        "width": 512,
        "height": 512,
        "clinical_history": "Patient presents with..."
    },
    mode="baseline"  # Optional - auto-detected from template name
)
```

### Automatic Mode Detection

The system can automatically detect the appropriate mode from template names:

```python
from nova_retrieval_vlm.prompts.prompt_loader import get_mode_from_template

# Automatic mode detection
mode = get_mode_from_template("multiturn/step1.jinja")  # Returns "multiturn"
mode = get_mode_from_template("visual_multiturn/step1.jinja")  # Returns "visual"
mode = get_mode_from_template("retrieval_diagnosis.jinja")  # Returns "retrieval"
```

## Prompt Structure

Each system prompt follows a consistent structure with the following sections:

### 1. Role Definition
Clear definition of the AI's role as a medical image analysis assistant.

### 2. Core Capabilities
List of specific capabilities for the given mode.

### 3. Analysis Guidelines
Critical guidelines for medical analysis, including:
- Clinical accuracy requirements
- Comprehensive analysis approach
- Differential diagnosis considerations
- Anatomical precision
- Clinical relevance focus
- Uncertainty acknowledgment

### 4. Output Requirements
Specific JSON format requirements and output structure.

### 5. Medical Ethics
Professional responsibility, patient safety, and clinical context considerations.

### 6. Communication Guidelines
Standards for medical terminology, reasoning presentation, and professional tone.

## Integration with CLI

The system prompts are automatically integrated into the CLI processing pipeline:

1. **Mode Detection**: The system automatically detects the appropriate mode based on the template name
2. **System Prompt Selection**: The appropriate system prompt is selected for the detected mode
3. **Prompt Combination**: The system prompt is combined with the task-specific Jinja template
4. **Enhanced Output**: The combined prompt provides comprehensive instructions to the model

## Customization

### Overriding System Prompts

You can provide custom system prompts:

```python
enhanced_prompt = create_enhanced_prompt(
    template_name="baseline/diagnosis.jinja",
    image_path=image_path,
    passages=passages,
    metadata=metadata,
    system_prompt_override="Your custom system prompt here..."
)
```

### Adding New Modes

To add a new mode:

1. Add the system prompt to `src/nova_retrieval_vlm/prompts/system_prompts.py`
2. Update the `get_system_prompt()` function
3. Add mode detection logic to `get_mode_from_template()`
4. Update the `combine_prompts()` function if needed

## Best Practices

### 1. Mode Selection
- Use **baseline** for standard analysis tasks
- Use **multiturn** for complex reasoning tasks
- Use **visual** when detailed image examination is needed
- Use **retrieval** when medical knowledge access is important
- Use **web_search** when current information is critical
- Use **comprehensive** for the most thorough analysis

### 2. Clinical Safety
- Always prioritize patient safety in assessments
- Acknowledge limitations and uncertainties
- Recommend clinical correlation when appropriate
- Use evidence-based reasoning

### 3. Output Quality
- Ensure JSON format compliance
- Provide clear reasoning for conclusions
- Use precise medical terminology
- Structure responses logically

## Examples

### Baseline Analysis
```python
# Standard medical image analysis
prompt = create_enhanced_prompt(
    template_name="baseline/diagnosis.jinja",
    image_path=image_path,
    passages=[],
    metadata={"clinical_history": "..."},
    mode="baseline"
)
```

### Multi-Turn Analysis
```python
# Iterative reasoning with retrieval
prompt = create_enhanced_prompt(
    template_name="multiturn/step1.jinja",
    image_path=image_path,
    passages=retrieved_passages,
    metadata={"step1_summary": "..."},
    mode="multiturn"
)
```

### Visual Analysis
```python
# Visual operations with web search
prompt = create_enhanced_prompt(
    template_name="visual_multiturn/step1.jinja",
    image_path=image_path,
    passages=[],
    metadata={"visual_operations": "..."},
    mode="visual"
)
```

## Troubleshooting

### Common Issues

1. **Mode Detection Fails**: Ensure template names follow the expected patterns
2. **System Prompt Not Found**: Check that the mode is supported in `get_system_prompt()`
3. **Template Loading Error**: Verify Jinja template exists and is properly formatted

### Debugging

Enable debug logging to see prompt creation details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

