# Prompt System

## Overview

The prompt system uses Jinja2 templates for both system prompts and task-specific prompts. System prompts are stored in `prompts/system/` with template inheritance from a base template.

## Architecture

### System Prompt Templates

```
src/nova_retrieval_vlm/prompts/system/
├── base.jinja           # Base template with common instructions
├── baseline.jinja       # Baseline analysis mode
├── multiturn.jinja      # Multi-turn reasoning mode
├── visual.jinja         # Visual operations mode
├── retrieval.jinja      # Retrieval-augmented mode
├── web_search.jinja     # Web search integration mode
└── comprehensive.jinja  # Comprehensive mode (all capabilities)
```

### Template Inheritance

System prompt templates extend `base.jinja` which provides:

- Core medical image analysis capabilities
- Clinical accuracy guidelines
- Medical ethics requirements
- JSON output format specifications

Child templates override blocks with mode-specific instructions.

## Available Modes

### 1. Baseline Mode (`baseline`)
- **Purpose**: Standard single-turn analysis
- **Use Case**: Basic medical image analysis tasks
- **Template**: `system/baseline.jinja`
- **CLI Usage**: `approach=baseline`

### 2. Multi-Turn Mode (`multiturn`)
- **Purpose**: Iterative, step-by-step reasoning
- **Use Case**: Complex cases requiring systematic analysis
- **Template**: `system/multiturn.jinja`
- **CLI Usage**: `approach=multiturn`
- **Features**:
  - Step-by-step reasoning process
  - Evidence gathering and hypothesis formation
  - Confidence calibration
  - Retrieval integration (optional)

### 3. Visual Mode (`visual`)
- **Purpose**: Visual operations and web search integration
- **Use Case**: Cases requiring image manipulation and external information
- **Template**: `system/visual.jinja`
- **CLI Usage**: `approach=visual_multiturn`
- **Features**:
  - Visual operations (zoom, crop, contrast, thresholding)
  - Web search integration
  - Multi-modal reasoning
  - Iterative visual analysis

### 4. Retrieval Mode (`retrieval`)
- **Purpose**: Evidence-based analysis with medical knowledge retrieval
- **Use Case**: Cases requiring guideline and research integration
- **Template**: `system/retrieval.jinja`
- **CLI Usage**: `approach=baseline use_retrieval=true`
- **Features**:
  - Medical knowledge retrieval
  - Clinical guideline integration
  - Evidence-based conclusions
  - Source evaluation

### 5. Web Search Mode (`web_search`)
- **Purpose**: Real-time medical information access
- **Use Case**: Cases requiring current medical information
- **Template**: `system/web_search.jinja`
- **CLI Usage**: `approach=visual_multiturn` (with web search enabled)
- **Features**:
  - Real-time medical information access
  - Research integration
  - Source authority assessment
  - Current guideline access

### 6. Comprehensive Mode (`comprehensive`)
- **Purpose**: All capabilities combined
- **Use Case**: Most complex cases requiring all available tools
- **Template**: `system/comprehensive.jinja`
- **CLI Usage**: `approach=visual_multiturn` (with all features enabled)
- **Features**:
  - All visual, retrieval, and web search capabilities
  - Multi-turn reasoning
  - Evidence synthesis
  - Clinical correlation

## Usage

### Basic Usage

```python
from nova_retrieval_vlm.prompts.prompt_loader import (
    create_enhanced_prompt,
    load_prompt,
    get_mode_from_template,
)
from nova_retrieval_vlm.prompts.system_prompts import get_system_prompt

# Load prompt with automatic mode detection
prompt = create_enhanced_prompt(
    template_name="baseline/caption.jinja",
    image_path=image_path,
    passages=passages,
    metadata=metadata,
)

# Get system prompt for specific mode
system_prompt = get_system_prompt("multiturn")
```

### CLI Usage

```bash
# Baseline analysis
python -m nova_retrieval_vlm.cli task=caption approach=baseline

# Multi-turn analysis with retrieval
python -m nova_retrieval_vlm.cli task=diagnosis approach=multiturn use_retrieval=true

# Visual multi-turn with web search
python -m nova_retrieval_vlm.cli task=localization approach=visual_multiturn visual_rounds=3
```

### Advanced Usage

```python
from nova_retrieval_vlm.prompts.prompt_loader import (
    load_prompt,
    get_mode_from_template,
    combine_prompts,
)
from nova_retrieval_vlm.prompts.system_prompts import get_system_prompt

# Available modes
modes = ["baseline", "multiturn", "visual", "retrieval", "web_search", "comprehensive"]

# Get mode from template name
mode = get_mode_from_template("multiturn/step1.jinja")  # Returns "multiturn"

# Load prompt with explicit mode
prompt = load_prompt(
    template_name="baseline/diagnosis.jinja",
    image_path=image_path,
    passages=passages,
    metadata=metadata,
    mode="baseline",
)
```

## Configuration

### Context Variables

System prompts support context variables that can be passed to customize behavior:

- `use_retrieval`: Enable/disable retrieval-specific instructions
- `custom_setting`: Any custom variable for template logic

### Template Customization

To customize system prompts:

1. **Modify existing templates**: Edit the Jinja files in `src/nova_retrieval_vlm/prompts/system/`
2. **Add new modes**: Create new Jinja templates and add them to the loader
3. **Override system prompts**: Use `system_prompt_override` parameter

### Adding New Modes

1. Create a new Jinja template in `src/nova_retrieval_vlm/prompts/system/`
2. Extend the base template: `{% extends "system/base.jinja" %}`
3. Override blocks as needed using `{{ super() }}`
4. Add the mode to the loader's `system_modes` dictionary

## Troubleshooting

**Template Not Found**: Check template path exists in `prompts/system/`

**Mode Detection**: Template naming conventions:
- `multiturn/` prefix → "multiturn" mode
- `visual_multiturn/` prefix → "visual" mode
- `retrieval_` prefix → "retrieval" mode
- `baseline/` prefix → "baseline" mode

**Debugging**: Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
