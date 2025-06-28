# Enhanced Prompt System with Jinja2 System Prompts

## Overview

The NOVA medical image analysis system now features an enhanced prompt loading system that uses Jinja2 templates for system prompts, following best practices for prompt management in GenAI applications. This system provides better organization, maintainability, and flexibility compared to the previous hardcoded system prompts.

## Architecture

### System Prompt Templates

The system uses a hierarchical template structure with Jinja2 inheritance:

```
src/nova_retrieval_vlm/prompts/system/
├── base.jinja                    # Base template with common instructions
├── baseline.jinja               # Baseline analysis mode
├── multiturn.jinja              # Multi-turn reasoning mode
├── visual.jinja                 # Visual operations mode
├── retrieval.jinja              # Retrieval-augmented mode
├── web_search.jinja             # Web search integration mode
└── comprehensive.jinja          # Comprehensive mode (all capabilities)
```

### Template Inheritance

All system prompt templates extend the base template (`base.jinja`) which provides:

- **Core Capabilities**: Basic medical image analysis capabilities
- **Analysis Guidelines**: Clinical accuracy and systematic analysis principles
- **Medical Ethics**: Professional responsibility and patient safety
- **Communication**: Medical terminology and structured reporting
- **Output Format**: JSON-only response requirements

Child templates can:
- Extend parent content using `{{ super() }}`
- Override specific blocks with mode-specific instructions
- Add conditional content based on context variables

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
from nova_retrieval_vlm.prompts.enhanced_prompt_loader import (
    load_enhanced_prompt, 
    get_system_prompt
)

# Load enhanced prompt with automatic mode detection
prompt = load_enhanced_prompt(
    template_name="baseline/caption.jinja",
    image_path=image_path,
    passages=passages,
    metadata=metadata,
)

# Get system prompt for specific mode
system_prompt = get_system_prompt("multiturn", {"use_retrieval": True})
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
from nova_retrieval_vlm.prompts.enhanced_prompt_loader import get_enhanced_loader

loader = get_enhanced_loader()

# List available modes
modes = loader.list_available_modes()
print(f"Available modes: {modes}")

# Validate mode
is_valid = loader.validate_mode("multiturn")

# Get system prompt with custom context
system_prompt = loader.get_system_prompt(
    "visual", 
    {
        "use_retrieval": True,
        "custom_setting": "value"
    }
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

## Benefits

### 1. **Maintainability**
- System prompts are stored as separate Jinja templates
- Easy to modify without touching code
- Version control friendly

### 2. **Reusability**
- Template inheritance reduces duplication
- Common instructions in base template
- Mode-specific additions in child templates

### 3. **Flexibility**
- Context variables for dynamic content
- Conditional blocks based on configuration
- Easy to add new modes and capabilities

### 4. **Consistency**
- Standardized structure across all modes
- Consistent medical terminology and guidelines
- Uniform output format requirements

### 5. **Extensibility**
- Easy to add new system prompt modes
- Support for complex conditional logic
- Integration with existing prompt templates

## Testing

The enhanced prompt system includes comprehensive testing:

```bash
# Run the test suite
python test_enhanced_prompts.py
```

Tests cover:
- System prompt template loading
- Template inheritance
- Context variable handling
- Mode detection
- Enhanced prompt combination

## Migration from Old System

The enhanced prompt loader is backward compatible with existing code:

### Old Usage
```python
from nova_retrieval_vlm.prompts.prompt_loader import load_prompt
prompt = load_prompt(template_name, image_path, passages, metadata)
```

### New Usage
```python
from nova_retrieval_vlm.prompts.enhanced_prompt_loader import load_enhanced_prompt
prompt = load_enhanced_prompt(template_name, image_path, passages, metadata)
```

The new system automatically detects the appropriate mode and combines system prompts with task-specific prompts.

## Best Practices

### 1. **Template Design**
- Keep base template focused on common instructions
- Use clear, descriptive block names
- Include comprehensive medical guidelines

### 2. **Mode Selection**
- Choose the simplest mode that meets requirements
- Use comprehensive mode only for complex cases
- Consider performance implications of advanced modes

### 3. **Context Variables**
- Use meaningful variable names
- Document expected values
- Provide sensible defaults

### 4. **Testing**
- Test all modes with various contexts
- Verify template inheritance works correctly
- Ensure backward compatibility

## Troubleshooting

### Common Issues

1. **Template Not Found**
   - Check template path in `system_modes` dictionary
   - Verify Jinja template exists in correct location

2. **Inheritance Issues**
   - Ensure templates extend base template correctly
   - Use `{{ super() }}` to include parent content

3. **Context Variables**
   - Check variable names in templates
   - Ensure variables are passed correctly

4. **Mode Detection**
   - Verify template naming conventions
   - Check `_detect_mode_from_template` logic

### Debugging

Enable debug logging to see template loading details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Dynamic Mode Selection**: AI-powered mode selection based on case complexity
2. **Template Versioning**: Support for multiple template versions
3. **Performance Optimization**: Caching of rendered templates
4. **Advanced Inheritance**: Support for multiple inheritance levels
5. **Template Validation**: Automated validation of template structure

### Contributing

To contribute to the enhanced prompt system:

1. Follow the existing template structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Follow medical terminology standards

## References

- [Jinja2 Prompting Guide](https://medium.com/@alecgg27895/jinja2-prompting-a-guide-on-using-jinja2-templates-for-prompt-management-in-genai-applications-e36e5c1243cf)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [LangChain Integration](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/) 