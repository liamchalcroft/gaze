# Multi-turn Prompting System

## Overview

The multi-turn system allows the model to decide whether to continue analysis or proceed to final output. This enables efficient handling of both simple cases (1 step) and complex cases (2-3 steps).

## How It Works

### Step 1: Initial Analysis
- Model analyzes the image
- Decides if additional analysis is needed based on confidence
- Returns `continue_analysis: true/false`

### Step 2: Guideline Integration (optional)
- Consults clinical guidelines
- Refines differential diagnosis
- Decides if further analysis needed

### Step 3: Final Analysis (optional)
- Completes detailed analysis
- Confirms readiness for task output

## Continuation Logic

**Continue when:**
- Findings are complex or ambiguous
- Multiple differential diagnoses need refinement
- Confidence is below threshold (typically < 0.8)

**Stop when:**
- Clear diagnosis with high confidence
- No additional information would change assessment

## Response Schemas

### Step 1
```json
{
  "technical_assessment": "string",
  "detailed_findings": "string",
  "differential_diagnosis": ["string"],
  "confidence": 0.0-1.0,
  "continue_analysis": true | false,
  "continuation_reason": "string"
}
```

### Step 2
```json
{
  "technical_assessment": "string",
  "guideline_analysis": "string",
  "refined_differential": ["string"],
  "confidence": 0.0-1.0,
  "continue_analysis": true | false,
  "continuation_reason": "string"
}
```

### Step 3
```json
{
  "final_assessment": "string",
  "confidence": 0.0-1.0,
  "analysis_complete": true | false,
  "completion_reason": "string"
}
```

## Visual Operations Schema

For visual multi-turn mode:

```json
{
  "zoom_factor": float | null,
  "crop_box": [x1, y1, x2, y2] | null,
  "contrast_factor": float | null,
  "intensity_range": [min_val, max_val] | null,
  "reset_to_original": boolean,
  "need_more_ops": boolean,
  "continue_analysis": boolean,
  "continuation_reason": "string",
  "analysis_notes": "string"
}
```

## CLI Usage

```bash
# Multi-turn analysis
python -m nova_retrieval_vlm.cli task=diagnosis approach=multiturn

# With agentic tools
python -m nova_retrieval_vlm.cli task=diagnosis agentic.enabled=true agentic.use_tools=true
```

## Template Files

Located in `src/nova_retrieval_vlm/prompts/`:

- `multiturn/step1.jinja` - Initial analysis
- `multiturn/step2.jinja` - Guideline integration
- `multiturn/step3.jinja` - Final analysis
- `system/multiturn.jinja` - System prompt
