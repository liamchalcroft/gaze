# Enhanced Multi-turn Prompting System

## Overview

The enhanced multi-turn prompting system allows the model to intelligently decide whether to continue with additional analysis steps or proceed directly to the final task output. This creates a more efficient and adaptive analysis process that can handle both simple cases (requiring only 1 step) and complex cases (requiring 2-3 steps).

## Key Features

### 1. Conditional Continuation
- **Step 1**: Model decides if additional analysis is needed
- **Step 2**: Model decides if final analysis is complete
- **Step 3**: Model confirms readiness for task-specific output

### 2. Flexible Step Execution
- **1 Step**: For simple, high-confidence cases
- **2 Steps**: For cases needing guideline consultation
- **3 Steps**: For complex cases requiring detailed analysis

### 3. Comprehensive Tracking
- Tracks which steps were completed
- Records continuation reasons for each step
- Maintains analysis confidence levels
- Logs total analysis steps taken

## System Components Updated

### 1. Multi-turn Prompts (`src/nova_retrieval_vlm/prompts/multiturn/`)

#### Step 1 (`step1.jinja`)
- Added `continue_analysis` boolean field
- Added `continuation_reason` string field
- Enhanced reasoning instructions for continuation decisions

#### Step 2 (`step2.jinja`)
- Added conditional continuation logic
- Enhanced guideline integration
- Improved confidence assessment

#### Step 3 (`step3.jinja`)
- New generic step for conditional analysis continuation
- Handles variable step results from previous steps
- Flexible metadata integration

#### Task-Specific Step 3 Templates
- `diagnosis_step3.jinja`: Updated for conditional step results
- `caption_step3.jinja`: Updated for conditional step results
- `localization_step3.jinja`: Updated for conditional step results

### 2. System Prompts (`src/nova_retrieval_vlm/prompts/system/`)

#### Multi-turn System (`multiturn.jinja`)
- Added conditional continuation capabilities
- Enhanced adaptive processing instructions
- Improved retrieval integration guidance
- Added continuation logic documentation

#### Visual System (`visual.jinja`)
- Added conditional continuation for visual analysis
- Enhanced visual operation guidance
- Improved web search integration
- Added adaptive processing capabilities

### 3. Visual Multi-turn Components

#### Visual Multi-turn Operations (`visual_multiturn/ops_request.jinja`)
- Added `continue_analysis` boolean field
- Added `continuation_reason` string field
- Enhanced continuation logic documentation
- Improved visual operation guidance

#### Visual Operations (`visual_ops/`)
- `step1.jinja`: Added conditional continuation logic
- `step2.jinja`: Added conditional continuation logic
- Enhanced visual analysis guidance

### 4. CLI Processing Functions

#### Enhanced Multi-turn Processing (`process_batch_multiturn`)
- Conditional step execution based on model flags
- Comprehensive tracking of analysis progress
- Flexible metadata handling
- Enhanced error handling and logging

#### Enhanced Visual Multi-turn Processing (`process_batch_visual_multiturn`)
- Conditional round execution based on model flags
- Improved visual operation handling
- Enhanced web search integration
- Comprehensive tracking of visual analysis

## Continuation Logic

### When to Continue Analysis
- Findings are complex or ambiguous
- Multiple differential diagnoses need refinement
- Clinical guidelines would improve confidence
- Additional analysis steps would enhance accuracy
- Confidence is below threshold (typically < 0.8)

### When to Stop Analysis
- Scan is clearly normal with high confidence
- Single clear diagnosis with high confidence
- No additional information would change the assessment
- Analysis is comprehensive and confident
- Ready to proceed to final task-specific output

## JSON Schema Updates

### Step 1 Response Schema
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

### Step 2 Response Schema
```json
{
  "technical_assessment": "string",
  "detailed_findings": "string",
  "guideline_analysis": "string",
  "refined_differential": ["string"],
  "confidence": 0.0-1.0,
  "continue_analysis": true | false,
  "continuation_reason": "string"
}
```

### Step 3 Response Schema
```json
{
  "additional_analysis": "string",
  "final_assessment": "string",
  "confidence": 0.0-1.0,
  "analysis_complete": true | false,
  "completion_reason": "string"
}
```

### Visual Operations Schema
```json
{
  "zoom_factor": float | null,
  "crop_box": [x1, y1, x2, y2] | null,
  "contrast_factor": float | null,
  "intensity_range": [min_val, max_val] | null,
  "reset_to_original": boolean,
  "need_more_ops": boolean,
  "continue_analysis": boolean,
  "continuation_reason": string,
  "analysis_notes": string,
  "web_search_requests": [...]
}
```

## Usage Examples

### Simple Case (1 Step)
```python
# Model determines high confidence in step 1
step1_result = {
    "confidence": 0.95,
    "continue_analysis": False,
    "continuation_reason": "Clear normal scan with high confidence"
}
# Analysis proceeds directly to final output
```

### Complex Case (3 Steps)
```python
# Step 1: Initial assessment
step1_result = {
    "confidence": 0.6,
    "continue_analysis": True,
    "continuation_reason": "Complex findings require guideline consultation"
}

# Step 2: Guideline analysis
step2_result = {
    "confidence": 0.8,
    "continue_analysis": True,
    "continuation_reason": "Multiple differential diagnoses need refinement"
}

# Step 3: Final analysis
step3_result = {
    "confidence": 0.9,
    "analysis_complete": True,
    "completion_reason": "All necessary analysis steps completed"
}
```

## Benefits

### 1. Efficiency
- Reduces unnecessary analysis steps for simple cases
- Optimizes processing time and cost
- Maintains accuracy for complex cases

### 2. Adaptability
- Handles varying case complexity automatically
- Adjusts analysis depth based on findings
- Provides consistent quality across all cases

### 3. Transparency
- Clear reasoning for continuation decisions
- Comprehensive tracking of analysis progress
- Detailed logging for debugging and optimization

### 4. Scalability
- Efficient resource utilization
- Consistent performance across different case types
- Easy to extend with additional analysis steps

## Testing

The enhanced system includes comprehensive testing:
- Continuation logic validation
- Step execution verification
- JSON schema compliance
- Error handling validation

Run tests with:
```bash
python scripts/test_enhanced_multiturn.py
```

## Future Enhancements

### Potential Improvements
1. **Dynamic Step Addition**: Allow models to request additional specialized analysis steps
2. **Confidence Thresholds**: Configurable confidence thresholds for different tasks
3. **Specialized Templates**: Task-specific continuation logic
4. **Performance Metrics**: Track efficiency improvements across different case types

### Integration Opportunities
1. **Active Learning**: Use continuation decisions for model improvement
2. **Quality Assurance**: Monitor continuation patterns for quality control
3. **Resource Optimization**: Analyze continuation patterns for system optimization

## Conclusion

The enhanced multi-turn prompting system represents a significant improvement in the efficiency and adaptability of medical image analysis. By allowing models to intelligently decide when additional analysis is needed, the system provides optimal performance across a wide range of case complexities while maintaining high accuracy and transparency. 