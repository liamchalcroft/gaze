# Model Expansion Plan

This document outlines the plan for expanding the NOVA evaluation framework to support additional VLM models.

## 🎯 Current Status

### Default Model
- **Primary**: `x-ai/grok-4.1-fast:free`
- **Status**: ✅ Fully integrated and tested
- **Features**: Strong reasoning capabilities, free tier access

## 🚀 Future Models to Add

### High Priority Models

#### 1. Zhipu AI GLM-4V
- **Identifier**: `z-ai/glm-4.5v`
- **Status**: Available on OpenRouter
- **Features**: Multimodal understanding, strong Chinese medical imaging
- **Integration**: Add to benchmark configs when ready

#### 2. Alibaba Qwen3-VL Series
- **Standard**: `qwen/qwen3-vl-235b-a22b-instruct`
- **Thinking**: `qwen/qwen3-vl-235b-a22b-thinking`
- **Features**: Chain-of-thought reasoning, strong VQA capabilities
- **Integration**: Add both variants for comparison

#### 3. StepFun AI Step-3
- **Identifier**: `stepfun-ai/step3`
- **Status**: Available on OpenRouter
- **Features**: Advanced multimodal understanding
- **Integration**: Add to benchmark suite

### Medium Priority Models

#### OpenAI Models
- `openai/gpt-4o` - GPT-4o (when API access available)
- `openai/gpt-4o-mini` - GPT-4o-mini (when API access available)
- **Features**: State-of-the-art performance, widespread adoption

#### Other OpenRouter Models
- `mistralai/mixtral-8x7b` - Mixtral 8x7B
- `meta-llama/llama-3.1-70b-instruct` - Llama 3.1 70B
- `microsoft/wizardlm-2-8x22b` - WizardLM 2 8x22B

## 🔧 Integration Steps

### Phase 1: Add High Priority Models
1. **GLM-4V Integration**
   ```bash
   # Add to benchmark configs
   "models": ["x-ai/grok-4.1-fast:free", "z-ai/glm-4.5v"]
   ```

2. **Qwen3-VL Integration**
   ```bash
   # Add both variants
   "models": [
       "x-ai/grok-4.1-fast:free",
       "z-ai/glm-4.5v",
       "qwen/qwen3-vl-235b-a22b-instruct",
       "qwen/qwen3-vl-235b-a22b-thinking"
   ]
   ```

### Phase 2: Comprehensive Model Set
```python
COMPREHENSIVE_MODELS = {
    "current": [
        "x-ai/grok-4.1-fast:free",
        "z-ai/glm-4.5v",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "qwen/qwen3-vl-235b-a22b-thinking",
        "stepfun-ai/step3"
    ],
    "expansion": [
        # OpenRouter models to test
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "mistralai/mixtral-8x7b",
        "meta-llama/llama-3.1-70b-instruct",
        # Additional models as they become available
    ]
}
```

## 📋 Implementation Checklist

### For Each New Model

#### 1. Model Compatibility Testing
```bash
# Test basic functionality
uv run python scripts/evaluate_nova_dataset.py \
    --task localization \
    --model <model_id> \
    --batch-size 1 \
    --output-dir ./test_<model_id> \
    --verbose
```

#### 2. Update Configuration Files
- Add to `scripts/run_comprehensive_benchmark.py` configs
- Update documentation in `docs/evaluation_guide.md`
- Update help text in shell wrapper

#### 3. Performance Validation
- Compare with baseline model
- Check token usage and cost
- Validate quality of medical imaging understanding

#### 4. Update Documentation
- Add model capabilities notes
- Include cost/usage information
- Update model recommendations

## 🎯 Model Selection Criteria

### Medical Imaging Specific Requirements
1. **Anatomical Understanding**: Strong medical domain knowledge
2. **Visual Acuity**: High-resolution image analysis capability
3. **Reasoning Skills**: Chain-of-thought and analytical reasoning
4. **Clinical Accuracy**: Consistent and reliable medical interpretations
5. **API Reliability**: Stable access and reasonable pricing

### Technical Requirements
1. **Vision Support**: Robust image processing capabilities
2. **API Integration**: Stable OpenRouter/OpenAI compatible API
3. **Rate Limits**: Reasonable request limits for batch processing
4. **JSON Output**: Structured response format compatibility
5. **Cost Effectiveness**: Sustainable pricing for research use

## 📊 Future Evaluation Matrix

### Model Comparison Framework
When multiple models are available, run systematic comparisons:

```bash
# Comprehensive model comparison
uv run python scripts/run_comprehensive_benchmark.py \
    --models x-ai/grok-4.1-fast:free z-ai/glm-4.5v qwen/qwen3-vl-235b-a22b-instruct \
    --preset comprehensive \
    --output-dir ./runs/model_comparison
```

### Evaluation Metrics
- **Task Performance**: mAP, accuracy, F1 scores
- **Cost Efficiency**: Cost per prediction
- **Speed**: Tokens per second, response time
- **Quality**: Medical imaging accuracy
- **Reliability**: Consistency across runs

## 🔍 Testing Strategy

### Phase 1: Single Model Testing
- Test each model individually
- Validate basic functionality
- Check output format compatibility
- Identify any model-specific issues

### Phase 2: Comparative Analysis
- Side-by-side model comparisons
- Statistical significance testing
- Performance regression detection
- Cost-benefit analysis

### Phase 3: Production Integration
- Full dataset evaluation
- Long-running stability tests
- Resource usage monitoring
- Error handling validation

## 📝 Integration Notes

### Configuration Updates
When adding models, update:
1. `scripts/run_comprehensive_benchmark.py` - Add to preset configs
2. `docs/evaluation_guide.md` - Update model options
3. `scripts/eval_nova.sh` - Update help text
4. `docs/model_expansion_plan.md` - Update this document

### Version Control
- Track model performance changes over time
- Maintain compatibility notes
- Document any breaking changes
- Archive older benchmark results

---

**Timeline**:
- ✅ **Phase 0**: Default Grok 4.1 integration (Complete)
- 🔄 **Phase 1**: High priority models (Next)
- ⏳ **Phase 2**: Comprehensive evaluation (Future)

This plan ensures systematic, thorough integration of new VLM models while maintaining evaluation quality and reproducibility.