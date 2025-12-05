# 🚨 CRITICAL FINDING: Why Baseline Outperforms Agentic Reasoning

## 📊 **The Shocking Truth**

**Both configurations are getting completely different diagnoses for the same images!**

### 🔍 **Case Study: Subject 3**

| Configuration | Diagnosis | Confidence | Ground Truth |
|---------------|-----------|------------|-------------|
| **Baseline** | "Agenesis of the corpus callosum" | 0.98 | ❌ WRONG |
| **Agentic** | "Colloid cyst" | 0.95 | ❌ WRONG |
| **Ground Truth** | "Intracranial epidermoid cyst" | — | ✅ CORRECT |

**Both models are wrong, but they're wrong in different ways!**

## 🎯 **Root Cause Analysis**

### 1. **Different Prompts = Different Diagnoses**
The same medical image produces completely different diagnoses depending on:
- Prompt structure and wording
- System prompt differences
- Reasoning context (single-turn vs multi-turn)

### 2. **Prompt Sensitivity Issues**
- **Baseline**: Longer, more structured prompts → different focus areas
- **Agentic**: Reasoning prompts → leads to overthinking or different interpretation
- **Ground Truth**: "Intracranial epidermoid cyst" (neither got this right!)

### 3. **Semantic Matching Penalty**
The semantic matching judge (using LLM) may:
- Penalize different types of wrong answers differently
- Favor certain diagnostic patterns over others
- Be inconsistent in what it considers "close enough"

## 🚨 **Critical Issues Identified**

### **A. Model Inconsistency**
- Same input + same model = **different outputs** ❌
- This indicates **prompt engineering problems**
- Not a reasoning failure per se

### **B. Both Are Wrong**
- Neither configuration got the right diagnosis
- Suggests **fundamental prompt/image interpretation issues**
- May need better medical domain prompting

### **C. Evaluation Methodology**
- Semantic matching may have biases
- Different wrong answers get different semantic scores
- May not be measuring true diagnostic accuracy

## 🔧 **Immediate Recommendations**

### 1. **Standardize Prompts**
```yaml
# Both configs should use identical prompts
# Only difference should be: reasoning_enabled = true/false
```

### 2. **Fix Medical Image Interpretation**
- Add better medical imaging guidance
- Include typical diagnostic patterns
- Use medical terminology consistently

### 3. **Investigate Prompt Differences**
- Compare exact prompts between baseline and agentic
- Identify wording that causes different diagnoses
- Test minimal prompt variations

### 4. **Validate Ground Truth**
- Check if ground truth diagnoses are reliable
- Verify image-diagnosis mappings
- Consider alternative evaluation methods

## 💡 **The Real Question**

**The issue isn't "baseline vs agentic" - it's that the prompts are so different that they're asking the model to solve different problems!**

The solution is to:
1. **Use identical prompts** (except reasoning toggle)
2. **Fix the medical domain prompting**
3. **Verify the evaluation methodology**

This explains why baseline appears "better" - it's not necessarily better reasoning, just different prompting that happens to align better with the semantic matching evaluation!