# Paper Skeleton Summary

## Overview
Complete LaTeX paper skeleton in `paper/main.tex` with accompanying bibliography in `paper/references.bib`. The paper is structured for submission to a venue like Nature Medicine, Nature Communications, npj Digital Medicine, or a top-tier ML conference (ICML, NeurIPS, ICLR with medical track).

## Paper Structure

### 1. Introduction (\S1)
- Motivation: Gap between single-shot VLM inference and clinical reasoning
- Problem: Iterative analysis, uncertainty quantification, information seeking
- Contribution: Three main contributions of Radiant Harness

### 2. Related Work (\S2)
Organized into four subsections:

#### 2.1 Vision-Language Models in Radiology
- **Key papers to compare:**
  - Med-Gemini (Saab et al., 2024) - SOTA foundation model
  - CheXagent (Chen et al., 2024) - Chest X-ray specialist
  - CXR-Agent (Chambon et al., 2024) - Tool-augmented CXR
  - AURA (Fathi et al., 2024) - Multi-modal agent

#### 2.2 Chain-of-Thought and Medical Reasoning
- **Key papers:**
  - Med-SCoT (Qiao et al., 2024) - Structured CoT
  - ChestX-Reasoner (Fan et al., 2025) - Step-by-step verification
  - M3CoTBench (Jiang et al., 2026) - Reasoning benchmark

#### 2.3 Agentic AI and Tool-Augmented Systems
- **Key papers:**
  - VoxelPrompt (Bar et al., 2024) - Grounded medical agent
  - Agentic Systems in Radiology survey (Blüethgen et al., 2025)
  - Toolformer (Schick et al., 2024) - Tool learning

#### 2.4 Medical Visual Question Answering
- **Key papers:**
  - Lin et al. (2023) survey
  - Zhang et al. (2024) large-scale dataset
  - Xu et al. (2025) generative models survey

### 3. Methods (\S3)
Detailed technical description:
- Framework architecture (AgenticProcessorBase)
- Autonomous continuation protocol
- Tool system (visual + search)
- Model adapters (OpenAI, HuggingFace)
- NOVA implementation specifics

### 4. Experiments (\S4)
- Experimental setup (5 configurations)
- Evaluation metrics (mAP@0.5, mAP@0.3, BERT-F1, METEOR, Clinical-F1)
- Results table with statistical significance (paired t-tests on per-case metrics)
- Ablation analysis
- Qualitative case studies

**Implementation notes:**
- Configuration is in frozen dataclasses (`src/radiant_harness/config.py`), not Hydra
- Prompt templates use minijinja, not Jinja2
- Experiment matrix defined in `examples/nova/experiments/config.py`
- Table generation via `examples/nova/experiments/aggregate.py`

### 5. Discussion (\S5)
- Key findings and implications
- Comparison with prior work
- Clinical implications
- Limitations
- Future directions

### 6. Conclusion (\S6)

## Key Contemporary Works for Comparison

> **Note:** The arXiv IDs, venue details, and impact factors below were generated as a starting point and may be inaccurate or outdated. Verify every citation against the live arXiv/journal before use.

### Must-Cite (Direct Competitors):

1. **CXR-Agent (Chambon et al., 2024)**
   - Most similar approach to yours
   - Tool-augmented radiology AI with uncertainty
   - Different: Task-specific, not a general framework
   - arXiv:2407.08811

2. **VoxelPrompt (Bar et al., 2024)**
   - Vision-language agent for medical images
   - Interactive visual exploration
   - Different: No external knowledge retrieval
   - arXiv:2410.08397

3. **AURA (Fathi et al., 2024)**
   - Multi-modal medical agent
   - Understanding, reasoning, annotation
   - Different: Less focus on iterative tool use
   - arXiv:2507.16940

4. **Agentic Systems in Radiology Survey (Blüethgen et al., 2025)**
   - Comprehensive survey just published
   - Establishes taxonomy and design principles
   - arXiv:2510.09404

### Important Context (Foundation Work):

5. **Med-Gemini (Saab et al., 2024; Yang et al., 2024)**
   - Current SOTA for medical VLMs
   - Provides baseline for comparison
   - arXiv:2404.18416, 2405.03162

6. **Chain-of-Thought Surveys**
   - Med-SCoT (Qiao et al., 2024)
   - ChestX-Reasoner (Fan et al., 2025)
   - M3CoTBench (Jiang et al., 2026) - important for benchmarking

### Additional Recent Work:

7. **Co-evolving Agentic AI (Li et al., 2025)**
   - Very recent arXiv paper
   - Similar space, good for comparison
   - arXiv:2509.20279

8. **AI Agents in Radiology Commentary (Koçak & Meşe, 2025)**
   - Recent commentary on agentic AI
   - Good for framing the field
   - Diagnostic and Interventional Radiology

## Recommended Target Venues

### High-Impact Medical Journals:
1. **Nature Medicine** (IF: 58.7) - Best fit for clinical impact
2. **npj Digital Medicine** (IF: 15.2) - Open access, strong AI focus
3. **Nature Communications** (IF: 14.7) - Broad scope, high visibility
4. **Medical Image Analysis** (IF: 10.9) - Technical focus
5. **IEEE TMI** (IF: 11.0) - Engineering perspective

### ML Conferences with Medical Tracks:
1. **ICML 2026** - Track on Healthcare/Medicine
2. **NeurIPS 2025/26** - Medical AI workshop
3. **ICLR 2026** - Growing medical AI presence
4. **ML4H** (Machine Learning for Healthcare) - Specialized venue
5. **MICCAI 2025/26** - Medical imaging conference

## Next Steps

1. **Fill in experimental results** - Add your actual numbers from NOVA runs
2. **Add figures** - Create ablation plots, qualitative examples
3. **Expand related work** - Read and cite the specific papers identified
4. **Add implementation details** - Hyperparameters, compute resources, API costs
5. **Clinical validation** - Consider radiologist reader study
6. **Ablation studies** - More detailed analysis of component contributions

## Bibliography Notes

The `references.bib` file contains 30+ citations covering:
- Foundation models (Med-Gemini, GPT-4V)
- Agentic systems (CXR-Agent, VoxelPrompt, AURA)
- Reasoning (CoT variants, benchmarks)
- Medical VQA datasets and methods
- Tool learning (Toolformer, MRKL)
- Clinical context (diagnostic error, rare diseases)

All citations use proper BibTeX format with arXiv IDs where applicable.
