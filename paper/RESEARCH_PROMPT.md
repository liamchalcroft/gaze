# Deep Literature Research Prompt: Radiant Harness

> **Staleness warning (2026-03):** This prompt was written as a one-time research
> scaffolding document. Some competitor references, arXiv IDs, and venue details
> may be outdated. Always verify citations against live sources before use.

## Research Objective

Gather comprehensive literature on agentic/multi-turn AI systems for medical image analysis, with focus on:
1. Direct competitors and similar frameworks
2. Technical foundations (tool use, CoT, reasoning)
3. Evaluation benchmarks and metrics
4. Clinical validation studies
5. Framework infrastructure and design patterns

---

## Search Strategy

For each section below, search using Google Scholar, PubMed, arXiv, and Semantic Scholar. Use exact phrase matching with quotes where indicated. Prioritize papers from 2023-2025.

---

## SECTION 1: Direct Competitors (High Priority)

These are systems most similar to Radiant Harness - agentic AI with tool use for medical imaging.

### Search Queries:
- `"medical agent" "tool use" radiology vision language`
- `"interactive" radiology AI zoom crop contrast tool`
- `"multi-turn" medical image analysis reasoning`
- `"agentic" radiology chest X-ray MRI CT`
- `"radiology copilot" OR "radiology assistant" autonomous`

### Key Papers to Find:
1. **CXR-Agent details** - Find the full paper at arXiv:2407.08811
   - What tools does it use?
   - How does it handle uncertainty?
   - What are its limitations vs. your approach?

2. **VoxelPrompt expansion** - arXiv:2410.08397
   - How does it ground visual analysis?
   - Does it support external knowledge?

3. **AURA details** - arXiv:2507.16940
   - What modalities does it support?
   - How is it different from a general framework?

4. **RadAgent, PathAgent, etc.** - Search for any other medical agent systems
   - Look for radiology-specific agents
   - Pathology agents (might have similar architecture)

5. **CheXagent Plus/Advanced** - Updates to CheXagent with agentic capabilities

### Questions to Answer:
- Which tools do each system provide?
- Do they support external knowledge retrieval?
- Is the continuation/reasoning fixed or autonomous?
- What are their evaluation benchmarks?

---

## SECTION 2: Technical Foundations

### 2.1 Chain-of-Thought in Medical AI

**Search Queries:**
- `"chain of thought" medical diagnosis radiology`
- `"structured chain of thought" Med-SCoT`
- `"reasoning" radiology report generation step by step`
- `"clinical reasoning" large language model`
- `"diagnostic reasoning" multimodal medical AI`

**Key Papers to Find:**
- Med-SCoT paper details and results
- ChestX-Reasoner methodology
- M3CoTBench dataset and metrics
- Any papers on "clinical chain-of-thought"
- TumorChain (mentioned in earlier search) - traceable clinical tumor analysis

**Questions:**
- How is CoT evaluated in medical contexts?
- What are the failure modes?
- Do clinicians prefer structured or free-form reasoning?

### 2.2 Tool Learning & Function Calling

**Search Queries:**
- `"tool use" vision language model medical`
- `"function calling" radiology AI`
- `"toolformer" medical application`
- `"API calling" clinical decision support`
- `"retrieval augmented generation" radiology`

**Key Papers:**
- Toolformer and its medical adaptations
- GPT-4V tool use capabilities
- Any papers specifically on medical tool learning
- Tool creation/enumeration for clinical AI

### 2.3 Uncertainty Quantification

**Search Queries:**
- `"uncertainty quantification" medical image analysis`
- `"calibration" radiology AI confidence`
- `"aleatoric uncertainty" OR "epistemic uncertainty" medical imaging`
- `"know when it doesn't know" radiology AI`

**Key Papers:**
- CXR-Agent uncertainty mechanisms
- Any papers on confidence calibration in medical VLMs
- Methods for detecting hallucinations in radiology reports

---

## SECTION 3: External Knowledge Retrieval

Your system integrates PubMed and Open-i search. Find papers on:

**Search Queries:**
- `"retrieval augmented generation" PubMed medical imaging`
- `"external knowledge" radiology diagnosis`
- `"literature search" AI radiology assistant`
- `"similar case retrieval" medical imaging`
- `"knowledge base" radiology AI integration`

**Key Papers:**
- Med-Gemini's web search capabilities
- Any systems using PubMed E-utilities
- Case-based reasoning in radiology AI
- Medical image retrieval systems (Open-i, similar)
- Knowledge graphs for radiology

**Questions:**
- How is retrieved knowledge formatted for LLMs?
- What are the latency implications?
- Does external knowledge actually improve accuracy?
- How to handle conflicting information from sources?

---

## SECTION 4: Visual Tools for Medical Imaging

Your system provides zoom, crop, contrast, threshold, flip, rotate. Find:

**Search Queries:**
- `"interactive visualization" radiology AI`
- `"windowing" OR "leveling" AI medical imaging`
- `"zoom" OR "crop" vision language model tool`
- `"image manipulation" radiology deep learning`
- `"attention guidance" medical imaging visualization`

**Key Papers:**
- VoxelPrompt visual grounding methods
- Any papers on saliency maps for radiology
- Interactive segmentation tools
- Multi-resolution analysis in medical imaging

**Questions:**
- Do other systems allow dynamic image manipulation?
- What tools are most used by radiologists?
- How to evaluate if visual tools help?

---

## SECTION 5: Benchmarks and Datasets

### 5.1 Medical VQA Benchmarks

**Search Queries:**
- `"VQA-RAD" OR "SLAKE" OR "PathVQA` medical visual question answering"
- `"Medical VQA" benchmark 2024 2025`
- `"radiology question answering` dataset"
- `"rare disease` medical imaging benchmark"

**Key Papers:**
- NOVA dataset details and statistics
- VQA-RAD, SLAKE, PathVQA papers
- New 2024/2025 Med-VQA datasets
- Zhang et al. 240k sample dataset (Nature Communications Medicine)

### 5.2 Radiology Report Datasets

**Search Queries:**
- `"MIMIC-CXR` OR `IU X-ray` OR `CheXpert` radiology report"
- `"radiology report generation` benchmark"
- `"brain MRI report` dataset"

**Key Papers:**
- MIMIC-CXR latest version
- Any brain MRI report datasets (might be limited)
- RadGenome, RadLLM datasets

### 5.3 Evaluation Metrics

**Search Queries:**
- `"mAP" medical imaging evaluation`
- `"METEOR" OR "ROUGE" radiology report`
- `"RadGraph" OR "RadCliN` evaluation"
- `"clinical efficacy` radiology AI evaluation"

**Key Papers:**
- RadGraph metric paper
- RadCliN evaluation framework
- Clinician evaluation studies
- Human-AI agreement metrics

---

## SECTION 6: Framework and Infrastructure Papers

Papers on building AI systems, not just models:

**Search Queries:**
- `"framework" medical AI vision language`
- `"infrastructure" radiology AI deployment`
- `"pipeline" multimodal medical AI`
- `"modular` OR `extensible` medical AI system"

**Key Papers:**
- Agentic Systems in Radiology survey (arXiv:2510.09404) - READ THIS COMPLETELY
- Any papers on MONAI, Medical Open Network for AI
- HuggingFace medical AI ecosystem papers
- Deploying medical AI in practice

---

## SECTION 7: Rare Disease and Edge Cases

Your NOVA evaluation focuses on rare conditions. Find:

**Search Queries:**
- `"rare disease" AI diagnosis medical imaging`
- `"long tail` distribution radiology AI"
- `"zero-shot` OR `few-shot` rare pathology detection"
- `"out-of-distribution` medical imaging diagnosis"

**Key Papers:**
- NOVA dataset paper details
- Any papers on rare pathology detection
- Long-tail learning in medical imaging
- Foundation model performance on rare conditions

---

## SECTION 8: Clinical Validation and Human Studies

**Search Queries:**
- `"radiologist` evaluation AI system multi-turn"
- `"clinician in the loop` AI radiology"
- `"human-AI collaboration` medical imaging"
- `"user study` radiology AI interface"

**Key Papers:**
- Any user studies with CXR-Agent, VoxelPrompt, etc.
- Radiologist workflow integration studies
- Time-to-diagnosis studies with AI assistance
- Clinical utility (not just accuracy) studies

---

## SECTION 9: Limitations, Safety, and Ethics

**Search Queries:**
- `"hallucination` radiology AI vision language"
- `"safety` medical AI autonomous system"
- `"bias` radiology AI fairness"
- `"regulatory` OR `FDA` medical AI agent"

**Key Papers:**
- Hallucination studies in medical VLMs
- Safety concerns for autonomous medical AI
- FDA/EMA guidance on AI agents vs. fixed models
- Bias in medical AI tools

---

## SECTION 10: Future Directions and Position Papers

**Search Queries:**
- `"future` medical AI 2025 2026"
- `"next generation` radiology AI vision language"
- `"grand challenges` medical imaging AI"
- `"perspective` OR "commentary" radiology AI future"

**Key Papers:**
- Recent perspective papers on AI in radiology
- Langlotz, Kohli, or other senior author perspectives
- RSNA/ACR position statements on AI

---

## Deliverables

For each paper found, document:

```markdown
### Paper Title
- **Authors:** 
- **Venue/Year:** 
- **URL:** 
- **Key Contribution:** 
- **Relation to Radiant Harness:** (competitor/building block/different approach)
- **Relevant Findings:**
  - Tools used:
  - Evaluation metrics:
  - Performance numbers:
  - Limitations noted:
- **Citations Needed:** (specific claims to cite)
```

---

## Priority Ranking

**MUST READ (Critical for Paper):**
1. CXR-Agent full paper
2. VoxelPrompt full paper
3. AURA full paper
4. Agentic Systems in Radiology survey
5. NOVA dataset paper
6. Med-Gemini papers (both)

**HIGH PRIORITY:**
7. Med-SCoT
8. ChestX-Reasoner
9. M3CoTBench
10. Any other medical agent papers found
11. Toolformer + medical adaptations

**MEDIUM PRIORITY:**
12. RadGraph evaluation paper
13. Recent Med-VQA surveys (2024-2025)
14. Rare disease detection papers
15. Uncertainty quantification in medical AI

**IF TIME PERMITS:**
16. Clinical validation studies
17. Safety/ethics papers
18. Future direction perspectives
19. Framework/infrastructure papers
20. Knowledge retrieval studies

---

## Search Tips

1. **Use Google Scholar's "Cited By"** - Find papers that cite CXR-Agent, VoxelPrompt
2. **Check arXiv categories** - cs.CV, cs.AI, cs.LG, eess.IV
3. **Semantic Scholar** - Good for finding related papers
4. **PubMed** - Search "artificial intelligence" + radiology + [specific topic]
5. **Follow author chains** - Search for other papers by Chambon, Bar, Fathi, etc.
6. **Check recent conferences** - MICCAI 2024, MIDL 2024, ICLR 2025 medical tracks
7. **Twitter/X** - Search recent tweets about medical AI agents

---

## Output Format

Create a markdown file organized by section with:
- Paper summaries
- Key quotes for potential inclusion
- Comparison matrix (tools, evaluation, performance)
- Bibliography entries ready to copy
- Notes on how each paper relates to your contribution
