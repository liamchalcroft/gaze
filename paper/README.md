# NOVA Experiments Paper

This directory contains a comprehensive LaTeX paper summarizing the NOVA dataset experiments.

## Generated Files

- **`nova_experiments.tex`** - Main LaTeX source file for the paper
- **`nova_experiments.pdf`** - Generated 5-page PDF (117KB)
- **`compile.sh`** - Compilation script for building the PDF

## Paper Overview

**Title:** Enhanced Vision-Language Model Evaluation on the NOVA Dataset: A Comprehensive Study of Retrieval-Augmented Approaches for Rare Brain Pathology Analysis

**Authors:** 
- Duaa Alim, Imperial College London
- Liam Chalcroft, University College London

**Content Sections:**
1. **Abstract** - Overview of the comprehensive evaluation framework
2. **Introduction** - Context and motivation for NOVA dataset evaluation
3. **Related Work** - Recent advances in medical AI and VLMs
4. **Methodology** - Detailed description of the six experimental approaches:
   - Enhanced Baseline
   - Retrieval-Augmented Generation (RAG)
   - Multi-turn Reasoning
   - Visual Operations
   - Web Search Integration
   - Comprehensive Combined Approach
5. **Technical Implementation** - System architecture and advanced features
6. **Expected Results** - Analysis framework and anticipated findings
7. **Conclusion** - Summary and future work
8. **References** - 8 relevant academic citations

## Key Features Covered

- **Dataset:** ~900 brain MRI scans spanning 281 rare pathologies
- **Tasks:** Anomaly localization, visual captioning, diagnostic reasoning
- **Models:** GPT-4o, Claude Sonnet, Gemini variants, LLaMA, Qwen2.5-VL, and others
- **Approaches:** Six enhanced methods for medical image analysis
- **Framework:** Comprehensive evaluation with multi-modal assessment

## Compilation

To regenerate the PDF:

```bash
# Make executable (if needed)
chmod +x compile.sh

# Compile the paper
./compile.sh

# Or manually:
pdflatex nova_experiments.tex
pdflatex nova_experiments.tex  # Run twice for references
```

## Requirements

- LaTeX distribution with LLNCS document class
- Standard packages: `fontenc`, `graphicx`, `amsmath`, `url`

The paper follows the LLNCS (Lecture Notes in Computer Science) format and includes proper academic citations and technical documentation of the experimental framework. 