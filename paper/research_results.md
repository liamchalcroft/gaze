
Deep literature research on agentic multi-turn AI systems for medical image analysis
Scope and framing
This review focuses on agentic (multi-step, multi-turn) AI systems for medical image analysis that combine (i) a reasoning loop (planning + reflection + memory) with (ii) tool use (calling specialised models/APIs) to produce grounded outputs such as localisations, measurements, reports, and visual evidence. The emphasis is on 2023–2025 work, plus a small number of early-2026 papers where they directly operationalise benchmarks requested in the prompt (for example, M3CoTBench). 

Key background observations motivating agentic design recur across the literature:

Single-pass medical VLMs struggle with multi-step clinical reasoning and error modes (hallucinations/inconsistencies), which makes a “plan → act → verify” loop attractive for safety-critical use. 
Radiology workflows naturally decompose into checklists, sub-tasks, and measurements (for example ABCDE-style inspection for CXR), aligning well with tool orchestration across modules (segmentation, grounding, classification, report drafting, quality control). 
Open-world / rare-condition stress testing is becoming a prominent evaluation theme (e.g., NOVA), which is highly relevant to Radiant Harness’ rare-condition NOVA evaluation intent. 
For repositories and indexing (as per the prompt), the core sources used here were arXiv, PubMed, PhysioNet, and OpenReview, alongside conference/journal portals and publisher sites. 

Competitive landscape of agentic medical imaging systems
What counts as a “direct competitor” to Radiant Harness?
Direct competitors are systems that (i) accept medical images + natural language, (ii) perform multi-step tool use (not just a single model forward pass), and (iii) expose interactive reasoning (multi-turn dialogues, tool traces, intermediate evidence, or structured workflows). This set includes CXR-focused agents and broader multimodal agents that are still imaging-centric.

Comparison matrix across key systems
System	Primary modalities	Tooling pattern	External retrieval / knowledge	Autonomy level	Main evaluations reported
CXR-Agent	CXR	Uses CheXagent probes + BioViL-T phrase grounding to drive uncertainty-aware report generation	Limited; evaluation references Med-Gemini rubric, not a retrieval engine	Agentic workflow for reporting; emphasis on prompt constraints to reduce hallucination	NLP metrics + CXR benchmarks + clinical user study with respiratory specialists 
MedRAX	CXR	ReAct-driven orchestration of multiple off-the-shelf CXR tools (segmentation, grounding, classification, report gen, generation utilities)	Not a core feature; focus is tool orchestration	High (multi-step planning, memory buffer, tool selection)	ChestAgentBench (2,500 Qs) + CheXbench subsets (Rad-Restruct, SLAKE, OpenI reasoning) 
RadAgents	CXR	Multi-agent decomposition (ABCDE subagents) + tool suite + “skill layer” abstraction; also includes contrast adjustment preprocessing utilities	Visual retrieval-augmented conflict resolution (V-RAG) using similar CXRs + context	High; parallel specialist agents + synthesiser conflict resolution	ChestAgentBench + CheXbench; explicit ablations on V-RAG and model scaling 
Radiologist Copilot	3D CT (liver task shown)	Orchestrated tools: segmentation (e.g., TotalSegmentator), region analysis planning, template selection, report QC + refinement; built on OctoTools	Not front-and-centre; focuses on report QC	High; explicit planner/executor + tool memory	AMOS-MM liver report task; NLG metrics + clinical efficacy-style metrics (RadGraph F1, GREEN) 
CT-Agent	3D CT	Anatomy-aware “action space”: region-specific LoRA plugins + query normalisation + exemplar retrieval + token compression	Memory-based exemplar retrieval for report fluency and relevance	High; dynamic task type recognition + tool routing	CT-RATE + RadGenome-ChestCT; NLG metrics + “Clinical Efficacy” precision/recall/F1; QA P/R/F1 across organs 
VoxelPrompt	3D MRI/CT (neuroimaging sandbox)	Code-generating agent with persistent execution env; jointly trained vision encoder/generator called via executable instructions; computes measurements	Not emphasised; authors explicitly note future integration with databases/APIs	High; multi-step code + tool use inside persistent environment	Neuroimaging segmentation Dice comparisons and language-based pathology characterisation tasks; “data limitations” discussed 
AURA	CXR	ReAct-style loop + modular “expert toolkit” (VQA, grounded report gen, segmentation/detection, counterfactual editing, self-evaluation)	Not retrieval-focused; uses internal self-evaluation metrics	High; generate–test–select strategy with tool-as-critic	CheXpert held-out test set; counterfactual editing metrics (CPG, CFR, SSIM, SIP) vs RadEdit/PRISM baselines 
NOVA + RADAR (rare disease)	Brain MRI	RADAR adds retrieval-augmented reasoning agents for rare disease; NOVA is evaluation-only benchmark	Strong emphasis on external evidence retrieval (case reports + literature embeddings)	High; retrieval-augmented reasoning module	NOVA benchmark; reported gains up to +10.2% in rare pathology recognition for some models 

Interpretation for Radiant Harness positioning:

If Radiant Harness’ distinctive feature is radiologist-like interactive image manipulation tools (zoom/crop/contrast/threshold/flip/rotate) + external knowledge retrieval (PubMed + Open-i), then the closest “architecture cousins” are RadAgents (explicit “contrast adjustment” preprocessing utilities + V-RAG conflict resolution), VoxelPrompt (persistent execution environment and code-invoked operations), and RADAR (retrieval-augmented reasoning for rare diseases). 
Must-read paper summaries in the requested template
CXR-Agent: Vision-language models for chest X-ray interpretation with uncertainty aware radiology reporting (arXiv:2407.08811) 

Authors: Naman Sharma et al. 
Venue/Year: arXiv, 2024. 
URL: arXiv:2407.08811. 
Key Contribution: Builds an agent-based CXR reporting workflow using CheXagent linear probes (for pathology detection/classification bottleneck analysis) and BioViL-T phrase grounding for localisation, then propagates confidence into report text for “uncertainty-aware” reporting. 
Relation to Radiant Harness: Competitor + building block (agentic reporting + localisation as a tool; uncertainty-aware language). 
Relevant Findings:
Tools used: CheXagent probes; BioViL-T phrase grounding; LLM report generation component(s). 
Evaluation metrics: NLP metrics (e.g., ROUGE-L indicated), benchmark-based evaluation, plus clinical rubric-based comparison platform. 
Performance numbers: The thesis reports ROUGE-L comparisons and separations between normal vs abnormal scans, and notes differences across agent backbones; it also highlights that “dangerous report” rates vary by backbone model. 
Limitations noted: Emphasises overfitting risk in large VLMs and the need for larger paired datasets + augmentation; also discusses missing function calling as future work for more versatile tool selection. 
Citations Needed (claims worth citing in your paper):
That an agent workflow can reduce hallucinations by constraining tool outputs passed to an LLM. 
That phrase grounding (BioViL-T) can serve as a localisation tool in reporting workflows. 
That evaluation differs materially between normal vs abnormal scan subsets. 
VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis (arXiv:2410.08397) 

Authors: Andrew Hoopes et al. 
Venue/Year: arXiv 2024; revised version v2 in 2025. 
URL: arXiv:2410.08397. 
Key Contribution: A code-predicting, feedback-driven vision-language agent that operates in a persistent execution environment, calls jointly trained volumetric vision networks, and performs downstream computations (measurements, longitudinal change, ROI stats) in multi-step programmes. 
Relation to Radiant Harness: Building block / design pattern (execution environment, compositional tool graphs, explicit computations; future integration with APIs and medical databases is also discussed). 
Relevant Findings:
Tools used: “Executable instructions as code” + predefined library of functions, plus internal volumetric encoder/generator modules invoked by that code. 
Evaluation metrics: segmentation overlap via Dice; accuracy for pathology characterisation (exact match of natural language response) across defined tasks. 
Performance numbers: Reports outperformance counts (e.g., outperforms SynthSeg on 23/45 ROIs) but many numeric deltas are embedded in figure annotations in the PDF; the ar5iv rendering preserves the “23/45” statement. 
Limitations noted: “Data limitations” due to training-from-scratch restricting utility to the training domain; plans to expand training data and integrate pretrained language backbones and external systems/APIs. 
Citations Needed:
Persistent execution environment + multi-step code improves traceability for clinical workflows. 
The argument that grounding via explicit computations can be more reliable than purely textual prediction. 
AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation (arXiv:2507.16940) 

Authors: Nima Fathi et al. 
Venue/Year: arXiv 2025; comments indicate MICCAI submission context. 
URL: arXiv:2507.16940. 
Key Contribution: Open-source modular, ReAct-style imaging agent that integrates a toolbox of VQA, grounded report generation, segmentation/detection, and counterfactual editing, with a prominent self-evaluation (“generate–test–select”) loop. 
Relation to Radiant Harness: Competitor (framework-level); its modular tool suite and self-evaluation pattern are directly relevant to tool orchestration + safety. 
Relevant Findings:
Tools used: CheXagent VQA; MAIRA-2 grounded report generation; RadEdit + PRISM for counterfactuals; MedSAM + PSPNet + TorchXRayVision for segmentation/detection/classification; analysis via difference maps. 
External knowledge retrieval: Not a core emphasis; instead relies on self-evaluation metrics and tool outputs. 
Evaluation metrics: Defines SIP (L1 distance), CPG (classifier prediction gain), CFR (flip rate), SSIM; compares AURA to RadEdit and PRISM including ensemble baselines. 
Performance numbers: Table reports AURA achieves CPG 0.443, CFR 0.71, SSIM 0.740, SIP 0.060 (with #CFs=5) on CheXpert test split used by PRISM. 
Limitations noted (implicit): The paper frames evaluation around CXR and counterfactual explainability; broader modality coverage is not demonstrated in this short paper version. 
Citations Needed:
Self-evaluation loop definition and the specific CF metric suite. 
Evidence that agentic “generate–test–select” improves over fixed baselines. 
NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI (arXiv:2505.14064) 

Authors: Cosmin I. Bercea et al. 
Venue/Year: arXiv 2025; associated with NeurIPS 2025. 
URL: arXiv:2505.14064. 
Key Contribution: An evaluation-only benchmark of ~900 brain MRI cases spanning 281 rare pathologies, with rich clinical narratives and radiologist bounding-box annotations to jointly test anomaly localisation, visual captioning, and diagnostic reasoning under distribution shift. 
Relation to Radiant Harness: Core benchmark alignment (rare disease / open-world stress testing). 
Relevant Findings:
Tools used: NOVA is a benchmark, not a tool system. It supports evaluating localisation + reasoning. 
Evaluation metrics: Baseline comparisons across tasks; emphasises long-tailed distribution and OOD generalisation stress testing. 
Performance numbers: Baseline findings (performance drops) are described; detailed baselines are in the paper/slides, and the key dataset scale is 906 scans / 281 diagnoses. 
Limitations noted: Because it is evaluation-only and rare, models cannot train on it; its purpose is stress-testing rather than training. 
Citations Needed:
Dataset composition (906 scans, 281 rare diagnoses) and annotation scheme. 
Argument that many “OOD” medical benchmarks collapse into a closed-set problem without rare/novel categories. 
Med-Gemini key papers (two-paper set) 

Authors: Many-author consortium papers (Google/DeepMind-affiliated) in 2024. 
Venue/Year: arXiv 2024. 
URL: arXiv:2404.18416 and arXiv:2405.03162. 
Key Contribution: Establishes multimodal “Gemini” model capability claims in medicine and a dedicated medical adaptation; frequently used as a baseline point of comparison in later agentic imaging papers (e.g., CXR-Agent references their report evaluation rubric; RadAgents uses MedGemma and references large multimodal baselines). 
Relation to Radiant Harness: Competitive baseline & evaluation methodology influence (rubrics, multimodal competence framing). 
Relevant Findings: (details require full-paper reading; the key near-term utility is as a benchmark comparison and rubric source in other work). 
Citations Needed:
The rubric / evaluation processes adopted downstream (e.g., by CXR-Agent). 
Additional high-priority competitor / neighbour systems
MedRAX: Medical Reasoning Agent for Chest X-ray (arXiv:2502.02673) 

Authors: First author not reliably extractable from the HTML excerpt in this session; see arXiv:2502.02673 record (cited below via paper text). 
Venue/Year: arXiv 2025. 
URL: arXiv:2502.02673. 
Key Contribution: ReAct-based agent integrating a multi-tool CXR stack and introducing ChestAgentBench (2,500 six-choice questions) derived from 675 chest cases from Eurorad. 
Relation to Radiant Harness: Direct competitor on “tool-using CXR agent” design and on benchmarking multi-step reasoning. 
Relevant Findings:
Tools used: LLaVA-Med / CheXagent for VQA; MedSAM+PSPNet for segmentation; MAIRA-2 for grounding; TorchXRayVision DenseNet for classification; CheXpert Plus report generator; RoentGen for CXR synthesis; utilities for plotting/DICOM handling. 
Evaluation metrics: Accuracy on ChestAgentBench; accuracy on CheXbench subsets (Rad-Restruct, SLAKE, OpenI reasoning). 
Performance numbers: MedRAX reports overall 63% accuracy on ChestAgentBench vs GPT-4o 56.4% and CheXagent 39.5%; and 68.1 overall on CheXbench subsets with strong VQA performance (e.g., 68.7 on Rad-Restruct; 82.9 on SLAKE). 
Limitations noted: Can struggle resolving contradictory tool outputs; computational overhead; and explicitly states lack of robust uncertainty quantification mechanisms. 
Citations Needed:
ChestAgentBench construction pipeline and dataset provenance (Eurorad/ESR). 
The “limitations” paragraph (contradictory tool outputs + no UQ). 
RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows (arXiv:2509.20490) 

Authors: Not fully extractable from excerpt in this session; see arXiv:2509.20490 record. 
Venue/Year: arXiv 2025. 
URL: arXiv:2509.20490. 
Key Contribution: A multi-agent CXR system that encodes a radiologist-like workflow (ABCDE subagents) and integrates a broad toolset, including contrast adjustment preprocessing utilities and visual retrieval for conflict resolution. 
Relation to Radiant Harness: Direct competitor (workflow + tools) and particularly relevant for your “toolbox + retrieval” pitch. 
Relevant Findings:
Tools used: ROI segmentation (CXAS, BiomedParser), phrase grounding (MAIRA-2), measurements (Python scripts), VQA (MedGemma + CheXagent), report generation (CheXpert Plus report generator + MAIRA-2), pathology classification (TorchXRayVision DenseNet), plus data utilities including DICOM loader and contrast adjustment. 
External knowledge retrieval: Implements V-RAG retrieving similar chest radiographs using embeddings (Rad-DINO) with associated context to adjudicate tool conflicts. 
Evaluation metrics: Accuracy (%) on ChestAgentBench categories and CheXbench (Rad-Restruct, SLAKE, OpenI reasoning). 
Performance numbers: Reports overall 73.6% on ChestAgentBench and 74.6% overall on CheXbench, outperforming baselines; ablations attribute ~6–7 points overall to V-RAG. 
Limitations/notes: Highlights maintainability pattern (skill layer) and shows conflict resolution is sensitive to model capacity (synthesiser scaling). 
Citations Needed:
Evidence that V-RAG improves conflict resolution and accuracy. 
Skill layer abstraction to decouple tools from workflows. 
Radiologist Copilot: An Agentic Assistant with Orchestrated Tools for Radiology Reporting with Quality Control (arXiv:2512.02814) 

Authors: Yongrui Yu et al. 
Venue/Year: arXiv 2025. 
URL: arXiv:2512.02814. 
Key Contribution: Agentic assistant for volumetric report generation that explicitly adds a quality control stage (assessment + feedback-driven refinement) and orchestrates tools for region localisation and region-level analysis planning. 
Relation to Radiant Harness: Competitor on infrastructure patterns (planner/executor, tool memory, QC stage). 
Relevant Findings:
Tools used: Segmentator (e.g., TotalSegmentator), Analyzer (Think with Image paradigm + region analysis planning items), Report generator (strategic template selection), Quality controller (assessment + refinement). 
Datasets/benchmarks: AMOS-MM liver radiology reporting subset; uses BioBERT embeddings to cluster template reports. 
Metrics: BLEU-1, ROUGE-L, METEOR, BERTScore; plus F1-RadGraph and GREEN. 
Performance numbers: Reports (example) BLEU-1 0.4025, ROUGE-L 0.3222, METEOR 0.4560, BERTScore 0.7024, F1-RadGraph 0.2585, GREEN 0.4379 for the full system, with ablations showing drops without region analysis planning (RAP) and template selection (STS). 
Citations Needed:
That report generation is only one phase; QC is essential yet neglected. 
Ablation evidence for RAP/STS/QC components. 
CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering (arXiv:2505.16229) 

Authors: Yuren Mao et al. 
Venue/Year: arXiv 2025. 
URL: arXiv:2505.16229. 
Key Contribution: Anatomy-aware planning and tool routing for CTQA using region-specific LoRA “plugins”, plus global-local token compression to reduce visual tokens (~75% reduction claim). 
Relation to Radiant Harness: Different modality competitor (3D CT), but strong design patterns for region routing + memory-based exemplars. 
Relevant Findings:
Tools used: Anatomy-specific LoRA modules as tools; query rewriting/normalisation; prediction-guided exemplar retrieval; token compression; memory module storing exemplars. 
Metrics: NLG metrics (BLEU/ROUGE-L/METEOR) plus clinical efficacy P/R/F1; QA P/R/F1 by anatomical region. 
Numbers: Notes CE-F1 gain of 0.199 in report generation vs baseline and average F1 gains in QA; shows token-compression ablations and exemplar retrieval ablations with concrete NLG metrics. 
Citations Needed:
That “anatomical confusion” is reduced by region-guided planning. 
Exemplar retrieval improving report metrics. 
Technical foundations underpinning tool-using medical imaging agents
Reasoning loops and “step-by-step verification”
Most of the direct competitors implement a variant of a “reasoning-and-acting” loop, even if they differ in how explicit they are:

MedRAX explicitly describes a ReAct loop that cycles through observation → thought → action (tool call) with a memory buffer. 
AURA is framed as a ReAct-style reasoning loop “powered by a code-instructed LLM” and uses programme-like function calls via an agent framework. 
VoxelPrompt is similar in spirit but trains a language agent to output executable instructions and interpret intermediate results in a persistent environment. 
Radiologist Copilot splits planning/execution explicitly into Action Planner + Action Executor, plus QC loops (feedback-driven refinement). 
RadAgents formalises the “workflow” level (ABCDE subagents) and further layers an abstraction (skills) between agent prompts and raw tools to improve maintainability. 
From a “Radiant Harness” perspective, the take-home is that tool use still benefits from workflow constraints and verification steps rather than unconstrained autonomy, because tool disagreement is common and resolving it is a distinct capability (RadAgents emphasises a synthesiser + retrieval for conflict resolution; MedRAX flags contradictory tool outputs as a limitation). 

Chain-of-thought in medical imaging: structured vs free-form
Three strands appear in recent (2024–2026) work:

Structured chain-of-thought for interpretability: Med-SCoT proposes a “structured chain-of-thought” approach for medical VQA, aiming to improve interpretability and reliability. 
Process supervision mined from clinical reports: ChestX-Reasoner trains a CXR diagnosis MLLM by mining reasoning chains from routine radiology reports, and introduces RadRBench-CXR plus RadRScore to evaluate factuality/completeness of reasoning steps. 
Benchmarks explicitly measuring reasoning quality: M3CoTBench (ICLR 2026 submission) proposes CoT-specific metrics (correctness, efficiency, impact, consistency) over a multi-modality dataset, addressing the gap where traditional benchmarks only measure final answer accuracy. 
A direct implication for Radiant Harness’ evaluation design:

If you provide tool traces or explanations, you can evaluate them beyond “answer correctness” using reasoning fidelity metrics similar to RadRScore (ChestX-Reasoner) or CoT-specific categories (M3CoTBench). 
Clinician preference tends to be driven by auditability and alignment to real report logic, which motivates structured traces rather than unconstrained “free-form CoT” output (ChestX-Reasoner’s process-supervision framing). 
Tool learning and function calling patterns
Across systems, two integration paradigms dominate:

Training-free orchestration of pretrained tools: MedRAX and AURA both emphasise operating as inference agents “leveraging off-the-shelf tools” rather than end-to-end retraining; similarly Radiologist Copilot states a training-free orchestration over pretrained models via OctoTools. 
Joint training of agent + vision modules: VoxelPrompt trains a unified agent/vision framework, enabling flexible ROI definitions and compositional workflows without enumerating a static tool list. 
Function-calling mechanisms matter operationally. In CXR-Agent, function calling is explicitly flagged as future work to make tool selection more versatile. 
 In MedRAX, tool execution is described as structured JSON API calls. 
 In AURA, the system is described as using programmatic Python function calls via an agent framework. 

Uncertainty quantification and hallucination control
Uncertainty and hallucination are addressed via three complementary levers:

Uncertainty-aware language generation: CXR-Agent explicitly focuses on “uncertainty-aware radiology reporting” and propagating confidence and localisation likelihoods into the final report. 
Framework-level acknowledgement of missing UQ: MedRAX explicitly lists “lacks robust uncertainty quantification mechanisms” as a limitation. 
Post-hoc hallucination detection / filtering: VisionSemanticEntropy (DSE) proposes hallucination detection in black-box VLMs via semantic inconsistency quantification, framed as a filtering strategy for clinical VLM applications. 
Grounding-based fact checking: A phrase-grounded fact-checking model is proposed to detect errors in findings and their locations in generated CXR reports. 
For uncertainty calibration more generally, recent work also targets confidence calibration in medical VQA, reporting reductions in expected calibration error (ECE) across multiple datasets. 
 This suggests that Radiant Harness can justify a calibration component (or an abstention/deferral policy) as an evidence-based safety mechanism.

External knowledge retrieval and similar-case search
Radiant Harness’ planned retrieval from PubMed + Open-i aligns with two established needs:

Grounding outputs in external evidence to reduce hallucinations and improve the clinical plausibility of reasoning.
Supporting rare, long-tail conditions where model priors are weak and clinicians consult literature/case reports.
Radiology-specific RAG and retrieval-in-the-loop studies
There is a growing radiology literature framing RAG as a method to ground LLM outputs, including a Radiology: AI overview of RAG applications in radiology. 

Empirical studies include:

RAG improving locally deployable LLM quality for radiology contrast media consultation (privacy-motivated local deployment + retrieval). 
A proof-of-concept showing RAG-enhanced GPT-4 improving precision/trust when diagnosing and classifying traumatic injuries from radiology reports. 
Radiology report generation methods using retrieval (e.g., ML4HC 2023 RAG for report writing, and LaB-RAG using label-boosted retrieval). 
A key caution is that retrieval quality and source correctness matter; “retrieved content is not guaranteed to be correct,” and effectiveness depends on coverage and quality. 

Rare diseases and literature-based retrieval agents
For rare disease imaging, RADAR (Retrieval Augmented Diagnostic Reasoning Agents) explicitly models how radiologists consult case reports and literature, embedding case reports/literature and retrieving relevant evidence for unseen diseases, with reported performance gains on NOVA. 

This is directly aligned with Radiant Harness’ emphasis on NOVA and rare conditions: it provides a literature-supported rationale for why retrieval helps and an example of how to operationalise it.

Open-i as an external retrieval target
Open-i is an NLM image search engine with radiology-specific subsets (including Indiana chest X-rays + reports) and larger image-scale access to figures from PubMed Central articles. 
 The Indiana chest X-ray collection’s availability for search/download through NLM is described in Demner-Fushman’s work on preparing radiology examinations for retrieval. 

This supports Radiant Harness’ use of Open-i both as:

“similar case” retrieval (image + report), and
a gateway to the biomedical literature image corpus.
Visual tools for medical imaging and grounding operations
Radiant Harness’ tool list (zoom/crop/contrast/threshold/flip/rotate) is closer to viewer-level interaction than to typical “AI tools” (segmentation/classification). The literature shows two relevant trends:

Agent tool sets increasingly include basic preprocessing utilities, e.g., RadAgents includes data processing utilities such as a DICOM loader and contrast adjustment in its tool suite. 
Region grounding is becoming central: multiple systems treat segmentation/phrase grounding as “evidence primitives” that can be shown to clinicians (CXR-Agent uses BioViL-T phrase grounding for localisation; RadAgents and AURA incorporate grounded report generation and phrase grounding; Radiologist Copilot uses segmentation masks to propose region analysis items). 
From a radiologist tooling perspective, common DICOM viewers explicitly advertise tools like zoom/pan and brightness/contrast adjustment, reinforcing that these operations match real viewing habits. For example, IMAIOS’ viewer lists brightness/contrast adjustment and zoom/pan, and RadiAnt DICOM Viewer lists brightness/contrast and zoom/pan as basic manipulation tools. 

How competitors evaluate whether visual tools help is still underdeveloped; most papers evaluate final-task outcomes (accuracy, report metrics) rather than “tool utility”. Radiologist Copilot begins to separate process quality (analysis process, tool selection, action planning, action execution) as evaluation axes, which is a useful template for your system’s user studies. 

Benchmarks, datasets, and evaluation metrics
Benchmarks for agentic reasoning
Two benchmarks are particularly central for CXR agents:

ChestAgentBench: Introduced with MedRAX, containing 2,500 six-choice questions across seven categories, built from 675 expert-curated chest cases; explicitly designed to test complex multi-step reasoning. 
CheXbench: Used as a broader evaluation axis; MedRAX and RadAgents report results on VQA subsets and OpenI image-text reasoning tasks. 
For tissue-specific or 3D tasks:

NOVA: evaluation-only stress test for rare brain MRI pathologies (906 scans; 281 diagnoses). 
CT-RATE and RadGenome-ChestCT: used by CT-Agent for 3D CT question answering and report generation. 
Medical VQA datasets referenced in agentic work
Several “classic” Med-VQA datasets remain in use:

VQA-RAD is described as a clinician-generated radiology VQA dataset in Scientific Data. 
SLAKE is used as part of CheXbench-based evaluation (e.g., MedRAX and RadAgents report SLAKE accuracy). 
Open-i is used as an “image-text reasoning” evaluation component in CheXbench subsets (MedRAX and RadAgents). 
Recent / larger-scale datasets also exist (e.g., 3D-RAD for 3D Med-VQA tasks), though they are not yet core to the agentic CXR papers. 

Radiology report datasets and grounded datasets
For report generation and linked tasks:

MIMIC-CXR v2.x is a large CXR dataset with DICOM images and free-text reports on PhysioNet (377,110 images; 227,835 studies). 
MIMIC-CXR-JPG provides derived JPG images and structured labels derived from reports. 
RadGenome-Chest CT is a grounded 3D CT dataset with organ-level masks and grounded reports/VQA pairs, published on arXiv and later in Scientific Data (2025). 
Evaluation metrics: from n-grams to clinically grounded metrics
A repeating theme is that standard text metrics (BLEU/ROUGE/METEOR) correlate imperfectly with clinical correctness, motivating radiology-specific metrics and human evaluation:

The “Evaluating progress…” work argues RadGraph F1 and RadCliQ correlate better with radiologist evaluations than prior metrics and analyses failure modes. 
GREEN proposes an LLM-based radiology report evaluation and error notation metric, emphasising clinically significant errors. 
RadEval provides a unified, open-source evaluation framework consolidating classic metrics and clinical concept–based and LLM-based evaluators (including GREEN), and extends support across modalities. 
RadGraph provides a dataset + extraction benchmark for entities and relations in radiology reports and is widely used as a “clinical correctness” scaffold. 
Radiology-Aware model-based evaluation (COMET-adapted) proposes another approach to aligning automated metrics with radiologist judgements. 
Agentic systems increasingly report both NLG and clinical-efficacy metrics in tandem. Radiologist Copilot, for example, reports BLEU/ROUGE/METEOR/BERTScore alongside F1-RadGraph and GREEN, and ablates its QC and planning components. 

Clinical validation, infrastructure, safety, and regulatory context
Human studies and clinical comparison work
Evidence of clinician-in-the-loop evaluation is growing but remains uneven:

CXR-Agent reports building a clinical evaluation platform and conducting a user study with respiratory specialists, emphasising safety and interpretability improvements and the importance of separating normal vs abnormal cases in evaluation. 
Flamingo-CXR (Nature Medicine) explicitly engages board-certified radiologists for expert evaluation, motivated by the difficulty of evaluating clinical quality with automatic metrics alone. 
Several papers and reviews warn that “automation bias” and silently plausible errors can alter clinical decision-making; this aligns with the need for explicit uncertainty and “deferral” behaviours in agentic systems. 
Deployment and infrastructure patterns
Infrastructure considerations are now regularly discussed in radiology AI deployment work:

MONAI is described as a PyTorch-based framework for medical imaging AI, including utilities intended to streamline development and deployment. 
Work on community-driven radiological AI deployment (including MONAI Deploy) frames the practical barriers between research and clinical use. 
Radiology workflow integration studies emphasise that standards-based interoperability reduces the operational burden of custom integrations. 
Practical deployment perspectives highlight regulatory and QA demands as part of real-world integration. 
For agentic imaging systems specifically, the arXiv “Agentic Systems in Radiology” survey is positioned to cover evaluation methods for planning/tool use as well as challenges such as error cascades and health IT integration. 
 Related review work in journals similarly frames agentic systems as LLMs embedded in frameworks for reasoning, planning, and action, with applications in radiology. 

Safety, ethics, and regulation relevant to autonomous tool-using systems
Regulatory guidance is evolving rapidly for AI-enabled device functions:

The U.S. Food and Drug Administration publishes guidance and pages on AI/ML software as a medical device, including Good Machine Learning Practice (GMLP) guiding principles (referencing IMDRF principles) and guidance on “predetermined change control plans” (PCCPs) for AI-enabled device software functions. 
The World Health Organization issued guidance on ethics and governance for large multimodal models in health in January 2024, providing governance recommendations relevant to multimodal agents. 
In the UK context, government response notes MHRA’s AI Airlock regulatory sandbox for AI as a medical device. 
For Radiant Harness, the most actionable “safety design patterns” supported by the literature include:

Traceability by design: log tool calls, intermediate evidence, and conflict-resolution steps (RadAgents trajectories; Radiologist Copilot’s planning/execution scoring). 
Uncertainty-aware outputs / calibration: explicit uncertainty mechanisms (CXR-Agent) and/or calibration metrics (ECE reductions) to decide when to abstain or defer. 
Grounding checks: phrase grounding and fact-checking for report hallucinations; hallucination filtering strategies for black-box VLMs. 
Retrieval with source governance: retrieval improves trust but is sensitive to coverage and correctness; conflict resolution via retrieval (RadAgents V-RAG; RADAR literature retrieval) should incorporate mechanisms for handling conflicting evidence. 
Copy-ready bibliography entries for key papers
bibtex
Copy
@article{sharma2024cxragent,
  title={CXR-Agent: Vision-language models for chest X-ray interpretation with uncertainty aware radiology reporting},
  author={Sharma, Naman},
  journal={arXiv preprint arXiv:2407.08811},
  year={2024}
}

@article{hoopes2024voxelprompt,
  title={VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis},
  author={Hoopes, Andrew and Dey, Neel and Butoi, Victor Ion and Guttag, John V. and Dalca, Adrian V.},
  journal={arXiv preprint arXiv:2410.08397},
  year={2024}
}

@article{fathi2025aura,
  title={AURA: A Multi-Modal Medical Agent for Understanding, Reasoning \& Annotation},
  author={Fathi, Nima and Kumar, Amar and Arbel, Tal},
  journal={arXiv preprint arXiv:2507.16940},
  year={2025}
}

@article{bercea2025nova,
  title={NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI},
  author={Bercea, Cosmin I. and Li, Jun and Raffler, Philipp and others},
  journal={arXiv preprint arXiv:2505.14064},
  year={2025}
}

@article{fan2025chestxreasoner,
  title={ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification},
  author={Fan, Ziqing and Liang, Cheng and Wu, Chaoyi and others},
  journal={arXiv preprint arXiv:2504.20930},
  year={2025}
}

@article{qiao2025medscot,
  title={Med-SCoT: Structured chain-of-thought reasoning and evaluation for enhancing interpretability in medical visual question answering},
  author={Qiao, Jinhao and others},
  journal={Computerized Medical Imaging and Graphics},
  volume={126},
  pages={102659},
  year={2025},
  doi={10.1016/j.compmedimag.2025.102659},
  pmid={41202542}
}

@article{yu2023radcliq,
  title={Evaluating progress in automatic chest X-ray radiology report generation},
  author={Yu, Felix and Endo, Mark and others},
  journal={Radiology: Artificial Intelligence},
  year={2023}
}

@article{ostmeier2024green,
  title={GREEN: Generative Radiology Report Evaluation and Error Notation},
  author={Ostmeier, Surafel and others},
  journal={Findings of EMNLP},
  year={2024}
}

@article{jiang2026m3cotbench,
  title={M3CoTBench: Benchmark Chain-of-Thought of MLLMs in Medical Image Understanding},
  author={Jiang, Juntao and Zhang, Jiangning and Bi, Yali and others},
  journal={arXiv preprint arXiv:2601.08758},
  year={2026}
}

@article{mao2025ctagent,
  title={CT-Agent: A Multimodal-LLM Agent for 3D CT Radiology Question Answering},
  author={Mao, Yuren and Xu, Wenyi and Qin, Yuyang and Gao, Yunjun},
  journal={arXiv preprint arXiv:2505.16229},
  year={2025}
}