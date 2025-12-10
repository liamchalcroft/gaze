# AgentClinic NEJM Example

Multi-turn diagnostic reasoning with standardized patient simulation.

## Overview

This example implements the AgentClinic NEJM environment using the **verifiers** package for multi-turn clinical case evaluation. The assistant acts as a clinician, gathering information from a simulated patient before making a final diagnosis.

Based on: [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic)

## Task Description

The assistant must:
1. **Gather information** by requesting HISTORY, EXAM, TESTS, or IMAGE
2. **Analyze** the patient case and medical images
3. **Diagnose** by providing the correct answer in `{Diagnosis}` format
4. **Verify** diagnosis against gold standard answers

## Dataset

- **Source**: New England Journal of Medicine (NEJM) clinical cases
- **Format**: Extended JSONL with patient information, images, and answer choices
- **Download**: Use the provided download script

## Quick Start

### 1. Install Dependencies

```bash
# Core install (verifiers included)
uv sync
# or
pip install -e .

# Dataset helpers
pip install datasets
```

### 2. Download Dataset

```bash
cd data
python download.py
```

### 3. Load Environment

```python
from src import load_environment

env = load_environment(
    dataset_path="./data/agentclinic_nejm_extended.jsonl",
    max_turns=10,
)

print(f"Loaded {len(env.dataset)} clinical cases")
```

### 4. Train Model

```bash
python train.py \
    --dataset ./data/agentclinic_nejm_extended.jsonl \
    --model gpt-4o \
    --learning-rate 1e-5 \
    --batch-size 16 \
    --epochs 3 \
    --max-turns 10
```

### 5. Evaluate

```bash
python eval.py \
    --dataset ./data/agentclinic_nejm_extended.jsonl \
    --model ./checkpoints/model \
    --output ./results
```

## Interaction Format

The environment follows a specific interaction pattern:

```
System: Think step by step. When you are ready, output ONE diagnosis inside {Diagnosis}
User: [Case description with patient info and answer choices]

Assistant: I need to gather more information. What is the patient's medical history?
HISTORY

Environment: Patient History
[Patient medical history details]

Assistant: What do the physical examinations show?
EXAM

Environment: Physical Examination Findings
[Exam results and clinical findings]

Assistant: Are there any test results or imaging studies?
TESTS

Environment: Test Results
[Laboratory and imaging results]

Assistant: Let me see the medical image if available.
IMAGE

Environment: [Image URL or message]

Assistant: Based on the patient's presentation with [symptoms] and [findings],
{Pneumonia}
```

## Requirements for Completion

An episode completes only when:
1. **Information Request**: Assistant has made at least one information request (asked=True)
2. **Diagnosis Format**: Final answer is wrapped in `{Diagnosis}` braces
3. **Exact Match**: Normalized diagnosis matches gold answer

## Reward Function

The accuracy reward provides:
- **1.0**: Exact normalized match with gold answer
- **0.0**: Incorrect diagnosis or no diagnosis

Normalization includes:
- Lowercase conversion
- Removing braces and punctuation
- Handling multiple answer choices

## File Structure

```
agentclinic_nejm/
├── src/
│   ├── __init__.py              # Package exports
│   └── environment.py           # MultiTurnEnv implementation
├── data/
│   ├── download.py              # Dataset download script
│   └── agentclinic_nejm_extended.jsonl  # Dataset file
├── train.py                     # Training script
├── eval.py                      # Evaluation script
└── README.md                    # This file
```

## Example Cases

### Case 1: Dermatology
```
A 55-year-old woman presented with skin darkening...
Choices: [Contact dermatitis, Exogenous ochronosis]

Assistant: What medications is the patient using?
HISTORY

Environment: Patient History
The patient has been using hydroquinone-containing creams...

Assistant: {Exogenous ochronosis}
```

### Case 2: Infectious Disease
```
A 9-year-old boy from Brazil presented with respiratory symptoms...
Choices: [Actinomycosis, Paracoccidiomycosis]

Assistant: I need to see the imaging findings.
IMAGE

Environment: [Chest CT image URL]

Assistant: Are there laboratory results available?
TESTS

Environment: Test Results
Peripheral eosinophilia present...

Assistant: {Paracoccidiomycosis}
```

## Integration with RL Training

```python
import verifiers as vf
from src import load_environment

# Load multi-turn environment
env = load_environment(dataset_path="agentclinic_train.jsonl")

# Use with verifiers training loop
trainer = vf.RLTrainer(
    environment=env,
    model="your-model",
    learning_rate=1e-5,
    max_rollouts=4,
)
trainer.train()
```

## Evaluation Metrics

- **Accuracy**: Percentage of correct diagnoses
- **Mean Turns**: Average number of turns before diagnosis
- **Completion Rate**: Percentage of episodes reaching completion
- **Question Distribution**: Frequency of different information requests

## Debugging

The environment logs detailed information to `log/debug.log`:
```
[score] raw_gold='Exogenous ochronosis' raw_pred='Exogenous ochronosis'
norm_gold='exogenous ochronosis' norm_pred='exogenous ochronosis' ok=True
```

## References

- [AgentClinic Repository](https://github.com/SamuelSchmidgall/AgentClinic)
- [NEJM Clinical Cases](https://www.nejm.org/medical-education/clinical-cases)
- [verifiers Package](https://github.com/richzhang/verifiers)

## License

This example follows the license terms of the AgentClinic project and NEJM clinical cases for educational use.