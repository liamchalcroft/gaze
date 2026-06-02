# AgentClinic NEJM Example

Multi-turn diagnostic reasoning with standardized patient simulation.

## Overview

This example implements the AgentClinic NEJM environment using the [verifiers](https://github.com/primeintellect-ai/verifiers) package for multi-turn clinical case evaluation. The assistant acts as a clinician, gathering information from a simulated patient before making a final diagnosis.

Based on: [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic)

## Task Description

The assistant must:
1. **Gather information** by requesting HISTORY, EXAM_AND_TESTS, or IMAGE
2. **Analyze** the patient case and medical images
3. **Diagnose** by providing the correct answer inside braces, e.g. `{Pneumonia}`

## Dataset

- **Source**: New England Journal of Medicine (NEJM) clinical cases
- **Format**: Extended JSONL with patient information, images, and answer choices
- **Download**: Use the provided download script

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download Dataset

```bash
cd data
python download.py
```

### 3. Load Environment

```python
from examples.agentclinic_nejm.src import load_environment

env = load_environment(
    dataset_path="./data/agentclinic_nejm_extended.jsonl",
    max_turns=10,
)

print(f"Loaded {len(env.dataset)} clinical cases")
```

### 4. Evaluate

```bash
uv run python -m examples.agentclinic_nejm.eval \
    --dataset ./data/agentclinic_nejm_extended.jsonl \
    --model qwen3.5-a3b \
    --base-url http://localhost:1234/v1 \
    --num-samples 10 \
    --output ./results
```

## Interaction Format

```
System: Think step by step. When ready, output ONE diagnosis inside braces.
User: [Case description with chief complaint and answer choices]

Assistant: I need to gather more information. What is the patient's medical history?
HISTORY

Environment: Patient History
[Patient medical history details]

Assistant: What do the physical examinations and test results show?
EXAM_AND_TESTS

Environment: Physical Examination and Test Results
[Exam results and clinical findings]

Assistant: Let me see the medical image if available.
IMAGE

Environment: [Image or 'No medical image is available for this case.']

Assistant: Based on the patient's presentation with [symptoms] and [findings],
{Pneumonia}
```

## Completion Criteria

An episode completes only when:
1. **Information Request**: Assistant has made at least one information request (asked=True)
2. **Diagnosis Format**: Final answer is wrapped in braces, e.g. `{Pneumonia}`

If the model outputs a diagnosis without first requesting information, the environment
sends a nudge and the episode continues.

## Reward Function

Combined reward: 0.8 * accuracy + 0.2 * token-F1.

Accuracy reward:
- **1.0**: Normalized prediction matches gold answer
- **0.0**: Incorrect or missing diagnosis

Normalization: lowercase, strip braces/punctuation, match against answer options.

## File Structure

```
agentclinic_nejm/
  src/
    __init__.py              # Package exports
    environment.py           # MultiTurnEnv implementation
  data/
    download.py              # Dataset download script
    agentclinic_nejm_extended.jsonl  # Dataset file
  train.py                   # Training integration template
  eval.py                    # Runnable evaluation loop
  README.md
```

## Debugging

The environment logs scoring details to `src/log/debug.log`:
```
[score] raw_gold='Exogenous ochronosis' raw_pred='Exogenous ochronosis'
norm_gold='exogenous ochronosis' norm_pred='exogenous ochronosis' ok=True
```

## References

- [AgentClinic Repository](https://github.com/SamuelSchmidgall/AgentClinic)
- [NEJM Clinical Cases](https://www.nejm.org/medical-education/clinical-cases)
- [verifiers Package](https://github.com/primeintellect-ai/verifiers)

## License

This example follows the license terms of the AgentClinic project and NEJM clinical cases for educational use.
