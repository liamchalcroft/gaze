# MedMarks Integration Guide

This document describes how to integrate Radiant Harness with the [MedMarks](https://medmarks.ai) medical LLM evaluation platform.

## Overview

MedMarks is an open-source automated evaluation suite for medical LLMs, developed by Sophont, MedARC, and Prime Intellect. Radiant Harness provides a MedMarks-compatible environment package for the NOVA brain-MRI benchmark.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MedMarks Platform                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ medmarks.ai  │  │ Prime Hub    │  │ medarc-eval CLI  │  │
│  │ Leaderboard  │  │ Dashboard    │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Verifiers Framework                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ vf.MultiTurnEnv → vf.evaluate() → vf.Rubric         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             NOVA Brain MRI Environment                       │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ NOVABrainMRIEnv│  │ Reward Functions│  │ CLI Tools   │  │
│  │ (MultiTurnEnv) │  │ (Caption,Diag,  │  │ (medarc-eval│  │
│  │                │  │  Localization)  │  │  compatible)│  │
│  └────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Radiant Harness                            │
│  ┌────────────────────┐  ┌─────────────────────────────┐   │
│  │AgenticProcessorBase│  │ VerifiableProcessorMixin    │   │
│  │ + ToolRegistry     │  │ + BaseMultiTurnEnv          │   │
│  │ + OpenAI Adapter   │  │ + RadiantHarnessAdapter     │   │
│  └────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install radiant-harness with MedMarks support
pip install radiant-harness[medmarks]

# Install the NOVA brain-MRI environment
cd environments/nova_brain_mri
pip install -e .
```

### Running Evaluations

#### Via medarc-eval CLI

```bash
# Evaluate GPT-4o on NOVA benchmark
medarc-eval nova-brain-mri -m gpt-4o -n 100

# Evaluate with specific task
medarc-eval nova-brain-mri -m gpt-4o --task diagnosis -n 50

# Full evaluation with tools
medarc-eval nova-brain-mri -m gpt-4o --use-tools --max-turns 10
```

#### Via Python API

```python
import verifiers as vf
from openai import OpenAI

# Load environment
env = vf.load_environment(
    "nova-brain-mri",
    split="test",
    task="all",
    max_turns=10,
    use_tools=True,
)

# Create client
client = OpenAI()

# Run evaluation
results = env.evaluate(
    client=client,
    model="gpt-4o",
    num_examples=100,
)

print(f"Mean reward: {results.mean_reward:.3f}")
```

#### Using Radiant Harness Directly

```python
from examples.nova.src.processor import NOVAAgenticProcessor

# Create processor with full agentic capabilities
processor = NOVAAgenticProcessor(
    model_name="openai/gpt-4o",
    use_tools=True,
    use_web_search=True,
    max_turns=10,
    task="all",
)

# Create verifiers environment
EnvClass = processor.as_verifiers_env(
    dataset_path="data/nova_test.jsonl",
    image_base_path=Path("data/images"),
)
env = EnvClass()

# Use with verifiers
import verifiers as vf
results = vf.evaluate(env, model="gpt-4o")
```

## Environment Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | "test" | Dataset split: "train", "validation", "test" |
| `task` | str | "all" | Task: "caption", "diagnosis", "localization", "all" |
| `max_turns` | int | 10 | Maximum conversation turns |
| `use_tools` | bool | True | Enable visual manipulation tools |
| `use_web_search` | bool | False | Enable PubMed literature search |
| `iou_threshold` | float | 0.3 | IoU threshold for localization |
| `data_dir` | str | None | Custom dataset directory |

### Environment Metadata

The environment is registered with verifiers using:

```toml
[tool.verifiers.environment]
loader = "nova_brain_mri:load"
display_name = "NOVA Brain MRI"
visibility = "PUBLIC"
modality = "vision"
domain = "radiology"
tasks = ["caption", "diagnosis", "localization"]
```

## Reward Functions

### Caption Reward

Token-level F1 score between predicted and reference captions:

```python
from nova_brain_mri.rewards import caption_reward

reward = caption_reward(prompt, completion, info)
# Returns: float in [0.0, 1.0]
```

### Diagnosis Reward

Top-k accuracy with medical term normalization:

```python
from nova_brain_mri.rewards import diagnosis_reward

reward = diagnosis_reward(prompt, completion, info)
# Returns: 0.6 * top1_match + 0.4 * coverage
```

### Localization Reward

IoU-based detection F1 score:

```python
from nova_brain_mri.rewards import localization_reward_factory

reward_fn = localization_reward_factory(iou_threshold=0.3)
reward = reward_fn(prompt, completion, info)
# Returns: Detection F1 in [0.0, 1.0]
```

### Combined Rubric

```python
from nova_brain_mri.rewards import create_nova_rubric

rubric = create_nova_rubric(task="all", iou_threshold=0.3)
# Returns weighted combination: 33% caption, 34% diagnosis, 33% localization
```

## Submitting to MedMarks Leaderboard

### 1. Run Evaluation

```bash
medarc-eval nova-brain-mri -m your-model -o results.json
```

### 2. Format Results

```python
import json

with open("results.json") as f:
    results = json.load(f)

submission = {
    "model": results["model"],
    "environment": "nova-brain-mri",
    "mean_reward": results["mean_reward"],
    "task_scores": {
        "caption": results.get("caption_score"),
        "diagnosis": results.get("diagnosis_score"),
        "localization": results.get("localization_score"),
    },
    "configuration": {
        "max_turns": 10,
        "use_tools": True,
    }
}
```

### 3. Submit via MedMarks

Contact the MedMarks team via:
- Email: (see medmarks.ai)
- Discord: MedARC community

## Creating Custom Environments

You can create additional MedMarks-compatible environments using Radiant Harness:

```python
from radiant_harness import AgenticProcessorBase
from radiant_harness.verifiers import VerifiableProcessorMixin, BaseRewardFunction

class MyMedicalProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    """Custom processor for your medical imaging task."""

    def get_system_prompt(self, images, metadata):
        return "You are a medical imaging expert..."

    def get_user_message(self, images, metadata):
        return "Analyze this medical image..."

    def get_response_schema(self):
        return {"type": "json_schema", ...}

    def validate_response(self, response):
        return "diagnosis" in response

    def get_reward_function(self) -> BaseRewardFunction:
        return MyCustomReward()

# Create MedMarks-compatible environment
EnvClass = MyMedicalProcessor.as_verifiers_env(
    max_turns=10,
    dataset_path="my_dataset.jsonl",
)
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   ```bash
   pip install radiant-harness[medmarks]
   ```

2. **Dataset not found**
   ```bash
   # Ensure data directory is set
   medarc-eval nova-brain-mri --data-dir /path/to/data
   ```

3. **API key issues**
   ```bash
   export OPENAI_API_KEY=sk-...
   # or
   export OPENROUTER_API_KEY=sk-...
   ```

4. **Tool execution errors**
   ```python
   # Disable tools if they cause issues
   env = load(use_tools=False)
   ```

### Debug Mode

```bash
# Enable verbose logging
medarc-eval nova-brain-mri -m gpt-4o -v

# Stream results
medarc-eval nova-brain-mri -m gpt-4o -s
```

## Resources

- [MedMarks Leaderboard](https://medmarks.ai)
- [Prime Intellect Hub](https://app.primeintellect.ai)
- [MedARC GitHub](https://github.com/MedARC-AI/med-lm-envs)
- [Verifiers Documentation](https://github.com/primeintellect-ai/verifiers)
- [Radiant Harness README](../README.md)
