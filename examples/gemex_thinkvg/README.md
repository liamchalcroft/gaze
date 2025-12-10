# GEMeX-ThinkVG Example

Visual grounding with chain-of-thought reasoning for chest X-ray analysis using verifiable rewards.

## Overview

This example demonstrates reinforcement learning fine-tuning using the Radiant Harness framework with the **verifiers** package for multi-rollout training. The GEMeX-ThinkVG task requires models to:

1. **Analyze** chest X-ray images with visual reasoning
2. **Use tools** for zoom, crop, contrast, threshold, and search
3. **Provide** structured responses with answer and bounding box location
4. **Receive** combined rewards from three verifiable components

## Dataset

- **Source**: [GEMeX-ThinkVG on HuggingFace](https://huggingface.co/datasets/BoKelvin/BoKelvin/GEMeX-ThinkVG)
- **Images**: Chest X-rays from MIMIC-CXR-JPG dataset
- **Annotations**: Findings with bounding boxes and anatomical locations
- **Format**: Chain-of-thought reasoning with visual grounding

## Reward Components

The verifiable reward system combines three components:

1. **Answer Reward** (`weights.answer`): Semantic matching of medical findings
   - Exact match, contains match, and token F1
   - Medical term normalization
   - Question type handling (open/closed)

2. **Location Reward** (`weights.location`): Anatomical region matching
   - Hierarchical matching (organ → subregion → specific)
   - Synonym normalization
   - Laterality awareness

3. **BBox Reward** (`weights.bbox`): Spatial accuracy with IoU
   - IoU and Generalized IoU
   - Center distance penalty
   - IoU@0.5 and IoU@0.3 thresholds

Default weights: `answer=0.4, location=0.3, bbox=0.3`

## Quick Start

### 1. Install Dependencies

```bash
# Core install (verifiers included)
uv sync
# or
pip install -e .

# Dataset helpers
pip install datasets huggingface-hub
```

### 2. Prepare Dataset

The dataset loader automatically handles MIMIC-CXR path resolution:

```python
from src import GEMeXDataset

dataset = GEMeXDataset(
    split="test",
    image_root="/path/to/mimic-cxr-jpg",
    cache_dir="./cache",
)
```

### 3. Load Environment

```python
from src import load_environment

env = load_environment(
    dataset_path="./data/test.jsonl",
    max_turns=8,
)
```

### 4. Train Model

```bash
python train.py \
    --dataset ./data/test.jsonl \
    --model gpt-4o \
    --learning-rate 1e-5 \
    --batch-size 8 \
    --epochs 3 \
    --reward-weights 0.4,0.3,0.3
```

### 5. Evaluate

```bash
python eval.py \
    --dataset ./data/test.jsonl \
    --model ./checkpoints/model \
    --output ./results
```

## Usage Examples

### Using the Processor Directly

```python
from src import GEMeXProcessor
from radiant_harness import HarnessConfig

# Configure processor
config = HarnessConfig(
    model_name="gpt-4o",
    max_turns=8,
    tools_enabled=True,
)

processor = GEMeXProcessor(config)

# Process a sample
image_path = "path/to/xray.jpg"
question = "What abnormality is present in the right lung?"

result = await processor.process(
    question=question,
    images=[image_path],
    question_type="open_ended",
)

print(f"Answer: {result.response['answer']}")
print(f"Location: {result.response['location']}")
```

### Computing Rewards

```python
from src import GEMeXRewardFunction, RewardWeights

# Initialize with custom weights
reward_fn = GEMeXRewardFunction(
    weights=RewardWeights(answer=0.5, location=0.25, bbox=0.25)
)

# Compute rewards
predictions = [
    {
        "answer": "pneumonia",
        "location": {"reference": "right lower lobe", "bbox": [120, 180, 220, 280]},
    }
]
references = [
    {
        "answer": "right lower lobe pneumonia",
        "location": {"reference": "right lung lower zone", "bbox": [110, 170, 230, 290]},
        "question_type": "open_ended",
    }
]

rewards = reward_fn(predictions, references)
print(f"Combined reward: {rewards[0]:.3f}")
```

## Multi-Turn Interaction

The environment supports tool requests during analysis:

```
Assistant: I need to examine the right lower lung more closely.
ZOOM [100, 150, 250, 300]

Environment: [ZOOM applied to region [100, 150, 250, 300]]
The zoomed region shows enhanced detail. Continue your analysis.

Assistant: I see an opacity in the right lower lobe. Let me check the contrast.
CONTRAST [1.5]

Environment: [CONTRAST adjusted by factor 1.5]
Image contrast enhanced. Continue your analysis.

Assistant: {
  "reasoning": "The zoomed and enhanced image shows a consolidative opacity...",
  "answer": "pneumonia",
  "location": {
    "reference": "right lower lobe",
    "bbox": [120, 180, 220, 280]
  }
}
```

## File Structure

```
gemex_thinkvg/
├── src/
│   ├── __init__.py              # Package exports
│   ├── processor.py             # GEMeXProcessor class
│   ├── dataset.py               # Dataset loader with MIMIC-CXR
│   ├── schemas.py               # ThinkVG XML/JSON schemas
│   ├── rewards/                 # Verifiable reward functions
│   │   ├── answer.py            # Answer semantic matching
│   │   ├── location.py          # Anatomical region matching
│   │   ├── bbox.py              # IoU-based bbox accuracy
│   │   └── combined.py          # Combined reward function
│   └── verifiers/               # verifiers package integration
│       └── environment.py       # MultiTurnEnv implementation
├── train.py                     # Training script
├── eval.py                      # Evaluation script
└── README.md                    # This file
```

## Integration with RL Training

This example integrates with the **verifiers** package for RL training:

```python
import verifiers as vf
from src import load_environment

# Load multi-turn environment
env = load_environment(dataset_path="gemex_train.jsonl")

# Use with verifiers training loop
trainer = vf.RLTrainer(
    environment=env,
    model="your-model",
    learning_rate=1e-5,
    # ...
)
trainer.train()
```

## References

- [GEMeX-ThinkVG Dataset](https://huggingface.co/datasets/BoKelvin/BoKelvin/GEMeX-ThinkVG)
- [MIMIC-CXR-JPG Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [Verifiers Documentation](https://docs.primeintellect.ai/verifiers)
- [Radiant Harness](https://github.com/anthropics/radiant-harness)

## License

This example follows the license terms of the Radiant Harness framework.