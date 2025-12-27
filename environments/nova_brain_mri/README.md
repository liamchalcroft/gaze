# NOVA Brain MRI Environment for MedMarks

A MedMarks-compatible evaluation environment for the NOVA brain MRI benchmark. This environment enables multi-task vision-language model evaluation on radiological image analysis.

## Overview

The NOVA Brain MRI environment provides three evaluation tasks:

| Task | Description | Metric |
|------|-------------|--------|
| **Caption** | Radiological description with sequence characteristics | Token F1 |
| **Diagnosis** | Primary and differential diagnoses | Top-k Accuracy |
| **Localization** | Abnormality detection with bounding boxes | IoU-based mAP |

## Installation

```bash
# Install with pip
pip install nova-brain-mri

# Or install from source
cd environments/nova_brain_mri
pip install -e .
```

## Usage

### Via medarc-eval CLI

```bash
# Evaluate on all tasks
medarc-eval nova-brain-mri -m gpt-4o -n 100

# Evaluate specific task
medarc-eval nova-brain-mri -m gpt-4o --task diagnosis -n 50

# With tool usage enabled
medarc-eval nova-brain-mri -m gpt-4o --use-tools --max-turns 10
```

### Via Python API

```python
import verifiers as vf

# Load environment
env = vf.load_environment(
    "nova-brain-mri",
    split="test",
    task="all",
    max_turns=10,
    use_tools=True,
)

# Evaluate model
results = env.evaluate(
    client=openai_client,
    model="gpt-4o",
    num_examples=100,
)

# Access metrics
print(f"Mean reward: {results.mean_reward:.3f}")
print(f"Caption score: {results.metrics['caption']:.3f}")
print(f"Diagnosis score: {results.metrics['diagnosis']:.3f}")
print(f"Localization mAP: {results.metrics['localization']:.3f}")
```

### Integration with Radiant Harness

```python
from examples.nova.src.processor import NOVAAgenticProcessor

# Create processor with full tool support
processor = NOVAAgenticProcessor(
    model_name="openai/gpt-4o",
    use_tools=True,
    use_web_search=True,
    max_turns=10,
)

# Create verifiers environment
EnvClass = processor.as_verifiers_env(
    dataset_path="data/nova_test.jsonl",
)
env = EnvClass()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | "test" | Dataset split: "train", "validation", "test" |
| `task` | str | "all" | Task type: "caption", "diagnosis", "localization", "all" |
| `max_turns` | int | 10 | Maximum conversation turns |
| `use_tools` | bool | True | Enable visual tools (zoom, crop, contrast) |
| `use_web_search` | bool | False | Enable PubMed literature search |
| `iou_threshold` | float | 0.3 | IoU threshold for localization matching |
| `data_dir` | str | None | Custom data directory path |

## Response Schema

Models must return JSON with the following structure:

```json
{
  "caption": {
    "description": "Axial T2-weighted MRI showing...",
    "sequence_characteristics": "T2W",
    "orientation": "axial",
    "confidence": 0.85,
    "findings": ["hyperintense lesion", "periventricular location"]
  },
  "diagnosis": {
    "primary_diagnosis": "Multiple sclerosis",
    "differential_diagnoses": [
      {"diagnosis": "ADEM", "confidence": 0.3},
      {"diagnosis": "Neuromyelitis optica", "confidence": 0.2}
    ],
    "confidence": 0.75,
    "evidence": ["periventricular lesions", "Dawson's fingers pattern"]
  },
  "localization": {
    "localizations": [
      {
        "finding": "Demyelinating lesion",
        "bounding_box": [120, 80, 180, 140],
        "anatomical_location": "Right periventricular white matter",
        "confidence": 0.8
      }
    ],
    "image_dimensions": {"width": 512, "height": 512},
    "coordinate_system": "absolute_pixels"
  },
  "continue": false,
  "reasoning": "Based on the imaging characteristics..."
}
```

## Reward Functions

### Caption Reward
Token-level F1 score between predicted and reference captions.

### Diagnosis Reward
Combined score:
- 60% weight: Top-1 accuracy (primary diagnosis matches reference)
- 40% weight: Coverage (fraction of reference diagnoses matched)

Medical term normalization is applied before comparison.

### Localization Reward
Detection F1 score based on IoU matching:
- Predictions matched to ground truth using Hungarian algorithm
- IoU threshold determines positive matches (default: 0.3)
- F1 computed from precision and recall

## MedMarks Integration

This environment is designed for the [MedMarks](https://medmarks.ai) leaderboard, providing:

- Standardized evaluation protocol
- Comparable metrics across models
- Integration with Prime Intellect infrastructure
- Support for both API and open-source models

## Data Format

Each case in the dataset should be a JSON object with:

```json
{
  "image_path": "/path/to/brain_mri.png",
  "clinical_history": "45yo male with headaches and visual disturbance",
  "modality": "MRI",
  "caption": "Ground truth caption...",
  "diagnosis": "Ground truth diagnosis",
  "boxes": [[x1, y1, x2, y2], ...]
}
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{nova_brain_mri_env,
  title = {NOVA Brain MRI Environment for MedMarks},
  year = {2024},
  url = {https://github.com/your-repo/nova-brain-mri}
}
```
