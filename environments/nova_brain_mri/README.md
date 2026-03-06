# NOVA Brain MRI Environment

MedMarks-compatible evaluation environment for the NOVA brain MRI benchmark.

## Tasks

| Task | Description | Metric |
|------|-------------|--------|
| Caption | Radiological description with sequence characteristics | Token F1 |
| Diagnosis | Primary and differential diagnoses | Top-k accuracy |
| Localization | Abnormality detection with bounding boxes | IoU-based detection F1 |

## Installation

```bash
cd environments/nova_brain_mri
pip install -e .
```

## Usage

### Via medarc-eval CLI

```bash
medarc-eval nova-brain-mri -m gpt-4o -n 100
medarc-eval nova-brain-mri -m gpt-4o --task diagnosis -n 50
medarc-eval nova-brain-mri -m gpt-4o --max-turns 10
```

### Via Python API

```python
import verifiers as vf

env = vf.load_environment(
    "nova-brain-mri",
    split="test",
    task="all",
    max_turns=10,
)

results = env.evaluate(client=openai_client, model="gpt-4o", num_examples=100)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | `"test"` | Dataset split |
| `task` | str | `"all"` | caption, diagnosis, localization, all |
| `max_turns` | int | 10 | Maximum conversation turns |
| `iou_threshold` | float | 0.5 | IoU threshold for localization (NOVA ACC50) |
| `data_dir` | str | None | Custom data directory |

## Response Schema

Models return JSON with this structure:

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
      {"diagnosis": "ADEM", "confidence": 0.3}
    ],
    "confidence": 0.75,
    "evidence": ["periventricular lesions"]
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
  "continue": false
}
```

## Reward Functions

### Caption
Token-level F1 between predicted and reference captions.

### Diagnosis
Combined: 60% top-1 accuracy (primary diagnosis matches reference) + 40% coverage (fraction of reference diagnoses matched). Medical term normalization applied before comparison.

### Localization
Detection F1 using greedy best-IoU matching. Each predicted box is matched to its best-IoU ground truth box. Matches above the threshold count as true positives. F1 computed from precision and recall.

### Combined Rubric
Weights: 33% caption, 34% diagnosis, 33% localization.

## Data Format

Each case:
```json
{
  "image_path": "/path/to/brain_mri.png",
  "clinical_history": "45yo male with headaches",
  "modality": "MRI",
  "caption": "Ground truth caption",
  "diagnosis": "Ground truth diagnosis",
  "boxes": [[x1, y1, x2, y2]]
}
```

## License

MIT
