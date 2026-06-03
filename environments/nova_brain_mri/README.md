# NOVA brain MRI environment

MedMarks-compatible evaluation environment for the NOVA brain MRI benchmark.

Part of [GAZE](https://github.com/liamchalcroft/gaze); see the [MedMarks integration](https://github.com/liamchalcroft/gaze/blob/main/docs/MEDMARKS_INTEGRATION.md) docs.

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

## Data

The environment loads cases from `data/nova_<split>.jsonl`, which is not shipped.
Build it once from the HuggingFace NOVA dataset (`c-i-ber/Nova`) with the bundled
`prepare_data.py`. It needs `pandas` and `huggingface-hub` in addition to the
package's own dependencies:

```bash
pip install -e .
pip install pandas huggingface-hub
python prepare_data.py --split test
```

This downloads NOVA, copies the brain-MRI images into `data/images/`, and writes
`data/nova_test.jsonl` with the caption, diagnosis, clinical history, and gold
bounding boxes for each case. Use `--max-samples N` for a quick subset, or
`--data-dir /path/to/local/Nova` to skip the download.

## Usage

### Run standalone

The package ships a self-contained CLI, so no external runner is required. After
`prepare_data.py`, evaluate with either the console script or the module form:

```bash
# Console script (registered in pyproject [project.scripts])
nova-brain-mri --model gpt-4o --num-examples 100 --task all

# Equivalent module invocation
python -m nova_brain_mri --model gpt-4o --num-examples 100 --task diagnosis
```

Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (or pass `--api-key`); an
OpenAI-compatible endpoint can be targeted with `--base-url`.

### Via medarc-eval CLI

`medarc-eval` is the MedARC evaluation runner from the [MedMarks](https://medmarks.ai)
ecosystem; this package registers itself as a verifiers environment (see
`[tool.verifiers.environment]` in `pyproject.toml`) so MedMarks can discover it.

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

## Response schema

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

## Reward functions

### Caption
Token-level F1 between predicted and reference captions.

### Diagnosis
Combined: 60% top-1 accuracy (primary diagnosis matches reference) + 40% coverage (fraction of reference diagnoses matched). Medical term normalization applied before comparison.

### Localization
Detection F1 using greedy best-IoU matching. Each predicted box is matched to its best-IoU ground truth box. Matches above the threshold count as true positives. F1 computed from precision and recall.

### Combined rubric
Weights: 33% caption, 34% diagnosis, 33% localization.

## Data format

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
