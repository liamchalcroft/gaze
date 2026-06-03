# NOVA brain-MRI environment

A MedMarks-style evaluation environment for the NOVA brain-MRI benchmark,
built on the [GAZE](https://github.com/liamchalcroft/gaze) framework
(`gaze-vlm`) and packaged for the Prime Intellect Environments Hub.

The environment evaluates the policy model under test: that model receives a
brain-MRI image plus clinical history and emits a single NOVA JSON object, and
the rubric scores it. GAZE supplies the system prompt, the JSON response
schema, the JSON and text extraction helpers, and the reward functions. GAZE
is not run as an agent here.

## How it depends on GAZE

`src/nova_brain_mri/_utils.py` re-exports the extraction and IoU helpers from
GAZE, and the reward functions import them from there:

- `gaze.utils.compute_iou` for IoU over `[x1, y1, x2, y2]` boxes.
- `gaze.utils.extract_json_from_text` for parsing the model's JSON.
- `gaze.verifiers.rewards.extract_completion_text` for reading the completion.

`src/nova_brain_mri/prompts.py` holds the GAZE-style NOVA system prompt and the
JSON schema (`NOVA_SCHEMA`) the model is instructed to follow.

## Tasks

| Task | Description | Metric |
|------|-------------|--------|
| caption | Radiological description with sequence characteristics | Token F1 |
| diagnosis | Primary diagnosis and differentials | Top-1 plus coverage F1 |
| localization | Abnormality detection with bounding boxes | IoU-based detection F1 |
| all | All three combined | Weighted (0.33 / 0.34 / 0.33) |

## Data

The environment loads cases from `data/nova_<split>.jsonl`, which is not
shipped. Build it once from the HuggingFace NOVA dataset (`c-i-ber/Nova`) with
`prepare_data.py`. It needs `pandas` and `huggingface-hub` in addition to the
package dependencies:

```bash
pip install -e .
pip install pandas huggingface-hub
python prepare_data.py --split test
```

This downloads NOVA, copies the brain-MRI images into `data/images/`, and
writes `data/nova_test.jsonl` with the caption, diagnosis, clinical history,
and gold bounding boxes for each case. Use `--max-samples N` for a quick subset,
or `--data-dir /path/to/local/Nova` to skip the download.

Each JSONL line:

```json
{
  "image_path": "/abs/path/case0001.png",
  "clinical_history": "45yo male with headaches",
  "modality": "MRI",
  "caption": "Ground-truth caption",
  "diagnosis": "Ground-truth diagnosis",
  "boxes": [[x1, y1, x2, y2]]
}
```

`image_path` is an absolute filesystem path with no scheme; the environment
prepends `file://` when building the user message.

## Loading the environment

```python
import verifiers as vf

env = vf.load_environment("nova-brain-mri", split="test", task="all")
results = env.evaluate(client=openai_client, model="gpt-4o", num_examples=100)
```

`load_environment(**kwargs)` is the hub entry point. Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | `"test"` | Dataset split |
| `task` | str | `"all"` | `caption`, `diagnosis`, `localization`, `all` |
| `max_turns` | int | 10 | Maximum turns per episode |
| `iou_threshold` | float | 0.5 | IoU threshold for localization (NOVA ACC50) |
| `data_dir` | str \| None | None | Directory holding `<split>.jsonl` |

## Standalone CLI

After `prepare_data.py`, evaluate with the console script or the module form:

```bash
nova-brain-mri --model gpt-4o --num-examples 100 --task all
python -m nova_brain_mri --model gpt-4o --num-examples 100 --task diagnosis
```

Set `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (or pass `--api-key`); an
OpenAI-compatible endpoint can be targeted with `--base-url`.

## Hub workflow

The package exposes `load_environment` and declares hub metadata in
`pyproject.toml`: `[project]` (with `tags`), `[build-system]` (hatchling), and
`[tool.verifiers.eval]` (`num_examples`, `rollouts_per_example`). Use
`prime env init` to scaffold a hub package and `prime env push` to publish.
Confirm the exact flags with `prime env --help` before publishing.

## Response schema

The model returns a single JSON object:

```json
{
  "caption": "Axial T2 FLAIR showing periventricular white matter lesions",
  "diagnosis": {
    "primary_diagnosis": "Multiple sclerosis",
    "differential_diagnoses": ["ADEM"],
    "confidence": 0.75,
    "evidence": ["periventricular lesions"]
  },
  "localization": [
    {"bounding_box": [120, 80, 180, 140], "label": "demyelinating lesion"}
  ],
  "continue": false,
  "reasoning": "Brief analysis."
}
```

Bounding boxes are `[x1, y1, x2, y2]` in absolute pixels. Set `continue` to
`false` once the final answer is ready.

## Reward functions

- Caption: token-level F1 between predicted and reference captions, with
  stopword filtering and multiset intersection.
- Diagnosis: 60% top-1 accuracy plus 40% coverage F1, with medical term
  normalization and abbreviation expansion.
- Localization: detection F1 via greedy best-IoU matching at the threshold,
  with an area penalty for degenerate full-image boxes.
- Combined (`all`): weighted 0.33 caption, 0.34 diagnosis, 0.33 localization.

## License

MIT
