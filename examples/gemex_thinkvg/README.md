# GEMeX-ThinkVG example

Visual grounding with chain-of-thought reasoning for chest X-ray analysis using verifiable rewards.

## Overview

Demonstrates RL fine-tuning with GAZE and the [verifiers](https://docs.primeintellect.ai/verifiers) package for multi-rollout training. The GEMeX-ThinkVG task asks a model to analyze chest X-rays with visual reasoning, use tools (zoom, crop, contrast, threshold, search), and return a structured response with an answer and a bounding-box location. Training and evaluation share the same `GEMeXThinkVGToolEnv` environment, and rollouts are scored by three verifiable reward components.

### Reward components

The reward combines three components (default weights `answer=0.4, location=0.3, bbox=0.3`):

1. Answer reward (`weights.answer`): semantic matching of medical findings via exact match, contains match, and token F1, with medical-term normalization and open/closed question handling.
2. Location reward (`weights.location`): anatomical-region matching with hierarchical matching (organ -> subregion -> specific), synonym normalization, and laterality awareness.
3. BBox reward (`weights.bbox`): spatial accuracy with IoU and generalized IoU, a center-distance penalty, and IoU@0.5 / IoU@0.3 thresholds.

## Dataset

- Source: [GEMeX-ThinkVG on HuggingFace](https://huggingface.co/datasets/BoKelvin/GEMeX-ThinkVG)
- Images: chest X-rays from MIMIC-CXR-JPG (PhysioNet credentialed access)
- Annotations: findings with bounding boxes and anatomical locations
- Format: chain-of-thought reasoning with visual grounding
- Splits: currently only `train` is available

MIMIC-CXR-JPG requires credentialed access:

1. Complete CITI training at [physionet.org/settings/credentialing](https://physionet.org/settings/credentialing/)
2. Sign the data use agreement at [physionet.org/content/mimic-cxr-jpg/2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
3. Download with `wget -r -N -c -np --user YOUR_USER --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/`
4. Pass the root directory as `--image-dir`

## Install

Run from the repository root:

```bash
uv sync --extra gemex
# or
pip install gaze-vlm[gemex]
```

## Run

```bash
uv run python -m examples.gemex_thinkvg.eval \
  --dataset ./data/test.jsonl \
  --image-dir /path/to/mimic-cxr-jpg \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --output ./results
```

## Run locally (LM Studio)

`run_local.sh` sweeps single-turn then agentic against a local OpenAI-compatible server. It requires the dataset and MIMIC-CXR image root as positional arguments; pass `--base-url` (default `http://localhost:1234/v1`) to point at LM Studio:

```bash
./examples/gemex_thinkvg/run_local.sh glm-4.6v-flash ./data/test.jsonl /path/to/mimic-cxr-jpg http://localhost:1234/v1 50
```

GEMeX needs `n_ctx >= 8192` (16384 recommended); the image plus schema prompt is large and thinking models need headroom. The script sets `--max-image-dim 256` to keep encoded images within budget. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs).

## Flags

- `--dataset PATH`: GEMeX JSONL dataset (required)
- `--image-dir PATH`: root directory for resolving `image_path` entries
- `--model NAME`: model name (OpenAI/OpenRouter format, or local ID for `--base-url`)
- `--base-url URL`: OpenAI-compatible server, e.g. `http://localhost:1234/v1`
- `--mode {single_turn,agentic}`: evaluation mode (default `agentic`)
- `--use-tools`: enable visual tools in agentic mode
- `--use-web-search`: enable search tools in agentic mode
- `--max-turns N`: override max turns (single_turn requires 1; agentic defaults to 8)
- `--max-tokens N`: completion tokens per turn (thinking models need >= 4096)
- `--max-image-dim N`: downscale images so neither side exceeds N pixels before encoding
- `--reward-weights A,L,B`: comma-separated weights for answer,location,bbox (default `0.4,0.3,0.3`)
- `--num-samples N`: number of samples (-1 for all)
- `--reasoning`: enable model reasoning mode
- `--seed N`: random seed for reproducibility
- `--output PATH`: output directory (default `./results`)
- `--verbose`: verbose logs

## Output

- Per-sample reward breakdowns and aggregate metrics (mean reward, answer accuracy, per-component rewards, mean IoU, IoU@0.5, IoU@0.3, mean turns, mean tokens) written under the chosen `--output` directory.

## Programmatic use

### Loading the environment

```python
from examples.gemex_thinkvg.src import load_environment

env = load_environment(
    dataset_path="./data/test.jsonl",
    max_turns=8,
)
```

### Running the processor directly

```python
from examples.gemex_thinkvg.src import GEMeXProcessor

processor = GEMeXProcessor(
    model_name="qwen3.5-a3b",
    max_turns=8,
    use_tools=True,
)

metadata = {
    "question": "Describe and locate any findings in the right lung.",
    "question_type": "open_ended",
}
result = await processor.analyze(images="path/to/xray.jpg", metadata=metadata)

print(result.final_response["answer"])
print(result.final_response["location"])
```

### Computing rewards

```python
from examples.gemex_thinkvg.src import GEMeXRewardFunction, RewardWeights

reward_fn = GEMeXRewardFunction(
    weights=RewardWeights(answer=0.5, location=0.25, bbox=0.25)
)

predictions = [
    {
        "answer": "consolidation",
        "location": {"reference": "right lower lobe", "bbox": [120, 180, 220, 280]},
    }
]
references = [
    {
        "answer": "right lower lobe consolidation",
        "location": {"reference": "right lung lower zone", "bbox": [110, 170, 230, 290]},
        "question_type": "open_ended",
    }
]

rewards = reward_fn(predictions, references)
print(f"Combined reward: {rewards[0]:.3f}")
```

### Multi-turn interaction

The framework drives tool use with JSON tool-calling under the hood. The exchange below is an illustrative paraphrase of the turn flow, not the literal JSON the model emits:

```
Assistant: I need to examine the right lower lung more closely.
           -> zoom region [100, 150, 250, 300]

Environment: [zoom applied to region [100, 150, 250, 300]]
             The zoomed region shows enhanced detail. Continue your analysis.

Assistant: I see an opacity in the right lower lobe. Let me check the contrast.
           -> adjust_contrast factor 1.5

Environment: [contrast adjusted by factor 1.5]
             Image contrast enhanced. Continue your analysis.

Assistant: {
  "reasoning": "The zoomed and enhanced image shows a consolidative opacity...",
  "answer": "consolidation",
  "location": {
    "reference": "right lower lobe",
    "bbox": [120, 180, 220, 280]
  }
}
```

## RL training

`train.py` is a training-integration template, not a standalone trainer. It builds the same `GEMeXThinkVGToolEnv` used by `eval.py` (guaranteeing train/eval parity), validates that the reward weights sum to 1.0, and either dispatches to the evaluator (`--mode eval`) or writes a `config.json` you pass to your own verifiers training loop (`--mode train`). It does not run gradient updates itself.

```bash
# Prepare a training config (writes config.json; no gradient updates)
uv run python -m examples.gemex_thinkvg.train \
    --mode train \
    --dataset ./data/train.jsonl \
    --model openai/gpt-4o \
    --learning-rate 1e-5 \
    --batch-size 8 \
    --epochs 3 \
    --answer-weight 0.4 \
    --location-weight 0.3 \
    --bbox-weight 0.3 \
    --output ./runs/gemex
```

Then drive a verifiers RL loop with the prepared environment (a `vf.MultiTurnEnv` subclass):

```python
from examples.gemex_thinkvg.src import load_environment

env = load_environment(dataset_path="./data/train.jsonl")
# Pass `env` and the saved config to your verifiers trainer.
# See the verifiers docs for the trainer API.
```

## Structure

```
gemex_thinkvg/
    src/
        __init__.py              # Package exports
        processor.py             # GEMeXProcessor
        dataset.py               # Dataset loader with MIMIC-CXR resolution
        schemas.py               # ThinkVG XML/JSON schemas
        rewards/                 # Verifiable reward functions
            answer.py            # Answer semantic matching
            location.py          # Anatomical region matching
            bbox.py              # IoU-based bbox accuracy
            combined.py          # Combined reward function
        verifiers/               # verifiers package integration
            environment.py       # GEMeXThinkVGToolEnv (MultiTurnEnv)
    tests/                       # Hermetic smoke tests
    train.py                     # Training-config prep template (see note)
    eval.py                      # Evaluation script
    run_local.sh                 # Local (LM Studio) evaluation sweep
    README.md
```

## References

- [GEMeX-ThinkVG dataset](https://huggingface.co/datasets/BoKelvin/GEMeX-ThinkVG)
- [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [verifiers documentation](https://docs.primeintellect.ai/verifiers)
- [GAZE](https://github.com/liamchalcroft/gaze)

## License

Follows the license terms of the GAZE framework.
