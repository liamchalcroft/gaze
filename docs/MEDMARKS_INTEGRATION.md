# MedMarks Integration

Integration with [MedMarks](https://medmarks.ai), an evaluation suite for medical LLMs developed by Sophont, MedARC, and Prime Intellect.

## Architecture

```
MedMarks Platform (medmarks.ai, medarc-eval CLI)
        |
Verifiers Framework (vf.MultiTurnEnv, vf.Rubric)
        |
NOVA Brain MRI Environment (environments/nova_brain_mri/)
        |
GAZE (AgenticProcessorBase, VerifiableProcessorMixin)
```

The `environments/nova_brain_mri/` package provides a MedMarks-compatible environment for the NOVA brain-MRI benchmark. It wraps the verifiers framework and uses GAZE reward utilities.

## Installation

```bash
# Install gaze-vlm with MedMarks dependencies (from source)
pip install -e .[medmarks]

# Install the NOVA brain-MRI environment
cd environments/nova_brain_mri
pip install -e .
```

## Usage

### Via medarc-eval CLI

```bash
medarc-eval nova-brain-mri -m gpt-4o -n 100
medarc-eval nova-brain-mri -m gpt-4o --task diagnosis -n 50
medarc-eval nova-brain-mri -m gpt-4o --use-tools --max-turns 10
```

### Via Python API

See [environments/nova_brain_mri/README.md](https://github.com/liamchalcroft/gaze/blob/main/environments/nova_brain_mri/README.md) for the Python API and configuration reference.

### Via GAZE Processor

```python
from examples.nova.src.processor import NOVAAgenticProcessor

processor = NOVAAgenticProcessor(
    model_name="openai/gpt-4o",
    use_tools=True,
    use_web_search=True,
    max_turns=10,
    task="all",
)

EnvClass = processor.as_verifiers_env(
    dataset_path="data/nova_test.jsonl",
    image_base_path=Path("data/images"),
)
env = EnvClass()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | `"test"` | Dataset split |
| `task` | str | `"all"` | Task: caption, diagnosis, localization, all |
| `max_turns` | int | 10 | Maximum conversation turns |
| `use_tools` | bool | True | Enable visual tools |
| `use_web_search` | bool | False | Enable PubMed search |
| `iou_threshold` | float | 0.5 | IoU threshold for localization |
| `data_dir` | str | None | Custom dataset directory |

## Reward Functions

The environment provides three task-specific rewards:

- **Caption**: token-level F1 between predicted and reference captions
- **Diagnosis**: 60% top-1 accuracy + 40% coverage of reference diagnoses (with normalization)
- **Localization**: detection F1 using greedy best-IoU matching at the configured threshold

Combined rubric weights: 33% caption, 34% diagnosis, 33% localization.

## Creating Custom Environments

Any `AgenticProcessorBase` subclass with `VerifiableProcessorMixin` can become a MedMarks-compatible environment:

```python
from gaze import AgenticProcessorBase
from gaze.verifiers import VerifiableProcessorMixin, BaseRewardFunction

class MyMedicalProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a medical imaging expert."

    def get_user_message(self, images, metadata):
        return "Analyze this medical image."

    def get_response_schema(self):
        return None

    def validate_response(self, response):
        return "diagnosis" in response

    def get_reward_function(self) -> BaseRewardFunction:
        return MyCustomReward()

EnvClass = MyMedicalProcessor.as_verifiers_env(
    max_turns=10,
    dataset_path="my_dataset.jsonl",
)
```

## Resources

- [MedMarks Leaderboard](https://medmarks.ai)
- [Prime Intellect Hub](https://app.primeintellect.ai)
- [MedARC GitHub](https://github.com/MedARC-AI/med-lm-envs)
- [Verifiers](https://github.com/primeintellect-ai/verifiers)
