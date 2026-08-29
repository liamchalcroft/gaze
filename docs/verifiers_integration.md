# Verifiers integration

Integration with the [verifiers](https://github.com/primeintellect-ai/verifiers) package for RL training with verifiable rewards.

See the [API reference](api/verifiers.md) for signatures.

## Overview

The `gaze.verifiers` module provides:

- `BaseMultiTurnEnv` -- base class for multi-turn RL environments (extends `vf.MultiTurnEnv`)
- `VerifiableProcessorMixin` -- mixin that adds `as_verifiers_env()` to processors
- `GazeAdapter` -- bridges processor and verifiers message formats
- Reward functions: `ExactMatchReward`, `TokenF1Reward`, `IoUReward`, `CombinedReward`

## Installation

`verifiers` is not part of the core `gaze-vlm` runtime dependencies. It is
declared in the `dev` dependency group and in several optional extras
(`[verifiers]`, `[medmarks]`, `[gemex]`, `[agentclinic]`), so it is
only pulled in when you ask for it.

End users install the optional extra:

```bash
pip install gaze-vlm[verifiers]
```

Contributors working from a checkout get it through the dev group, which
`uv sync` installs by default:

```bash
uv sync
```

RL training additionally needs torch/transformers/datasets, provided by the
`rl` group:

```bash
uv sync --group rl
```

## Quick start

### 1. Multi-turn environment

```python
from gaze.verifiers import BaseMultiTurnEnv

class MyEnvironment(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant."

    def _build_user_message(self, case):
        return f"Question: {case['question']}"

env = MyEnvironment(
    dataset_path="my_data.jsonl",
    max_turns=5,
)
```

### 2. Reward functions

```python
from gaze.verifiers import ExactMatchReward, TokenF1Reward, CombinedReward

# Single reward
reward = ExactMatchReward(normalize=True)
score = reward(prompt, completion, {"answer": "4"})

# Combined rewards
combined = CombinedReward(
    rewards=[ExactMatchReward(), TokenF1Reward()],
    weights=[0.6, 0.4],
    names=["exact", "f1"],
)
```

### 3. Processor-based environment

Use `VerifiableProcessorMixin` to turn a processor into a verifiers environment:

```python
from gaze import AgenticProcessorBase
from gaze.verifiers import VerifiableProcessorMixin, ExactMatchReward

class MyProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a helpful assistant."

    def get_user_message(self, images, metadata):
        return metadata.get("question", "")

    def get_response_schema(self):
        return None

    def validate_response(self, response):
        return "continue" in response

    def get_reward_function(self):
        return ExactMatchReward()

EnvClass = MyProcessor.as_verifiers_env(
    max_turns=5,
    dataset_path="my_data.jsonl",
)
env = EnvClass()
```

## Components

### BaseMultiTurnEnv

Extends `vf.MultiTurnEnv`. Provides dataset loading, turn tracking, and logging.

Constructor:
```python
BaseMultiTurnEnv(
    cases=None,              # Pre-loaded cases (list of dicts)
    dataset_path=None,       # Path to JSONL file
    max_turns=10,
    name="BaseGazeEnv",
    log_dir=None,
)
```

Methods to override:
- `get_system_prompt() -> str`
- `_build_user_message(case) -> str | list`
- `async setup_state(state) -> State` -- seed per-episode state
- `async env_response(messages, state, **kwargs) -> Messages` -- mutate `state` in place
- a `@vf.stop`-decorated `async` predicate for a custom stop condition
  (the base class ships `_turn_limit_reached`)

### Reward functions

All inherit from `BaseRewardFunction` which defines `__call__(prompt, completion, info) -> float`.

**ExactMatchReward** -- string equality after normalization:
```python
ExactMatchReward(
    normalize=True,       # collapse whitespace (see case_sensitive for casing)
    case_sensitive=False,
    strip_braces=True,    # also strip surrounding {}[]().,;
)
```

**TokenF1Reward** -- token-level F1 score:
```python
TokenF1Reward(
    normalize=True,
    case_sensitive=False,
    tokenize="simple",    # "simple", "word", or "character"
)
```

**IoUReward** -- bounding box overlap:
```python
IoUReward(
    iou_threshold=0.5,    # used only when continuous=False
    normalized=True,      # coordinates in [0,1]
    continuous=True,      # raw IoU for smooth gradients; False = step at threshold
    area_penalty_start=0.5,  # penalise boxes covering more than this fraction
)
```

With `area_penalty_start < 1.0` and `normalized=True`, a prediction whose
coordinates fall outside `[0, 1]` is treated as pixel-space. The area penalty
then needs the image area, so `info["image_area"]` is required and the reward
**fails closed at 0.0** without it -- including for an otherwise perfect box:

```python
IoUReward()("", '{"bbox": [10, 10, 50, 50]}', {"bbox": [10, 10, 50, 50]})
# -> 0.0, with a warning

IoUReward()("", '{"bbox": [10, 10, 50, 50]}', {"bbox": [10, 10, 50, 50], "image_area": 65536})
# -> 1.0
```

**CombinedReward** -- weighted combination:
```python
CombinedReward(
    rewards=[ExactMatchReward(), TokenF1Reward()],
    weights=[0.6, 0.4],
    names=["exact", "f1"],
)
```

### GazeAdapter

Bridges a GAZE processor with verifiers message formats:

```python
from gaze.verifiers import GazeAdapter

adapter = GazeAdapter(processor=my_processor)
result = await adapter.process_verifiers_messages(messages, info)
EnvClass = adapter.create_environment_class(max_turns=5)
```

### Custom reward functions

```python
from gaze.verifiers import BaseRewardFunction

class MyReward(BaseRewardFunction):
    def __call__(self, prompt, completion, info) -> float:
        pred = self._extract_prediction(completion)
        ref = info.get("answer", "")
        return float(pred == ref)
```

## Data format

Use JSONL with consistent fields:
```json
{"question": "...", "answer": "...", "image": "..."}
{"question": "...", "answer": "...", "context": "..."}
```

## Troubleshooting

- **Import errors**: install the verifiers extra (`pip install gaze-vlm[verifiers]`), or run `uv sync` from a checkout
- **Memory issues**: reduce batch size
- **Debugging**: pass `log_dir="./logs"` to the environment constructor
