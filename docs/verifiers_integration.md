# Verifiers Integration

Integration with the [verifiers](https://github.com/primeintellect-ai/verifiers) package for RL training with verifiable rewards.

## Overview

The `radiant_harness.verifiers` module provides:

- `BaseMultiTurnEnv` -- base class for multi-turn RL environments (extends `vf.MultiTurnEnv`)
- `VerifiableProcessorMixin` -- mixin that adds `as_verifiers_env()` to processors
- `RadiantHarnessAdapter` -- bridges processor and verifiers message formats
- Reward functions: `ExactMatchReward`, `TokenF1Reward`, `IoUReward`, `CombinedReward`

## Installation

```bash
# verifiers is a core dependency, installed with:
uv sync
```

For RL training with torch/transformers:
```bash
uv sync --group rl
```

## Quick Start

### 1. Multi-Turn Environment

```python
from radiant_harness.verifiers import BaseMultiTurnEnv

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

### 2. Reward Functions

```python
from radiant_harness.verifiers import ExactMatchReward, TokenF1Reward, CombinedReward

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

### 3. Processor-Based Environment

Use `VerifiableProcessorMixin` to turn a processor into a verifiers environment:

```python
from radiant_harness import AgenticProcessorBase
from radiant_harness.verifiers import VerifiableProcessorMixin, ExactMatchReward

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
    name="BaseRadiantEnv",
    log_dir=None,
)
```

Methods to override:
- `get_system_prompt() -> str`
- `_build_user_message(case) -> str | list`
- `build_initial_state(prompt, info) -> dict`
- `is_completed(messages, state, info) -> bool`
- `env_response(messages, state, info) -> tuple[Messages, State]`

### Reward Functions

All inherit from `BaseRewardFunction` which defines `__call__(prompt, completion, info) -> float`.

**ExactMatchReward** -- string equality after normalization:
```python
ExactMatchReward(
    normalize=True,       # lowercase + strip whitespace
    case_sensitive=False,
    strip_braces=True,    # remove {}[]()
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
    iou_threshold=0.5,
    normalized=True,      # coordinates in [0,1]
)
```

**CombinedReward** -- weighted combination:
```python
CombinedReward(
    rewards=[ExactMatchReward(), TokenF1Reward()],
    weights=[0.6, 0.4],
    names=["exact", "f1"],
)
```

### RadiantHarnessAdapter

Bridges a Radiant Harness processor with verifiers message formats:

```python
from radiant_harness.verifiers import RadiantHarnessAdapter

adapter = RadiantHarnessAdapter(processor=my_processor)
result = await adapter.process_verifiers_messages(messages, info)
EnvClass = adapter.create_environment_class(max_turns=5)
```

### Custom Reward Functions

```python
from radiant_harness.verifiers import BaseRewardFunction

class MyReward(BaseRewardFunction):
    def __call__(self, prompt, completion, info) -> float:
        pred = self._extract_prediction(completion)
        ref = info.get("answer", "")
        return float(pred == ref)
```

## Data Format

Use JSONL with consistent fields:
```json
{"question": "...", "answer": "...", "image": "..."}
{"question": "...", "answer": "...", "context": "..."}
```

## Troubleshooting

- **Import errors**: run `uv sync` to install dependencies
- **Memory issues**: reduce batch size
- **Debugging**: pass `log_dir="./logs"` to the environment constructor
