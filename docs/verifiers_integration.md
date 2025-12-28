# Verifiers Integration

This guide explains how to integrate Radiant Harness with the [verifiers](https://docs.primeintellect.ai/verifiers) package for reinforcement learning training with verifiable rewards.

## Overview

The integration provides:

- **BaseMultiTurnEnv**: A reusable base class for multi-turn environments
- **ToolEnv-first examples**: Examples now use `vf.ToolEnv` for tool-calling workflows
- **Reward Functions**: Common reward implementations (exact match, F1, IoU)
- **Adapter Utilities**: Seamless conversion between formats
- **Training Examples**: Ready-to-use patterns for common tasks

## Installation

```bash
# Install the main package (verifiers is included)
uv sync
# or
pip install -e .
```

## Quick Start

### 1. Basic Multi-Turn Environment

```python
from radiant_harness.verifiers import BaseMultiTurnEnv

class MyEnvironment(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant."

    def _build_user_message(self, case):
        return f"Question: {case['question']}"

# Create environment
env = MyEnvironment(
    dataset_path="my_data.jsonl",
    max_turns=5,
)
```

### 2. Using Reward Functions

```python
from radiant_harness.verifiers import ExactMatchReward, CombinedReward

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

### 3. Adapting Processors

```python
from radiant_harness import AgenticProcessorBase
from radiant_harness.verifiers import VerifiableProcessorMixin
from radiant_harness.verifiers import ExactMatchReward

class MyProcessor(VerifiableProcessorMixin, AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a helpful assistant."

    def get_user_message(self, images, metadata):
        return metadata.get("question", "")

    def get_response_schema(self):
        return {"type": "object", "properties": {"continue": {"type": "boolean"}}}

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

## Core Components

### BaseMultiTurnEnv

The base class provides:

- **Dataset Loading**: Automatic JSONL parsing
- **Message Processing**: Standard message format handling
- **State Management**: Turn tracking and limits
- **Tool Request Parsing**: Optional tool support
- **Logging**: Debug logging utilities

#### Key Methods to Override

```python
class MyEnvironment(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        """Return system prompt for the task."""
        return "Your system prompt here."

    def _build_user_message(self, case: dict) -> str | list:
        """Build user message from case data."""
        # Return string for text-only or list for multimodal
        return f"Question: {case['question']}"

    def build_initial_state(self, prompt, info) -> dict:
        """Initialize episode state."""
        return {
            "turn": 0,
            "info": info,
            "custom_state": "value",
        }

    async def is_completed(self, messages, state, info=None) -> bool:
        """Check if episode should end."""
        # Custom completion logic
        return state.get("turn", 0) >= self._max_turns

    async def env_response(self, messages, state, info=None) -> tuple:
        """Generate environment response."""
        # Custom response logic
        new_state = dict(state)
        new_state["turn"] += 1
        return [{"role": "user", "content": "Response"}], new_state
```

### Reward Functions

#### Available Types

1. **ExactMatchReward**: Exact string matching
2. **TokenF1Reward**: Token-level F1 score
3. **IoUReward**: Bounding box overlap
4. **CombinedReward**: Weighted combination

#### Custom Reward Function

```python
from radiant_harness.verifiers import BaseRewardFunction

class MyReward(BaseRewardFunction):
    def __call__(self, prompt, completion, info) -> float:
        # Extract prediction
        pred = self._extract_prediction(completion)
        # Extract reference
        ref = info.get("answer", "")
        # Compute reward
        return float(pred == ref)
```

#### Reward Function Examples

```python
# Exact match with normalization
exact = ExactMatchReward(
    normalize=True,      # Lowercase, strip whitespace
    case_sensitive=False,
    strip_braces=True,   # Remove {}[]()
)

# Token F1 with word tokenization
f1 = TokenF1Reward(
    normalize=True,
    tokenize="word",     # "simple", "word", or "character"
)

# IoU for bounding boxes
iou = IoUReward(
    iou_threshold=0.5,   # Minimum IoU for full reward
    normalized=True,     # Coordinates are 0-1
)

# Combine multiple rewards
combined = CombinedReward(
    rewards=[exact, f1, iou],
    weights=[0.4, 0.3, 0.3],
    names=["exact", "semantic", "spatial"],
)

# Get details about each component
score = combined(prompt, completion, info)
details = info.get("_reward_details", {})
print(f"Exact: {details['exact']:.2f}")
print(f"F1: {details['semantic']:.2f}")
print(f"IoU: {details['spatial']:.2f}")
```

### Adapter Utilities

#### RadiantHarnessAdapter

Bridges between Radiant Harness processors and verifiers:

```python
from radiant_harness.verifiers import RadiantHarnessAdapter

adapter = RadiantHarnessAdapter(processor=processor)

# Process verifiers messages
result = await adapter.process_verifiers_messages(messages, info)

# Or build a MultiTurnEnv class
EnvClass = adapter.create_environment_class(max_turns=5)
env = EnvClass()
```

## Training Setup

### Basic RL Training

```python
import verifiers as vf

# Create environment
env = MyEnvironment(dataset_path="train.jsonl")

# Create reward rubric
reward = ExactMatchReward()
rubric = vf.Rubric(
    funcs=[reward],
    weights=[1.0],
    names=["accuracy"],
)

# Configure trainer
trainer = vf.RLTrainer(
    environment=env,
    model="gpt-4o",  # or base model for fine-tuning
    reward_rubric=rubric,
    learning_rate=1e-5,
    batch_size=16,
    max_rollouts=4,
    epochs=5,
    # RL-specific parameters
    ppo_epochs=4,
    ppo_clip=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
)

# Train
trainer.train()
```

### Evaluation

```python
# Create evaluator
evaluator = vf.Evaluator(
    environment=env,
    model="your-checkpoint",
    reward_rubric=rubric,
)

# Evaluate
results = evaluator.evaluate()
print(f"Mean reward: {results['mean_reward']:.3f}")
print(f"Success rate: {results['success_rate']:.2%}")
```

## Examples

### 1. Text QA (like PubMedQA)

```python
from radiant_harness.verifiers import BaseMultiTurnEnv, ExactMatchReward

class QAEnvironment(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        return "Answer the medical question with yes/no/maybe."

    def _build_user_message(self, case):
        context = case.get("context", "")
        question = case.get("question", "")
        return f"Context: {context}\n\nQuestion: {question}"

    async def is_completed(self, messages, state, info=None):
        if state.get("turn", 0) >= self._max_turns:
            return True

        last = self._last_assistant_text(messages)
        # Check for yes/no/maybe answer
        return any(word in last.lower() for word in ["yes", "no", "maybe"])

# Usage
env = QAEnvironment(
    dataset_path="pubmedqa.jsonl",
    max_turns=3,
)
reward = ExactMatchReward()
```

### 2. Visual Grounding (like GEMeX)

```python
from radiant_harness.verifiers import (
    BaseMultiTurnEnv,
    ExactMatchReward,
    IoUReward,
    CombinedReward
)

class VisualGroundingEnv(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        return "Analyze the image and locate findings."

    def _build_user_message(self, case):
        return [
            {"type": "image_url", "image_url": {"url": case["image"]}},
            {"type": "text", "text": f"Question: {case['question']}"}
        ]

    async def is_completed(self, messages, state, info=None):
        last = self._last_assistant_text(messages)
        # Check for JSON response with bbox
        return "bbox" in last and "[" in last

# Combined reward for answer + localization
reward = CombinedReward(
    rewards=[ExactMatchReward(), IoUReward()],
    weights=[0.6, 0.4],
    names=["answer", "localization"],
)
```

### 3. Multi-turn Reasoning (like AgentClinic)

```python
class DiagnosticEnv(BaseMultiTurnEnv):
    def get_system_prompt(self) -> str:
        return "Gather information by requesting HISTORY, EXAM, TESTS, or IMAGE."

    def build_initial_state(self, prompt, info):
        return {
            "turn": 0,
            "info": info,
            "asked": False,  # Track if info was requested
        }

    async def env_response(self, messages, state, info=None):
        text = self._last_assistant_text(messages).lower()
        new_state = dict(state)
        new_state["turn"] += 1

        # Check for information requests
        if "history" in text:
            new_state["asked"] = True
            return [{"role": "user", "content": info.get("history", "")}], new_state
        elif "exam" in text:
            new_state["asked"] = True
            return [{"role": "user", "content": info.get("exam", "")}], new_state
        # ... other requests

        # Prompt for diagnosis
        return [{"role": "user", "content": "Provide diagnosis in {Answer} format"}], new_state

    async def is_completed(self, messages, state, info=None):
        if not state.get("asked", False):
            return False  # Must ask for info first

        last = self._last_assistant_text(messages)
        # Check for brace-wrapped answer
        return "{" in last and "}" in last
```

## Best Practices

### 1. Data Format

Use JSONL format with consistent fields:

```json
{"question": "...", "answer": "...", "image": "..."}
{"question": "...", "answer": "...", "context": "..."}
```

### 2. Reward Design

- **Normalize** inputs for robust matching
- **Combine** multiple metrics for comprehensive evaluation
- **Weight** rewards based on task importance
- **Log** detailed scores for analysis

### 3. Environment Design

- **Limit turns** to prevent infinite episodes
- **Track state** for complex completion logic
- **Provide clear feedback** to guide the model
- **Handle errors** gracefully

### 4. Performance

- **Batch** reward computations
- **Cache** expensive operations
- **Use async** for I/O operations
- **Monitor** memory usage

## API Reference

### BaseMultiTurnEnv

```python
class BaseMultiTurnEnv:
    def __init__(
        self,
        cases: List[Dict] | None = None,
        *,
        dataset_path: str | None = None,
        max_turns: int = 10,
        name: str = "BaseRadiantEnv",
        log_dir: str | None = None,
    )
```

### Reward Functions

```python
class ExactMatchReward:
    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        strip_braces: bool = True,
    )

class TokenF1Reward:
    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        tokenize: str = "simple",
    )

class IoUReward:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        normalized: bool = True,
    )

class CombinedReward:
    def __init__(
        self,
        rewards: List[BaseRewardFunction],
        weights: List[float] | None = None,
        names: List[str] | None = None,
    )
```

### Adapter

```python
class RadiantHarnessAdapter:
    def __init__(
        self,
        processor: AgenticProcessorBase,
    )

    def create_environment_class(
        self,
        base_class: type[vf.MultiTurnEnv] | None = None,
        **env_kwargs: Any,
    ) -> type[vf.MultiTurnEnv]
```

## Troubleshooting

### Common Issues

1. **Import Error**: ensure core deps are installed (`uv sync` or `pip install -e .`)
2. **Memory Issues**: Reduce batch size, use streaming
3. **Slow Training**: Cache rewards, use async
4. **Poor Rewards**: Check normalization, adjust weights

### Debug Logging

Enable logging to track episodes:

```python
env = MyEnvironment(
    dataset_path="data.jsonl",
    log_dir="./logs",  # Creates debug.log
)
```

### Profile Performance

```python
import time

start = time.time()
results = evaluator.evaluate()
print(f"Evaluation time: {time.time() - start:.2f}s")
```

## Related Packages

- [verifiers](https://github.com/richzhang/verifiers): RL training framework
- [datasets](https://github.com/huggingface/datasets): Data loading utilities
- [transformers](https://github.com/huggingface/transformers): Model training

## Contributing

To contribute new reward functions or utilities:

1. Inherit from `BaseRewardFunction`
2. Add comprehensive docstrings
3. Include type hints
4. Write tests
5. Update this documentation
