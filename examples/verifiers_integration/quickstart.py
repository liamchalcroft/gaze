#!/usr/bin/env python
"""Quickstart example for verifiers integration.

Shows how to use the new verifiers utilities with Radiant Harness
for multi-turn RL training.
"""

from __future__ import annotations

import json
from pathlib import Path

# NOTE: Verifiers is a core dependency; ensure your environment is installed
# (e.g., `uv sync` or `pip install -e .`).

try:
    import verifiers as vf
except ImportError as exc:
    raise ImportError(
        "verifiers package is required. It ships as a core dependency; "
        "run `uv sync` or `pip install -e .` if your environment is missing it."
    ) from exc

from radiant_harness import AgenticProcessorBase
from radiant_harness.verifiers import BaseMultiTurnEnv
from radiant_harness.verifiers import ExactMatchReward
from radiant_harness.verifiers import TokenF1Reward
from radiant_harness.verifiers import CombinedReward
from radiant_harness.verifiers import RadiantHarnessAdapter
from radiant_harness.verifiers import wrap_processor_for_verifiers


class SimpleProcessor(AgenticProcessorBase):
    """Simple processor for demonstration."""

    def get_system_prompt(self, images, metadata) -> str:
        return "You are a helpful assistant. Answer questions accurately."

    def get_user_message(self, images, metadata) -> str:
        question = metadata.get("question", "What do you see?")
        if images:
            question += "\nAnalyze the provided image."
        return question

    def get_response_schema(self) -> dict | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["answer"],
            },
        }

    def validate_response(self, response) -> bool:
        return isinstance(response, dict) and "answer" in response


def example_1_basic_environment() -> None:
    """Example 1: Basic multi-turn environment."""
    print("\n=== Example 1: Basic Multi-Turn Environment ===")

    # Create a simple dataset
    cases = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "type": "math",
        },
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "type": "geography",
        },
    ]

    # Create environment
    env = BaseMultiTurnEnv(
        cases=cases,
        max_turns=3,
        name="SimpleQA",
    )

    print(f"Created environment with {len(env.dataset)} samples")

    # Check a sample
    sample = env.dataset[0]
    print(f"Sample 0 prompt: {sample['prompt'][1]['content'][:50]}...")
    print(f"Sample 0 info: {sample['info']}")


def example_2_reward_functions() -> None:
    """Example 2: Using reward functions."""
    print("\n=== Example 2: Reward Functions ===")

    # Create reward functions
    exact_reward = ExactMatchReward(
        normalize=True,
        case_sensitive=False,
    )

    f1_reward = TokenF1Reward(
        normalize=True,
        tokenize="word",
    )

    # Test rewards
    prompt = "What is 2 + 2?"
    completion = "The answer is 4."
    info = {"answer": "4"}

    exact_score = exact_reward(prompt, completion, info)
    f1_score = f1_reward(prompt, completion, info)

    print(f"Exact match reward: {exact_score:.2f}")
    print(f"Token F1 reward: {f1_score:.2f}")

    # Combine rewards
    combined = CombinedReward(
        rewards=[exact_reward, f1_reward],
        weights=[0.6, 0.4],
    )

    combined_score = combined(prompt, completion, info)
    details = info.get("_reward_details", {})
    print(f"Combined reward: {combined_score:.2f}")
    print(f"Reward details: {details}")


def example_3_processor_adapter() -> None:
    """Example 3: Adapting a processor for verifiers."""
    print("\n=== Example 3: Processor Adapter ===")

    # Create processor
    processor = SimpleProcessor(
        model_name="gpt-4o-mini",
        use_tools=False,
        max_turns=3,
    )

    # Quick wrapper
    env = wrap_processor_for_verifiers(
        processor,
        max_turns=3,
        name="ProcessorEnv",
    )

    print(f"Wrapped processor as environment: {env.__class__.__name__}")

    # Or create adapter manually
    adapter = RadiantHarnessAdapter(processor)
    print(f"Created adapter: {adapter.__class__.__name__}")


def example_4_complete_setup() -> None:
    """Example 4: Complete training setup."""
    print("\n=== Example 4: Complete Training Setup ===")

    # Create dataset
    dataset_path = Path(__file__).parent / "data" / "sample_qa.jsonl"
    dataset_path.parent.mkdir(exist_ok=True)

    # Create sample dataset
    sample_data = [
        {
            "question": "What is the capital of Japan?",
            "answer": "Tokyo",
            "type": "geography",
        },
        {
            "question": "What is H2O?",
            "answer": "Water",
            "type": "chemistry",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "Shakespeare",
            "type": "literature",
        },
    ]

    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")

    print(f"Created sample dataset: {dataset_path}")

    # Create processor
    processor = SimpleProcessor(
        model_name="gpt-4o-mini",
        use_tools=False,
        max_turns=3,
    )

    # Create custom environment class
    adapter = RadiantHarnessAdapter(processor)

    class QAEnvironment(BaseMultiTurnEnv):
        """Simple Q&A environment."""

        def get_system_prompt(self) -> str:
            return "You are a knowledgeable assistant. Provide accurate, concise answers."

        def _build_user_message(self, case: dict[str, Any]) -> str:
            return f"Question: {case['question']}\n\nType: {case['type']}"

        async def is_completed(
            self,
            messages: vf.Messages,
            state: vf.State,
            info: dict[str, Any] | None = None,
        ) -> bool:
            # End at max turns or if we have a substantial answer
            if state.get("turn", 0) >= self._max_turns:
                return True

            last_asst = self._last_assistant_text(messages)
            # Simple heuristic: if answer has more than 5 words, likely complete
            if len(last_asst.split()) > 5:
                return True

            return False

    # Create environment
    env = QAEnvironment(
        dataset_path=str(dataset_path),
        max_turns=3,
        name="SimpleQA",
    )

    # Create reward function
    reward_fn = ExactMatchReward(
        normalize=True,
        case_sensitive=False,
    )

    # Create rubric
    rubric = adapter.create_verifiers_rubric(
        reward_functions=[reward_fn],
        weights=[1.0],
    )

    print(f"Environment: {env.name}")
    print(f"Dataset size: {len(env.dataset)}")
    print(f"Reward function: {reward_fn.__class__.__name__}")

    # Show training setup
    print("\nTraining Setup:")
    print("```python")
    print("import verifiers as vf")
    print("")
    print("# Train with RL")
    print("trainer = vf.RLTrainer(")
    print("    environment=env,")
    print("    model='your-model',")
    print("    reward_rubric=rubric,")
    print("    learning_rate=1e-5,")
    print("    batch_size=16,")
    print("    epochs=3,")
    print(")")
    print("trainer.train()")
    print("```")


def main() -> None:
    """Run all examples."""
    print("Radiant Harness + Verifiers Integration Examples")
    print("=" * 50)

    # Run examples
    example_1_basic_environment()
    example_2_reward_functions()
    example_3_processor_adapter()
    example_4_complete_setup()

    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nNext steps:")
    print("1. Ensure dependencies are installed (verifiers is core): `uv sync` or `pip install -e .`")
    print("2. Set up your API keys for model access")
    print("3. Run training with your own dataset")
    print("\nSee documentation for more details.")


if __name__ == "__main__":
    main()