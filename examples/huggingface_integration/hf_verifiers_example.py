"""
Example: Using Hugging Face models with Verifiers for RL training.

This shows how to combine HF models with the verifiers package for
reinforcement learning fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer

# Import verifiers and harness utilities
from radiant_harness import AggressiveProcessorBase
from radiant_harness import HarnessConfig
from radiant_harness.verifiers import BaseMultiTurnEnv
from radiant_harness.verifiers import ExactMatchReward
from radiant_harness.adapters import AdapterProtocol


class HFModelAdapter(AdapterProtocol):
    """Simple HF model adapter for RL training."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Generate response with HF model."""
        # Format messages as prompt
        prompt = self.format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response, [], {}

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the model."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from multimodal content
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            prompt += f"{role}: {content}\n"
        prompt += "ASSISTANT:"
        return prompt


class SimpleRLProcessor(AggressiveProcessorBase):
    """Processor for RL training with HF models."""

    def __init__(self, model_name: str, **kwargs: Any):
        config = HarnessConfig(model_name=model_name, **kwargs)
        super().__init__(config)
        self.adapter = HFModelAdapter(model_name)

    def get_system_prompt(self, images, metadata) -> str:
        return "You are a helpful assistant. Provide accurate answers."

    def get_user_message(self, images, metadata) -> str:
        return metadata.get("question", "")

    async def _generate_response(self, prompt, images=None, temperature=0.7, **kwargs):
        messages = [
            {"role": "system", "content": self.get_system_prompt(images, kwargs)},
            {"role": "user", "content": prompt},
        ]
        return await self.adapter.generate_chat(
            messages,
            max_tokens=self.config.max_tokens,
            temperature=temperature,
        )


class SimpleQARLEnvironment(BaseMultiTurnEnv):
    """Simple QA environment for RL training."""

    def __init__(
        self,
        *,
        dataset_path: str,
        max_turns: int = 1,
        name: str = "SimpleQA",
    ) -> None:
        # Load rows from JSONL: {"question": ..., "answer": ...}
        rows: list[dict[str, Any]] = []
        with open(dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        prompts: list[list[dict[str, Any]]] = []
        infos: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            question = row.get("question", "")
            answer = row.get("answer", "")
            prompts.append([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"Question: {question}"},
            ])
            infos.append({"id": idx, "answer": answer})

        dataset = Dataset.from_dict({
            "id": list(range(len(rows))),
            "prompt": prompts,
            "info": infos,
        })

        super().__init__(name=name, dataset=dataset)
        self._max_turns = max_turns

    def get_system_prompt(self) -> str:
        return "Answer the question with a brief, accurate response."

    def _build_user_message(self, case: dict[str, Any]) -> str:
        return f"Question: {case['question']}"

    async def is_completed(
        self,
        messages,
        state,
        info=None
    ) -> bool:
        # Complete after 1 turn for this simple example
        return state.get("turn", 0) >= self._max_turns

    def build_initial_state(self, prompt, info):
        return {"turn": 0, "info": info}

    async def env_response(self, messages, state, info=None):
        new_state = dict(state)
        new_state["turn"] = state.get("turn", 0) + 1
        # No additional environment message; just advance turn
        return [], new_state


class HuggingFaceTrainer:
    """Custom trainer for HF models with verifiers rewards."""

    def __init__(
        self,
        model_name: str,
        environment: BaseMultiTurnEnv,
        reward_fn,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.environment = environment
        self.reward_fn = reward_fn

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Training config
        self.training_args = TrainingArguments(
            output_dir="./hf_rl_results",
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=3,
            save_steps=500,
            logging_steps=100,
            evaluation_strategy="no",
            remove_unused_columns=False,
        )

    def prepare_dataset(self, dataset_path: str) -> Dataset:
        """Prepare dataset for training."""
        # Load raw data
        with open(dataset_path, encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f if line.strip()]

        # Convert to training format
        training_data = []
        for item in raw_data:
            # Format as instruction-response pair
            prompt = f"Question: {item['question']}\nAnswer:"
            response = item["answer"]

            # Tokenize
            inputs = self.tokenizer(
                prompt + response,
                truncation=True,
                max_length=512,
                padding=False,
            )

            # Create labels
            labels = inputs["input_ids"].copy()
            # Mask prompt in labels
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            labels[:prompt_len] = -100

            training_data.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels,
            })

        return Dataset.from_list(training_data)

    def compute_rewards(self, predictions: list[str], references: list[str]) -> list[float]:
        """Compute rewards for predictions."""
        rewards = []
        for pred, ref in zip(predictions, references):
            reward = self.reward_fn("", pred, {"answer": ref})
            rewards.append(reward)
        return rewards

    def train(self, dataset_path: str) -> None:
        """Train the model with RL."""
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset_path)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )

        # Train
        trainer.train()

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom data collator."""
        # Pad sequences
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            # Pad input_ids
            padded = item["input_ids"] + [self.tokenizer.pad_token_id] * (max_len - len(item["input_ids"]))
            input_ids.append(padded)

            # Pad attention_mask
            mask = item["attention_mask"] + [0] * (max_len - len(item["attention_mask"]))
            attention_mask.append(mask)

            # Pad labels
            label = item["labels"] + [-100] * (max_len - len(item["labels"]))
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    def evaluate_with_rewards(self, eval_dataset_path: str) -> dict[str, float]:
        """Evaluate model with reward functions."""
        # Load eval data
        with open(eval_dataset_path, encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f if line.strip()]

        # Generate predictions
        predictions = []
        references = []

        self.model.eval()
        with torch.no_grad():
            for item in eval_data:
                # Format prompt
                prompt = f"Question: {item['question']}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Decode response
                response_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                predictions.append(response)
                references.append(item["answer"])

        # Compute rewards
        rewards = self.compute_rewards(predictions, references)

        # Return metrics
        return {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "num_samples": len(rewards),
        }


def main() -> None:
    """Main example showing HF + verifiers integration."""
    print("Hugging Face + Verifiers RL Training Example")
    print("=" * 50)

    # Create sample data
    sample_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        {"question": "What is the largest planet?", "answer": "Jupiter"},
        {"question": "What year did WW2 end?", "answer": "1945"},
    ]

    # Save sample data
    data_dir = Path("./hf_rl_data")
    data_dir.mkdir(exist_ok=True)

    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in sample_data[:4]:
            f.write(json.dumps(item) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data[4]) + "\n")

    print(f"Created sample data in {data_dir}")

    # Create environment
    env = SimpleQARLEnvironment(
        dataset_path=str(train_path),
        max_turns=1,
        name="SimpleQA",
    )

    # Create reward function
    reward_fn = ExactMatchReward(normalize=True)

    # Create trainer
    trainer = HuggingFaceTrainer(
        model_name="gpt2",  # Small model for demo
        environment=env,
        reward_fn=reward_fn,
        learning_rate=5e-5,
        batch_size=2,
    )

    print("\nSetup complete!")
    print("To run training:")
    print("```python")
    print("trainer.train(str(train_path))")
    print("```")
    print("\nTo evaluate:")
    print("```python")
    print("metrics = trainer.evaluate_with_rewards(str(eval_path))")
    print("print(f'Mean reward: {metrics[\"mean_reward\"]:.3f}')")
    print("```")

    # Show example with verifiers
    print("\n--- Verifiers Integration Example ---")
    print("""
# Verifiers is included with the core install
import verifiers as vf

# Wrap processor for verifiers
from radiant_harness.verifiers import wrap_processor_for_verifiers

processor = SimpleRLProcessor("gpt2")
env = wrap_processor_for_verifiers(processor, max_turns=1)

# Train with verifiers
trainer = vf.RLTrainer(
    environment=env,
    model="gpt2",
    reward_rubric=vf.Rubric(funcs=[reward_fn], weights=[1.0]),
    learning_rate=5e-5,
    batch_size=2,
)
trainer.train()
    """)


if __name__ == "__main__":
    main()