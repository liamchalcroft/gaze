"""
Hugging Face model adapter for Radiant Harness.

This example shows how to use Hugging Face models (both text and vision-language)
directly with Radiant Harness for inference and fine-tuning.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from transformers import AutoTokenizer

import asyncio
from pathlib import Path
from typing import Any

from radiant_harness import AgenticProcessorBase
from radiant_harness.models import AdapterProtocol
from radiant_harness.types import GenerationLog


class HuggingFaceTextAdapter(AdapterProtocol):
    """Adapter for Hugging Face text generation models."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        """Initialize HF text model adapter.

        Args:
            model_name: HF model identifier (e.g., "microsoft/DialoGPT-medium")
            device: Device to run on ("cuda", "cpu", or "auto")
            torch_dtype: Data type ("auto", "float16", "bfloat16")
        """
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.device = self._get_device(device)
        dtype = self._get_dtype(torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device if device != "auto" else None,
        )

        if device == "auto":
            self.model = self.model.to(self.device)

    def _get_device(self, device: str) -> str | torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _get_dtype(self, dtype: str) -> torch.dtype:
        """Get appropriate torch dtype."""
        if dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]], GenerationLog]:
        """Generate response using HF model."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)

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

        # Decode response
        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Log generation
        log = GenerationLog(
            model=self.model_name,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=len(response_ids),
            total_tokens=inputs["input_ids"].shape[1] + len(response_ids),
            temperature=temperature,
        )

        return response, [], log

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages to prompt format."""
        # Simple chat format - customize as needed
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal - extract text
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            prompt += f"{role}: {content}\n"
        prompt += "ASSISTANT:"
        return prompt


class HuggingFaceVLMAdapter(AdapterProtocol):
    """Adapter for Hugging Face Vision-Language Models."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        """Initialize HF VLM adapter.

        Args:
            model_name: HF model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
            device: Device to run on
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name

        # Load processor (tokenizer + image processor)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load model
        self.device = self._get_device(device)
        dtype = self._get_dtype(torch_dtype)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device if device != "auto" else None,
        )

        if device == "auto":
            self.model = self.model.to(self.device)

    def _get_device(self, device: str) -> str | torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _get_dtype(self, dtype: str) -> torch.dtype:
        """Get appropriate torch dtype."""
        if dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]], GenerationLog]:
        """Generate response using HF VLM."""
        from PIL import Image

        # Extract text from messages
        prompt = self._extract_text_prompt(messages)

        # Process images if provided
        image_inputs = None
        if images:
            # Load images
            pil_images = [Image.open(img) for img in images]
            image_inputs = self.processor(images=pil_images, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Tokenize text
        text_inputs = self.processor(text=prompt, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Combine inputs
        if image_inputs:
            inputs = {
                **text_inputs,
                **image_inputs,
            }
        else:
            inputs = text_inputs

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode response
        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Log generation
        log = GenerationLog(
            model=self.model_name,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=len(response_ids),
            total_tokens=inputs["input_ids"].shape[1] + len(response_ids),
            temperature=temperature,
            has_images=len(images) if images else 0,
        )

        return response, [], log

    def _extract_text_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Extract text content from messages."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text, skip images
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = "\n".join(text_parts)
                if any(item.get("type") == "image_url" for item in msg.get("content", [])):
                    content = f"<image>\n{content}"
            prompt += f"{role}: {content}\n"
        prompt += "ASSISTANT:"
        return prompt


# Example processors using HF adapters
class HFTextProcessor(AgenticProcessorBase):
    """Example processor using HF text model."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        torch_dtype: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.adapter = HuggingFaceTextAdapter(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        super().__init__(
            model_name=model_name,
            use_tools=False,
            adapter_factory=lambda: self.adapter,
            **kwargs,
        )

    def get_system_prompt(self, images, metadata) -> str:
        return "You are a helpful assistant."

    def get_user_message(self, images, metadata) -> str:
        return metadata.get("question", "")

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        return isinstance(response, (dict, str))

    def process(self, question: str, images: list[str] | None = None):
        """Run a blocking analysis for convenience."""
        image_paths = [Path(p) for p in images] if images else None
        return asyncio.run(self.analyze(images=image_paths, metadata={"question": question}))


class HFVLMProcessor(AgenticProcessorBase):
    """Example processor using HF vision-language model."""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "auto",
        torch_dtype: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.adapter = HuggingFaceVLMAdapter(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        super().__init__(
            model_name=model_name,
            use_tools=False,
            adapter_factory=lambda: self.adapter,
            **kwargs,
        )

    def get_system_prompt(self, images, metadata) -> str:
        return "You are a helpful vision-language assistant. Analyze images and answer questions."

    def get_user_message(self, images, metadata) -> str:
        msg = metadata.get("question", "")
        if images:
            msg = f"<image>\n{msg}"
        return msg

    def get_response_schema(self) -> dict[str, Any] | None:
        return None

    def validate_response(self, response: dict[str, Any]) -> bool:
        return isinstance(response, (dict, str))

    def process(self, question: str, images: list[str] | None = None):
        """Run a blocking analysis for convenience."""
        image_paths = [Path(p) for p in images] if images else None
        return asyncio.run(self.analyze(images=image_paths, metadata={"question": question}))


# Usage examples
def example_text_model() -> None:
    """Example using a text-only model."""
    processor = HFTextProcessor(
        model_name="microsoft/DialoGPT-medium",
        max_tokens=512,
    )

    # Run inference
    result = processor.process(
        question="What is the capital of France?",
        images=None,
    )
    print(f"Response: {result.final_response}")


def example_vlm_model() -> None:
    """Example using a vision-language model."""
    processor = HFVLMProcessor(
        model_name="llava-hf/llava-1.5-7b-hf",
        max_tokens=512,
    )

    # Run inference with image
    result = processor.process(
        question="What do you see in this image?",
        images=["path/to/image.jpg"],
    )
    print(f"Response: {result.final_response}")


def example_fine_tuning() -> None:
    """Example of fine-tuning setup."""
    from transformers import TrainingArguments
    from transformers import Trainer

    # Load model and processor
    adapter = HuggingFaceVLMAdapter("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Prepare dataset
    # dataset = load_dataset("your-dataset")

    # Configure training
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        save_steps=1000,
        logging_steps=100,
    )

    # Create trainer
    trainer = Trainer(
        model=adapter.model,
        args=training_args,
        # train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        # data_collator=your_collator_function,
    )

    # Fine-tune
    # trainer.train()


if __name__ == "__main__":
    print("Hugging Face Integration Examples")
    print("=" * 40)

    print("\n1. Text Model Example")
    example_text_model()

    print("\n2. Vision-Language Model Example")
    example_vlm_model()

    print("\n3. Fine-tuning Setup")
    print("See example_fine_tuning() function")