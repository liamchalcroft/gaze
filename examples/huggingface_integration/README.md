# Hugging Face Integration

This directory shows how to use Hugging Face models directly with Radiant Harness for both inference and training.

## Overview

Radiant Harness is model-agnostic and works with any Hugging Face model through the adapter pattern. This integration provides:

- **Text Models**: GPT-2, T5, BLOOM, Llama, etc.
- **Vision-Language Models**: LLaVA, BLIP, Fuyu, etc.
- **Fine-tuning Support**: Both supervised and RL fine-tuning
- **Verifiers Integration**: For RL training with custom rewards

## Quick Start

### 1. Install Dependencies

```bash
# Core install (verifiers included by default)
uv sync
# or
pip install -e .

# HF model extras
pip install transformers torch datasets accelerate
```

### 2. Use a Text Model

```python
from examples.huggingface_integration.hf_model_adapter import HFTextProcessor

# Create processor with GPT-2
processor = HFTextProcessor(
    model_name="gpt2",
    max_tokens=512,
)

# Run inference
result = processor.process(
    question="What is the capital of France?",
    images=None,
)
print(result.response)
```

### 3. Use a Vision-Language Model

```python
from examples.huggingface_integration.hf_model_adapter import HFVLMProcessor

# Create processor with LLaVA
processor = HFVLMProcessor(
    model_name="llava-hf/llava-1.5-7b-hf",
    max_tokens=512,
)

# Run inference with image
result = processor.process(
    question="What do you see in this image?",
    images=["path/to/image.jpg"],
)
print(result.response)
```

## Supported Model Types

### Text Generation Models

```python
# Chat models
models = [
    "microsoft/DialoGPT-medium",
    "facebook/blenderbot-400M-distill",
    "google/flan-t5-base",
    "meta-llama/Llama-2-7b-chat-hf",
]

processor = HFTextProcessor(model_name=models[0])
```

### Vision-Language Models

```python
# VLMs
models = [
    "llava-hf/llava-1.5-7b-hf",
    "Salesforce/blip-vqa-base",
    "microsoft/Florence-2-base",
]

processor = HFVLMProcessor(model_name=models[0])
```

## Training

### Supervised Fine-Tuning

```python
from transformers import TrainingArguments, Trainer

# Load model and processor
adapter = HuggingFaceVLMAdapter("llava-hf/llava-1.5-7b-hf")

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=3,
)

trainer = Trainer(
    model=adapter.model,
    args=training_args,
    train_dataset=your_dataset,
)

trainer.train()
```

### RL Fine-Tuning with Verifiers

```python
import verifiers as vf
from radiant_harness.verifiers import wrap_processor_for_verifiers
from radiant_harness.verifiers import ExactMatchReward

# Create processor
processor = HFVLMProcessor("llava-hf/llava-1.5-7b-hf")

# Wrap for verifiers
env = wrap_processor_for_verifiers(
    processor,
    max_turns=5,
    name="VLM_RL",
)

# Create reward function
reward_fn = ExactMatchReward(normalize=True)

# Train with RL
trainer = vf.RLTrainer(
    environment=env,
    model="llava-hf/llava-1.5-7b-hf",
    reward_rubric=vf.Rubric(funcs=[reward_fn], weights=[1.0]),
    learning_rate=1e-5,
    batch_size=2,
    max_rollouts=4,
)

trainer.train()
```

## Custom Model Adapter

Create a custom adapter for any HF model:

```python
from radiant_harness.models import AdapterProtocol

class CustomAdapter(AdapterProtocol):
    def __init__(self, model_name: str):
        # Load your model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def generate_chat(
        self,
        messages,
        max_tokens=1024,
        temperature=0.7,
        **kwargs
    ):
        # Convert messages to model format
        prompt = self.format_messages(messages)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # Decode and return
        response = self.tokenizer.decode(outputs[0])
        return response, [], log
```

## Examples

### 1. Medical VQA

```python
# Use a medical VLM for radiology questions
processor = HFVLMProcessor("microsoft/BioGPT-Large")

result = processor.process(
    question="Is there evidence of pneumonia in this chest X-ray?",
    images=["chest_xray.jpg"],
)
```

### 2. Multi-turn Dialogue

```python
from radiant_harness.verifiers import BaseMultiTurnEnv

class DialogueEnv(BaseMultiTurnEnv):
    def get_system_prompt(self):
        return "You are a helpful medical assistant."

    def _build_user_message(self, case):
        return f"Patient: {case['patient_query']}"

    async def is_completed(self, messages, state, info=None):
        # End when assistant provides diagnosis
        last = self._last_assistant_text(messages)
        return "diagnosis" in last.lower()

# Wrap processor for multi-turn
env = wrap_processor_for_verifiers(
    processor=HFTextProcessor("microsoft/DialoGPT-medium"),
    max_turns=5,
)
```

### 3. Custom Reward Function

```python
from radiant_harness.verifiers import BaseRewardFunction

class MedicalAccuracyReward(BaseRewardFunction):
    def __call__(self, prompt, completion, info):
        # Custom medical accuracy logic
        prediction = self.extract_diagnosis(completion)
        reference = info.get("diagnosis", "")

        # Use medical knowledge base
        score = self.evaluate_medical_accuracy(prediction, reference)
        return score

# Use in RL training
reward_fn = MedicalAccuracyReward()
trainer = vf.RLTrainer(
    environment=env,
    reward_rubric=vf.Rubric(funcs=[reward_fn]),
)
```

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for larger models
2. **Quantization**: Use `torch_dtype="float16"` for memory efficiency
3. **Batching**: Process multiple examples together when possible
4. **Caching**: Cache tokenized inputs for repeated prompts
5. **LoRA**: Use PEFT/LoRA for parameter-efficient fine-tuning

```python
# LoRA fine-tuning example
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller models or quantization
2. **Slow Generation**: Use GPU or reduce sequence length
3. **Poor Quality**: Adjust temperature and top_p parameters
4. **Format Issues**: Customize message formatting for your model

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model outputs
adapter = HFModelAdapter("gpt2")
messages = [{"role": "user", "content": "Hello"}]
response, tool_calls, log = await adapter.generate_chat(messages)
print(f"Response: {response}")
print(f"Log: {log}")
```

## Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (LoRA)](https://huggingface.co/docs/peft/)
- [Verifiers Documentation](https://docs.primeintellect.ai/verifiers)
- [Radiant Harness Docs](../../../docs/verifiers_integration.md)