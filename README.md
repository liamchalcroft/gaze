# Radiant Harness

[![Tests](https://github.com/your-org/radiant-harness/workflows/tests/badge.svg)](https://github.com/your-org/radiant-harness/actions)
[![Coverage](https://img.shields.io/codecov/c/github/your-org/radiant-harness)](https://codecov.io/gh/your-org/radiant-harness)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

A production-ready, modular framework for building multi-turn agentic vision-language model systems. Built for medical image analysis but versatile enough for any visual reasoning task.

## ✨ Features

- **🛠️ Tool System**: Extensible registry with built-in visual and search tools
- **🔄 Multi-turn Conversations**: Full agentic loop with tool calling support
- **🎯 Task-Specific Processors**: Easy dependency injection for custom workflows
- **📦 Multiple Model Adapters**: OpenAI, OpenRouter, and local HuggingFace models
- **🔍 Integrated Search**: PubMed literature and medical image search
- **⚡ Production Ready**: Retry logic, error handling, resource management
- **🧪 Verifiers Integration**: Reward functions for RL training

## 🚀 Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/your-org/radiant-harness.git
cd radiant-harness
uv sync  # or pip install -e .
```

### Basic Usage
```python
from pathlib import Path
from radiant_harness import AgenticProcessorBase, ImageInput, ToolRegistry
from radiant_harness import create_visual_tools, create_search_tools

# Define your task-specific processor
class MyProcessor(AgenticProcessorBase):
    def get_system_prompt(self, images, metadata):
        return "You are a medical imaging expert. Analyze the provided images."

    def get_user_message(self, images, metadata):
        return f"Analyze this scan. History: {metadata.get('history', '')}"

    def get_response_schema(self):
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "findings": {"type": "string"},
                    "continue": {"type": "boolean"}
                }
            }
        }

    def validate_response(self, response):
        return "findings" in response

# Run analysis
processor = MyProcessor(model_name="openai/gpt-4o", use_tools=True)
result = await processor.analyze(
    images=Path("scan.jpg"),
    metadata={"modality": "MRI", "history": "Patient presents with..."}
)

print(f"Findings: {result.final_response.get('findings')}")
print(f"Confidence: {result.confidence:.2f}")
```

## 📁 Project Structure

```
nova_retrieval_vlm/
├── src/radiant_harness/         # Core framework
│   ├── tools/                  # Tool system
│   ├── models/                 # Model adapters
│   ├── retrieval/              # Search integration
│   └── prompts/                # Template loading
├── examples/                   # Example implementations
│   ├── nova/                   # NOVA brain-MRI benchmark
│   ├── agentclinic_nejm/       # Diagnostic reasoning
│   ├── pubmedqa/              # Medical Q&A
│   └── vqa_rad/               # Radiology VQA
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## 🧪 Running Tests

```bash
# Core framework tests
uv run pytest

# Run with coverage
uv run pytest --cov=radiant_harness --cov-report=html

# Run specific test suites
uv run pytest tests/test_tool_registry.py
uv run pytest tests/test_base_processor.py
```

## 📚 Documentation

- [API Reference](docs/api.md)
- [Tool Development Guide](docs/tools.md)
- [Model Adapter Guide](docs/adapters.md)
- [Example: NOVA Benchmark](examples/nova/README.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🔧 Development

```bash
# Install dev dependencies
uv sync --group dev

# Code quality checks
uv run ruff check .
uv run ruff format .
uv run pyright

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

- 🛠️ New tools (segmentation, measurement, etc.)
- 🤖 Additional model adapters
- 📊 Evaluation metrics
- 📚 Documentation improvements
- 🐛 Bug fixes and performance optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with inspiration from the medical AI community and designed to accelerate research in vision-language models for healthcare.

