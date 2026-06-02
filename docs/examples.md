# Examples

GAZE includes five complete example applications demonstrating different use cases.

## NOVA Brain MRI

**Location:** `examples/nova/`

Brain MRI analysis with three sub-tasks: caption generation, diagnosis prediction, and lesion localization. Uses the [NOVA dataset](https://huggingface.co/datasets/c-i-ber/Nova) which auto-downloads from HuggingFace.

```bash
pip install gaze-vlm[nova]

# Single-turn mode
uv run python -m examples.nova.src.cli \
  --model openai/gpt-4o \
  --mode single_turn \
  --max-samples 10

# Agentic mode with tools
uv run python -m examples.nova.src.cli \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --max-turns 5 \
  --max-samples 10
```

## GEMeX Visual Grounding

**Location:** `examples/gemex_thinkvg/`

Visual grounding with chain-of-thought reasoning on chest X-rays. Requires MIMIC-CXR access (PhysioNet credentialed).

```bash
pip install gaze-vlm[gemex]
```

## AgentClinic NEJM

**Location:** `examples/agentclinic_nejm/`

Multi-turn diagnostic reasoning where the model gathers clinical information (history, exam, tests, imaging) before making a diagnosis.

```bash
pip install gaze-vlm[agentclinic]
```

## PubMedQA

**Location:** `examples/pubmedqa/`

Text-only medical Q&A with yes/no/maybe answers. Uses the [PubMedQA dataset](https://huggingface.co/datasets/qiaojin/PubMedQA) (auto-downloads).

```bash
pip install gaze-vlm[pubmedqa]

uv run python -m examples.pubmedqa.src.cli \
  --model openai/gpt-4o \
  --mode single_turn \
  --max-samples 50
```

## VQA-RAD

**Location:** `examples/vqa_rad/`

Radiology visual question answering with closed and open-ended questions. Uses the [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (auto-downloads).

```bash
pip install gaze-vlm[vqa-rad]

uv run python -m examples.vqa_rad.src.cli \
  --model openai/gpt-4o \
  --mode agentic \
  --use-tools \
  --max-samples 20
```

## Local Models

All examples support local model inference via LM Studio. Pass `--base-url` to point at your instance:

```bash
uv run python -m examples.pubmedqa.src.cli \
  --model qwen3.5-a3b \
  --base-url http://localhost:1234/v1 \
  --mode single_turn \
  --max-samples 5
```

## Writing Your Own Example

See [Getting Started](getting-started.md) for the pattern. The examples vary
in size, but the NOVA example (`examples/nova/src/`) is representative. Note
that `evaluation/` is a package, not a single module, and dataset loading
lives under `data/`:

```
examples/your_task/
    src/
        __init__.py
        cli.py           # CLI entry point (argparse + run_evaluation)
        config.py        # Frozen config dataclass for the task
        processor.py     # AgenticProcessorBase subclass
        schemas.py       # Response schema(s) and validate_response()
        data/            # Dataset loading package
            __init__.py
        evaluation/      # Metrics package, one module per sub-task
            __init__.py
            caption.py
            detection.py
            diagnosis.py
    run_local.sh         # LM Studio convenience script
    README.md
```

Smaller examples (for instance `pubmedqa`) collapse some of these into fewer
modules. The only hard requirement is a processor that subclasses
`AgenticProcessorBase` and implements the four abstract methods.
