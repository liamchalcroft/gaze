# AgentClinic NEJM example

Multi-turn diagnostic reasoning with standardized patient simulation.

## Overview

Implements the AgentClinic NEJM environment using the [verifiers](https://github.com/primeintellect-ai/verifiers) package for multi-turn clinical case evaluation. The model acts as a clinician: it gathers information (history, exam and tests, imaging) from a simulated patient, then commits to a final diagnosis wrapped in braces, e.g. `{Pneumonia}`. An episode completes only after at least one information request and a brace-wrapped diagnosis; a premature diagnosis triggers a nudge and the episode continues.

Based on [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic).

## Dataset

- Source: New England Journal of Medicine (NEJM) clinical cases
- Format: extended JSONL with patient information, images, and answer choices
- Access: the dataset is NOT shipped with this repository. Run the download script below before the first evaluation; it fetches `agentclinic_nejm_extended.jsonl` from the upstream AgentClinic repository into `data/` (gitignored).

```bash
cd examples/agentclinic_nejm/data
python download.py
```

`download.py` is required: without it the default command below raises `FileNotFoundError`.

## Install

Run from the repository root:

```bash
uv sync --extra agentclinic
# or
pip install gaze-vlm[agentclinic]
```

## Run

```bash
uv run python -m examples.agentclinic_nejm.eval \
  --dataset ./examples/agentclinic_nejm/data/agentclinic_nejm_extended.jsonl \
  --model openai/gpt-4o \
  --max-turns 10 \
  --num-samples 10 \
  --output ./results
```

`--model openai/...` runs go through OpenRouter/OpenAI: set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) first, or the run fails with "No API key found". The local LM Studio path via `--base-url` (below) needs no key.

Omit `--dataset` to use the default path (`data/agentclinic_nejm_extended.jsonl`), which is where `download.py` writes the file. To load the multi-turn environment directly (for example, to inspect cases or wire up training):

```python
from examples.agentclinic_nejm.src import load_environment

env = load_environment(
    dataset_path="./examples/agentclinic_nejm/data/agentclinic_nejm_extended.jsonl",
    max_turns=10,
)
print(f"Loaded {len(env.dataset)} clinical cases")
```

## Run locally (LM Studio)

`run_local.sh` runs the multi-turn benchmark against a local OpenAI-compatible server. Pass `--base-url` (default `http://localhost:1234/v1`):

```bash
./examples/agentclinic_nejm/run_local.sh qwen3.5-35b-a3b http://localhost:1234/v1 50
```

Conversation history grows each turn, so use `n_ctx >= 8192`; thinking models need headroom, hence `--max-tokens 8192`. NEJM cases may include an image: a vision model is needed to interpret IMAGE responses, though text-only models still run. Only load one model in LM Studio at a time (the health-check probe can trigger model swapping on memory-constrained GPUs).

## Flags

- `--dataset PATH`: NEJM JSONL dataset (default: `data/agentclinic_nejm_extended.jsonl`, written by `download.py`)
- `--model NAME`: model name (OpenAI/OpenRouter format, or local ID for `--base-url`)
- `--base-url URL`: OpenAI-compatible server, e.g. `http://localhost:1234/v1`
- `--max-turns N`: maximum conversation turns (default 10)
- `--num-samples N`: number of samples (-1 for all)
- `--max-tokens N`: maximum completion tokens (default 4096)
- `--temperature F`: sampling temperature (default 0.0)
- `--reasoning`: enable reasoning mode for OpenAI/OpenRouter
- `--seed N`: random seed for reproducibility
- `--output PATH`: directory for summary output (default `./results`)
- `--verbose`: verbose logs

## Output

- `agentclinic_eval_<model>.json` in the chosen `--output` directory: per-case predictions and aggregate accuracy / token-F1
- The environment also logs scoring details to `src/log/debug.log`:

```
[score] raw_gold='Exogenous ochronosis' raw_pred='Exogenous ochronosis'
norm_gold='exogenous ochronosis' norm_pred='exogenous ochronosis' ok=True
```

## Interaction flow

The framework drives the conversation with JSON tool-calling under the hood. The exchange below is an illustrative paraphrase of the turn flow, not the literal JSON the model emits:

```
System: Think step by step. When ready, output ONE diagnosis inside braces.
User:   [Case description with chief complaint and answer choices]

Assistant: I need to gather more information. What is the patient's medical history?
           -> requests HISTORY

Environment: Patient History
             [Patient medical history details]

Assistant: What do the physical examinations and test results show?
           -> requests EXAM_AND_TESTS

Environment: Physical Examination and Test Results
             [Exam results and clinical findings]

Assistant: Let me see the medical image if available.
           -> requests IMAGE

Environment: [Image, or "No medical image is available for this case."]

Assistant: Based on the patient's presentation with [symptoms] and [findings],
           {Pneumonia}
```

## Reward function

Combined reward: `0.8 * accuracy + 0.2 * token-F1`.

Accuracy reward:
- 1.0: normalized prediction matches gold answer
- 0.0: incorrect or missing diagnosis

Normalization: lowercase, strip braces/punctuation, match against answer options.

## Structure

```
agentclinic_nejm/
    src/
        __init__.py              # Package exports
        environment.py           # MultiTurnEnv + reward functions
    data/
        download.py              # Dataset download script (run this first)
        agentclinic_nejm_extended.jsonl  # Produced by download.py (gitignored, not shipped)
    tests/                       # Hermetic smoke tests
    train.py                     # Training-config prep template (see note)
    eval.py                      # Runnable evaluation loop
    run_local.sh                 # Local (LM Studio) evaluation
    README.md
```

`train.py` is a training-integration template, not a standalone trainer: it builds the environment, validates settings, and writes a `config.json` for you to pass to your own verifiers training loop. It does not run gradient updates.

## References

- [AgentClinic repository](https://github.com/SamuelSchmidgall/AgentClinic)
- [NEJM clinical cases](https://www.nejm.org/medical-education/clinical-cases)
- [verifiers package](https://github.com/primeintellect-ai/verifiers)
- [GAZE](https://github.com/liamchalcroft/gaze)

## License

Follows the license terms of the AgentClinic project and NEJM clinical cases for educational use.
