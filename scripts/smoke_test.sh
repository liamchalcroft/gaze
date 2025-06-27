#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# smoke_test.sh – quick end-to-end sanity check                                  
# -----------------------------------------------------------------------------
# This helper runs the NOVA pipeline on a **handful of images** (default 5) for
# every (task × model) combination so you can confirm that both the prompts and
# the adapters work before launching the full benchmark.
#
# Usage (from repo root):
#   bash scripts/smoke_test.sh
#
# Optional environment variables:
#   MODEL_LIST   – space-separated slugs understood by OpenRouter / OpenAI SDK
#                  Default: "google/gemma-3-4b-it:free"
#   NUM_IMAGES   – number of samples per task (default: 5)
#   TASKS        – space-separated list of tasks (default: caption diagnosis localization)
#
# Note: the script uses *baseline* prompts without retrieval augmentation for
# speed.  If you want to include retrieval, set USE_RETRIEVAL=true.
# -----------------------------------------------------------------------------

set -euo pipefail

# Number of samples per run – keep very small for speed
NUM_IMAGES="${NUM_IMAGES:-3}"
# Tasks to exercise.  Adjust via `TASKS="caption diagnosis" bash ...` if desired.
TASKS=( ${TASKS:-caption diagnosis localization} )
# Which controller pipelines to run.  Supported values:
#   baseline            – single-turn, no retrieval
#   retrieval           – baseline + guideline retrieval (use_retrieval=true)
#   multiturn           – text-based multi-turn reasoning
#   visual_multiturn    – multi-turn with image ops loop
# Override via `APPROACHES="baseline visual_multiturn"`.
APPROACHES=( ${APPROACHES:-baseline retrieval multiturn visual_multiturn} )

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./model_list.sh
source "${SCRIPT_DIR}/model_list.sh"

# If the caller didn't specify MODEL_LIST, inherit MODELS_DEFAULT from the
# shared list.  Allow space-separated override via environment variable.
if [[ -z "${MODEL_LIST:-}" ]]; then
  MODEL_LIST="${MODELS_DEFAULT[*]}"
fi

ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "$ROOT_DIR" || exit 1

echo "[smoke] Running sanity checks for tasks: ${TASKS[*]} on models: ${MODEL_LIST}" 1>&2
echo "[smoke] Approaches: ${APPROACHES[*]}  |  images per combo: ${NUM_IMAGES}" 1>&2

ts="$(date +%s)"
for model in ${MODEL_LIST}; do
  safe_model="${model//\//_}"  # slash → underscore for folder names
  for task in "${TASKS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
      echo "\n=== [smoke] Task: $task | Model: $model | Approach: $approach ===" 1>&2

      # Map the shorthand "retrieval" to baseline+retrieval flag.
      extra_args=()
      real_approach="$approach"
      if [[ "$approach" == "retrieval" ]]; then
        real_approach="baseline"
        extra_args+=(use_retrieval=true)
      fi

      python -m nova_retrieval_vlm.cli \
        task="$task" \
        model.name="$model" \
        approach="$real_approach" \
        batch_size=1 \
        max_iterations="$NUM_IMAGES" \
        paths.output_dir="runs/smoke_${ts}/${safe_model}_${task}_${approach}" \
        ${extra_args[@]+"${extra_args[@]}"} \
        || { echo "[smoke] Failure for $model / $task / $approach" 1>&2; exit 1; }
    done
  done
done

echo "\n[smoke] All checks completed successfully." 1>&2 