#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# smoke_test.sh – Enhanced end-to-end sanity check for all system modes
# -----------------------------------------------------------------------------
# This helper runs the NOVA pipeline on a **handful of images** (default 3) for
# every (task × model × approach) combination to confirm that all enhanced system
# prompts and approaches work correctly before launching full benchmarks.
#
# Usage (from repo root):
#   bash scripts/smoke_test.sh
#
# Optional environment variables:
#   MODEL_LIST   – space-separated slugs understood by OpenRouter / OpenAI SDK
#                  Default: "google/gemma-3-4b-it:free"
#   NUM_IMAGES   – number of samples per task (default: 3)
#   TASKS        – space-separated list of tasks (default: caption diagnosis localization)
#   APPROACHES   – space-separated list of approaches (default: all 6 enhanced modes)
#
# Enhanced System Modes Tested:
#   baseline        – Enhanced JSON format and clinical accuracy
#   retrieval       – Knowledge synthesis and evidence integration
#   multiturn       – Conditional continuation and systematic analysis
#   visual          – Enhanced visual operations guidance
#   web_search      – Query formulation strategies
#   comprehensive   – All capabilities combined with optimization
# -----------------------------------------------------------------------------

set -euo pipefail

# Number of samples per run – keep very small for speed
NUM_IMAGES="${NUM_IMAGES:-3}"
# Tasks to exercise.  Adjust via `TASKS="caption diagnosis" bash ...` if desired.
TASKS=( ${TASKS:-caption diagnosis localization} )
# Enhanced approaches to test.  All six optimized system modes:
#   baseline        – Enhanced JSON format specification and clinical accuracy
#   retrieval       – Knowledge synthesis and evidence integration
#   multiturn       – Conditional continuation and systematic analysis
#   visual          – Enhanced visual operations guidance
#   web_search      – Query formulation strategies and current information
#   comprehensive   – All capabilities combined with performance optimization
# Override via `APPROACHES="baseline retrieval"` for subset testing.
APPROACHES=( ${APPROACHES:-baseline retrieval multiturn visual web_search comprehensive} )

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

echo "🔥 [smoke] Enhanced System Modes Smoke Test" 1>&2
echo "   Tasks: ${TASKS[*]}" 1>&2
echo "   Models: ${MODEL_LIST}" 1>&2
echo "   Approaches: ${APPROACHES[*]}" 1>&2
echo "   Images per combo: ${NUM_IMAGES}" 1>&2
echo "   Total combinations: $((${#TASKS[@]} * $(echo ${MODEL_LIST} | wc -w) * ${#APPROACHES[@]}))" 1>&2
echo "" 1>&2

ts="$(date +%s)"
total_tests=0
passed_tests=0

for model in ${MODEL_LIST}; do
  safe_model="${model//\//_}"  # slash → underscore for folder names
  for task in "${TASKS[@]}"; do
    for approach in "${APPROACHES[@]}"; do
      total_tests=$((total_tests + 1))
      echo "=== [smoke] $total_tests | Task: $task | Model: $model | Approach: $approach ===" 1>&2

      # Configure approach-specific parameters
      extra_args=()
      case "$approach" in
        "baseline")
          extra_args+=(strict_mode=true)
          ;;
        "retrieval")
          extra_args+=(use_retrieval=true retrieval.type=hybrid "retrieval.top_k=5")
          ;;
        "multiturn")
          extra_args+=(use_retrieval=true "multiturn_max_steps=3" retrieval.type=hybrid "retrieval.top_k=5")
          ;;
        "visual")
          extra_args+=(use_retrieval=true "visual_rounds=2" retrieval.type=hybrid "retrieval.top_k=5")
          ;;
        "web_search")
          extra_args+=(use_web_search=true)
          ;;
        "comprehensive")
          extra_args+=(use_retrieval=true use_web_search=true retrieval.type=hybrid "retrieval.top_k=8")
          extra_args+=("multiturn_max_steps=3" "comprehensive_timeout=300")
          ;;
        "visual_multiturn")
          # Legacy support - map to visual
          echo "   ⚠️  [smoke] Legacy approach 'visual_multiturn' mapped to 'visual'" 1>&2
          approach="visual"
          extra_args+=(use_retrieval=true "visual_rounds=2" retrieval.type=hybrid "retrieval.top_k=5")
          ;;
        *)
          echo "   ❌ [smoke] Unknown approach: $approach" 1>&2
          exit 1
          ;;
      esac

      output_dir="runs/smoke_${ts}/${safe_model}_${task}_${approach}"
      mkdir -p "$output_dir"
      
      # Log configuration
      {
        echo "Smoke Test Configuration"
        echo "======================="
        echo "Timestamp: $(date)"
        echo "Task: $task"
        echo "Model: $model"
        echo "Approach: $approach"
        echo "Enhanced Features: ${extra_args[*]}"
        echo "Max Iterations: $NUM_IMAGES"
      } > "$output_dir/smoke_config.txt"

      if python -m nova_retrieval_vlm.cli \
        task="$task" \
        model.name="$model" \
        approach="$approach" \
        batch_size=1 \
        max_iterations="$NUM_IMAGES" \
        paths.output_dir="$output_dir" \
        ${extra_args[@]+"${extra_args[@]}"} \
        > "$output_dir/smoke_output.log" 2>&1; then
        
        echo "   ✅ [smoke] SUCCESS" 1>&2
        echo "SUCCESS" > "$output_dir/smoke_status.txt"
        passed_tests=$((passed_tests + 1))
      else
        echo "   ❌ [smoke] FAILED" 1>&2
        echo "FAILED" > "$output_dir/smoke_status.txt"
        echo "   📄 [smoke] Check logs: $output_dir/smoke_output.log" 1>&2
        
        # Show last few lines of error for debugging
        echo "   Last 3 lines of output:" 1>&2
        tail -n 3 "$output_dir/smoke_output.log" | sed 's/^/      /' 1>&2
      fi
      echo "" 1>&2
    done
  done
done

# Generate smoke test summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" 1>&2
echo "🔥 [smoke] Enhanced System Modes Smoke Test Summary" 1>&2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" 1>&2
echo "   Total Tests: $total_tests" 1>&2
echo "   Passed: $passed_tests" 1>&2
echo "   Failed: $((total_tests - passed_tests))" 1>&2
echo "   Success Rate: $(( (passed_tests * 100) / total_tests ))%" 1>&2
echo "" 1>&2

if [[ $passed_tests -eq $total_tests ]]; then
  echo "✅ [smoke] All enhanced system modes working correctly!" 1>&2
  echo "🚀 [smoke] Ready for full benchmark execution" 1>&2
  echo "📊 [smoke] Tested approaches: ${APPROACHES[*]}" 1>&2
  echo "📋 [smoke] Results saved to: runs/smoke_${ts}/" 1>&2
else
  echo "⚠️  [smoke] Some tests failed - please review the logs above" 1>&2
  echo "📄 [smoke] Check individual logs in: runs/smoke_${ts}/" 1>&2
  exit 1
fi 