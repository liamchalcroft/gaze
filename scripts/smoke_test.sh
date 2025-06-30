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
#   baseline        – Standard single-turn analysis with enhanced JSON format
#   multiturn       – Chain of thought reasoning with up to 3 turns
#   visual          – Chain of thought with visual operations (zoom, crop, contrast)
#   web_search      – Chain of thought with web search capabilities
#   comprehensive   – Chain of thought with both visual operations AND web search
# -----------------------------------------------------------------------------

set -euo pipefail

# Number of samples per run – keep very small for speed
NUM_IMAGES="${NUM_IMAGES:-3}"
# Tasks to exercise.  Adjust via `TASKS="caption diagnosis" bash ...` if desired.
TASKS=( ${TASKS:-caption diagnosis localization} )
# Enhanced approaches to test.  All five optimized system modes:
#   baseline        – Standard single-turn analysis with enhanced JSON format
#   multiturn       – Chain of thought reasoning with up to 3 turns
#   visual          – Chain of thought with visual operations (zoom, crop, contrast)
#   web_search      – Chain of thought with web search capabilities
#   comprehensive   – Chain of thought with both visual operations AND web search
# Override via `APPROACHES="baseline multiturn"` for subset testing.
# APPROACHES=( ${APPROACHES:-baseline multiturn visual web_search comprehensive} )
APPROACHES=( ${APPROACHES:-web_search comprehensive} )

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
          # Baseline uses standard single-turn analysis
          extra_args=()
          ;;
        "multiturn")
          # Multi-turn chain of thought reasoning (up to 3 turns)
          extra_args=()
          ;;
        "visual")
          # Visual chain of thought with visual operations
          extra_args=()
          ;;
        "web_search")
          # Web search chain of thought with web search capabilities
          extra_args+=(use_web_search=true)
          ;;
        "comprehensive")
          # Comprehensive chain of thought with both visual operations AND web search
          extra_args+=(use_web_search=true)
          ;;
        "visual_multiturn")
          # Legacy support - map to visual
          echo "   ⚠️  [smoke] Legacy approach 'visual_multiturn' mapped to 'visual'" 1>&2
          approach="visual"
          extra_args=()
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
        echo "Enhanced Features: ${extra_args[*]:-none}"
        echo "Max Iterations: $NUM_IMAGES"
      } > "$output_dir/smoke_config.txt"

      # Build command arguments
      cmd_args=(
        task="$task"
        model.name="$model"
        approach="$approach"
        batch_size=1
        max_iterations="$NUM_IMAGES"
        paths.output_dir="$output_dir"
      )
      
      # Add extra args if any
      if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd_args+=("${extra_args[@]}")
      fi

      if python -m nova_retrieval_vlm.cli "${cmd_args[@]}" \
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