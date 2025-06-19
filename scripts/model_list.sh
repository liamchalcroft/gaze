#!/usr/bin/env bash
# Shared list of OpenRouter VLM model identifiers (free-tier where possible).
# Source this file from other benchmark scripts:
#   source "$(dirname "$0")/model_list.sh"

MODELS_DEFAULT=(
  "opengvlab/internvl3-2b:free"
  "opengvlab/internvl3-14b:free"
  "qwen/qwen-2.5-vl-7b-instruct:free"
  "qwen/qwen2.5-vl-32b-instruct:free"
  "qwen/qwen2.5-vl-72b-instruct:free"
  "meta-llama/llama-4-scout:free"
  "meta-llama/llama-4-maverick:free"
  "google/gemma-3-1b-it:free"
  "google/gemma-3-4b-it:free"
  "google/gemma-3-12b-it:free"
  "google/gemma-3-27b-it:free"
) 