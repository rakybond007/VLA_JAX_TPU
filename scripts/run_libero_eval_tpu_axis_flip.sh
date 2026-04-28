#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"
export VENV_PATH="${VENV_PATH:-$PWD/.venv-tpu}"
if [[ ! -d "$VENV_PATH" ]]; then
  export VENV_PATH="$PWD/.venv-local-eval"
fi
export PORT="${PORT:-8913}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/outputs/checkpoints/libero_full_jax_qwen3_vl_tpu}"
export DATASET_ROOT="${DATASET_ROOT:-/mnt/disks/vla-data/datasets/libero_gr00t_delta}"
export VIDEO_OUT="${VIDEO_OUT:-outputs/libero_rollouts_tpu_axis_flip}"
export CONFIG_NAME="${CONFIG_NAME:-libero_full_jax_qwen3_vl_legacy_eval}"
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
export TASK_IDX="${TASK_IDX:-0}"
export NUM_TRIALS="${NUM_TRIALS:-10}"
export STATE_ROT_FORMAT="${STATE_ROT_FORMAT:-axis_angle}"
export IMAGE_FLIP_180="${IMAGE_FLIP_180:-True}"
export VIDEO_FLIP_180="${VIDEO_FLIP_180:-False}"

bash scripts/run_libero_eval_same_shell.sh
