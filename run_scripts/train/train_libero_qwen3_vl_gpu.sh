#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-libero-qwen3vl-flow-gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-tpu}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/outputs/checkpoints/libero_full_jax_qwen3_vl_gpu}"
CONFIG_NAME="${CONFIG_NAME:-libero_full_jax_qwen3_vl_legacy}"
NUM_STEPS="${NUM_STEPS:-40000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_SIZE="${PREFETCH_SIZE:-32}"
SAVE_EVERY="${SAVE_EVERY:-500}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/${RUN_NAME}.log}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv-local-eval}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$CHECKPOINT_DIR")"

source "$VENV_PATH/bin/activate"

NVIDIA_PKG_DIR="$VENV_PATH/lib/python3.10/site-packages/nvidia"
if [[ -d "$NVIDIA_PKG_DIR" ]]; then
  export LD_LIBRARY_PATH="$(echo "$NVIDIA_PKG_DIR"/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_NAME="$RUN_NAME"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  VLA JAX GPU LIBERO Train"
echo "============================================"
echo "  run_name:        $RUN_NAME"
echo "  config:          $CONFIG_NAME"
echo "  dataset_root:    $DATASET_ROOT"
echo "  checkpoint_dir:  $CHECKPOINT_DIR"
echo "  num_steps:       $NUM_STEPS"
echo "  global_batch:    $GLOBAL_BATCH_SIZE"
echo "  num_workers:     $NUM_WORKERS"
echo "  prefetch_size:   $PREFETCH_SIZE"
echo "  save_every:      $SAVE_EVERY"
echo "  cuda_devices:    $CUDA_VISIBLE_DEVICES"
echo "  jax_platforms:   $JAX_PLATFORMS"
echo "  xla_flags:       $XLA_FLAGS"
echo "  venv:            $VENV_PATH"
if [[ -n "$RESUME_CHECKPOINT_DIR" ]]; then
  echo "  resume_from:     $RESUME_CHECKPOINT_DIR"
fi
echo "============================================"
echo ""

CMD=(
  python -u -m vla_tpu.training.train
  --config "$CONFIG_NAME"
  --num_steps "$NUM_STEPS"
  --batch_size "$GLOBAL_BATCH_SIZE"
  --dataset_root "$DATASET_ROOT"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --num_workers "$NUM_WORKERS"
  --prefetch_size "$PREFETCH_SIZE"
  --save_every "$SAVE_EVERY"
  --use_wandb True
)

if [[ -n "$RESUME_CHECKPOINT_DIR" ]]; then
  CMD+=(--resume_checkpoint_dir "$RESUME_CHECKPOINT_DIR")
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
