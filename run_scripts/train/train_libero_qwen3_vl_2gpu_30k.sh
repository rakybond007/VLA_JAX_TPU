#!/bin/bash
#SBATCH --job-name="vla-libero-qwen3vl-2gpu-30k"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --output=slurm_out/%j-vla_libero_qwen3vl_2gpu_30k.out
#SBATCH --error=slurm_out/%j-vla_libero_qwen3vl_2gpu_30k.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

RUN_NAME="${RUN_NAME:-libero-qwen3vl-flow-2gpu-30k}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-tpu}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/outputs/checkpoints/libero_full_jax_qwen3_vl_2gpu_30k}"
CONFIG_NAME="${CONFIG_NAME:-libero_full_jax_qwen3_vl}"
NUM_STEPS="${NUM_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_SIZE="${PREFETCH_SIZE:-32}"
SAVE_EVERY="${SAVE_EVERY:-500}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/${RUN_NAME}.log}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$CHECKPOINT_DIR")"

source "$ROOT_DIR/.venv-local-eval/bin/activate"

NVIDIA_PKG_DIR="$ROOT_DIR/.venv-local-eval/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cu13/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_cccl/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cuda_nvrtc/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${NVIDIA_PKG_DIR}/cufft/lib:${NVIDIA_PKG_DIR}/cusolver/lib:${NVIDIA_PKG_DIR}/cusparse/lib:${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/nccl/lib:${NVIDIA_PKG_DIR}/nvjitlink/lib:${NVIDIA_PKG_DIR}/nvshmem/lib:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_NAME="$RUN_NAME"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  VLA TPU JAX Local 2GPU Train"
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
