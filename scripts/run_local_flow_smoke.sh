#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source .venv-local-eval/bin/activate
export LD_LIBRARY_PATH="$(echo "$ROOT"/.venv-local-eval/lib/python3.10/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/libero_debug_jax_qwen3_vl_flow_smoke}"
XLA_FLAGS="${XLA_FLAGS:-}"

# Some CUDA/JAX combinations can spend a long time or fail in Triton GEMM autotune.
# Set this explicitly for local GPU smoke only; TPU runs should not inherit it.
if [[ -z "$XLA_FLAGS" && "${JAX_PLATFORMS:-}" == "cuda" ]]; then
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

python -u -m vla_tpu.training.train \
  --config libero_debug_jax_qwen3_vl \
  --num_steps 10 \
  --batch_size 2 \
  --num_workers 0 \
  --prefetch_size 0 \
  --use_wandb False \
  --dataset_root "$DATASET_ROOT" \
  --checkpoint_dir "$CHECKPOINT_DIR"
