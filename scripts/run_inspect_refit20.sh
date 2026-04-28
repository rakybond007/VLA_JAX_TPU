#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source .venv-local-eval/bin/activate

python -u -m vla_tpu.tools.inspect_checkpoint_batch_fit \
  --config libero_debug_jax_qwen3_vl \
  --checkpoint_dir "$ROOT/outputs/checkpoints/libero_debug_jax_qwen3_vl_flow_refit_20" \
  --dataset_root "${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta"
