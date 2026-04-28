#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source .venv-local-eval/bin/activate
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$(echo "$ROOT"/.venv-local-eval/lib/python3.10/site-packages/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"

python -m vla_tpu.tools.convert_checkpoint_to_portable \
  --config libero_debug_jax_qwen3_vl \
  --src-checkpoint-dir "$ROOT/outputs/checkpoints/libero_debug_jax_qwen3_vl_flow_smoke_10step_v2" \
  --dst-checkpoint-dir "$ROOT/outputs/checkpoints/libero_debug_jax_qwen3_vl_flow_smoke_10step_v2_portable" \
  --dataset_root "${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta" \
  --batch-size 1
