#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv-local-eval/bin/activate

python -m vla_tpu.serving.websocket_policy_server \
  --config libero_full_jax_qwen3_vl_legacy_eval \
  --checkpoint_dir "$PWD/outputs/checkpoints/libero_full_jax_qwen3_vl_2gpu_40k" \
  --dataset_root "${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta" \
  --host 127.0.0.1 \
  --port 8908
