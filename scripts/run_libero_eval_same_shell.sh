#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8908}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDX="${TASK_IDX:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
STATE_ROT_FORMAT="${STATE_ROT_FORMAT:-axis_angle}"
IMAGE_FLIP_180="${IMAGE_FLIP_180:-False}"
VIDEO_OUT="${VIDEO_OUT:-outputs/libero_rollouts_40k_same_shell}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/outputs/checkpoints/libero_full_jax_qwen3_vl_2gpu_40k}"
DATASET_ROOT="${DATASET_ROOT:-${HF_LEROBOT_CACHE:-/path/to/lerobot}/kimtaey/libero_gr00t_delta}"
CONFIG_NAME="${CONFIG_NAME:-libero_full_jax_qwen3_vl_legacy_eval}"
LOG_DIR="${LOG_DIR:-logs}"
VIDEO_FLIP_180="${VIDEO_FLIP_180:-False}"
SERVER_VENV_PATH="${VENV_PATH:-$PWD/.venv-local-eval}"
CLIENT_VENV_PATH="${CLIENT_VENV_PATH:-$PWD/.venv-local-libero-client}"
if [[ ! -x "$CLIENT_VENV_PATH/bin/python" && -x "$SERVER_VENV_PATH/bin/python" ]]; then
  CLIENT_VENV_PATH="$SERVER_VENV_PATH"
fi
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
if [[ -n "$EXTRA_PYTHONPATH" ]]; then
  RUNTIME_PYTHONPATH="$PWD:$EXTRA_PYTHONPATH"
else
  RUNTIME_PYTHONPATH="$PWD"
fi

mkdir -p "$LOG_DIR" "$VIDEO_OUT"
SERVER_LOG="$LOG_DIR/libero_server_${PORT}.log"
CLIENT_LOG="$LOG_DIR/libero_client_${PORT}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PYTHONPATH="$RUNTIME_PYTHONPATH" \
  "$SERVER_VENV_PATH/bin/python" -m vla_tpu.serving.websocket_policy_server \
  --config "$CONFIG_NAME" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --dataset_root "$DATASET_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "server_pid=$SERVER_PID"
echo "server_log=$SERVER_LOG"

READY=0
for _ in $(seq 1 180); do
  if grep -q "Websocket policy server listening" "$SERVER_LOG" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited early"
    tail -n 120 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "server did not become ready"
  tail -n 120 "$SERVER_LOG" || true
  exit 1
fi

echo "server_ready=true"

CLIENT_ARGS=(
  -m vla_tpu.eval.libero_eval
  --policy_mode websocket
  --host 127.0.0.1
  --port "$PORT"
  --task_suite_name "$TASK_SUITE"
  --task_idx "$TASK_IDX"
  --num_trials_per_task "$NUM_TRIALS"
  --video_out_path "$VIDEO_OUT"
  --state_rot_format "$STATE_ROT_FORMAT"
)

if [[ "$IMAGE_FLIP_180" == "True" ]]; then
  CLIENT_ARGS+=(--image_flip_180)
fi
if [[ "$VIDEO_FLIP_180" == "True" ]]; then
  CLIENT_ARGS+=(--video_flip_180)
fi

MPLCONFIGDIR=/tmp/matplotlib-vla-tpu \
PYTHONPATH="$RUNTIME_PYTHONPATH" \
  "$CLIENT_VENV_PATH/bin/python" "${CLIENT_ARGS[@]}" \
  >"$CLIENT_LOG" 2>&1

CLIENT_RC=$?
echo "client_rc=$CLIENT_RC"
echo "client_log=$CLIENT_LOG"
tail -n 120 "$CLIENT_LOG" || true
exit "$CLIENT_RC"
