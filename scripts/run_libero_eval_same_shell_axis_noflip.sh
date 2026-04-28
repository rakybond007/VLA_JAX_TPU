#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PORT="${PORT:-8908}"
export VIDEO_OUT="${VIDEO_OUT:-outputs/libero_rollouts_40k_same_shell_axis_noflip}"
export STATE_ROT_FORMAT="${STATE_ROT_FORMAT:-axis_angle}"
export IMAGE_FLIP_180="${IMAGE_FLIP_180:-False}"
export VIDEO_FLIP_180="${VIDEO_FLIP_180:-True}"

bash scripts/run_libero_eval_same_shell.sh
