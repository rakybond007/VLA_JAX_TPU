#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CLOUDSDK_CONFIG="${ROOT_DIR}/.gcloud"
mkdir -p "${CLOUDSDK_CONFIG}"

exec "${ROOT_DIR}/.local/google-cloud-sdk/bin/gcloud" "$@"
