# GCP TPU Setup For This Workspace

This document adapts the root [`installation.md`](../installation.md) flow to the VLA TPU reproduction repo.

Replace the placeholders below with resources from your own GCP project:

- Project: `<PROJECT_ID>`
- Zone: `<ZONE>`
- TPU VM name: `<TPU_VM_NAME>`
- Checkpoint bucket: `gs://<CHECKPOINT_BUCKET>`
- Optional attached data disk: `<DATA_DISK_NAME>`
- TPU-side mounted data path: `/mnt/disks/vla-data`

## Recommended Initial Path

1. Create a dedicated GCP project for the TPU reproduction work.
2. Link billing before trying to enable TPU or Compute APIs.
3. Enable TPU-related APIs.
4. Probe multiple zones and TPU generations instead of assuming one default zone will work.
5. Choose the first TPU shape that is both obtainable and reasonable for the model port.
6. Clone or sync this workspace onto the TPU VM.
7. Install the `vla-tpu` environment.
8. Run smoke tests before touching large checkpoints.

## Local gcloud Checklist

```bash
./scripts/gcloudw.sh --version
./scripts/gcloudw.sh auth list
./scripts/gcloudw.sh config get-value project
```

This repo uses a workspace-local `CLOUDSDK_CONFIG` directory so the CLI does not depend on `~/.config/gcloud`.

If authentication is missing, run:

```bash
./scripts/gcloudw.sh init
```

That step usually requires browser login, so it is the point where I will need your help if we run it interactively.

## Create a TPU VM

```bash
./scripts/gcloudw.sh compute tpus accelerator-types list \
  --project=<PROJECT_ID> \
  --zone=<ZONE>

./scripts/gcloudw.sh compute tpus tpu-vm create <TPU_VM_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE> \
  --accelerator-type=v6e-4 \
  --version=v2-alpha-tpuv6e
```

## SSH Into the TPU VM

```bash
./scripts/gcloudw.sh compute tpus tpu-vm ssh <TPU_VM_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

## Day-To-Day Operations

Check status:

```bash
./scripts/gcloudw.sh compute tpus tpu-vm describe <TPU_VM_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Stop when the TPU will be idle for a while:

```bash
./scripts/gcloudw.sh compute tpus tpu-vm stop <TPU_VM_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Start again before the next session:

```bash
./scripts/gcloudw.sh compute tpus tpu-vm start <TPU_VM_NAME> \
  --project=<PROJECT_ID> \
  --zone=<ZONE>
```

Practical note:

- If the TPU will sit unused for hours or days, stopping it is the safer default.
- A stopped TPU VM keeps the VM configuration, but runtime startup still takes some time on the next session.
- Availability can still vary by region and generation, so a future start is not identical to keeping the TPU continuously warm.

## Storage Setup

Checkpoint storage uses GCS:

```bash
gcloud storage ls gs://<CHECKPOINT_BUCKET>
```

Example checkpoint prefix:

```text
gs://<CHECKPOINT_BUCKET>/<RUN_NAME>/checkpoints
```

The config defaults intentionally use local `outputs/checkpoints/...` paths. For TPU or GCS-backed runs, pass the checkpoint location at runtime instead of hard-coding private bucket names into the repo:

```bash
python -m vla_tpu.training.train \
  --config libero_full_jax_qwen3_vl \
  --checkpoint_dir gs://<CHECKPOINT_BUCKET>/<RUN_NAME>/checkpoints \
  --dataset_root /mnt/disks/vla-data/datasets/libero_gr00t_delta
```

The provided training scripts follow the same pattern through environment variables:

```bash
CHECKPOINT_DIR=gs://<CHECKPOINT_BUCKET>/<RUN_NAME>/checkpoints \
DATASET_ROOT=/mnt/disks/vla-data/datasets/libero_gr00t_delta \
bash run_scripts/train/train_libero_qwen3_vl_tpu.sh
```

Additional TPU-side disk:

```text
disk name: <DATA_DISK_NAME>
type: hyperdisk-balanced
size: 300GB
mount point: /mnt/disks/vla-data
symlink: ~/tpu_proj/vla-data
```

Suggested usage:

- `/mnt/disks/vla-data/datasets`
- `/mnt/disks/vla-data/checkpoints_cache`
- `/mnt/disks/vla-data/tmp`

Check mounted disk capacity:

```bash
df -h /mnt/disks/vla-data
```

## Environment Setup On The TPU VM

```bash
cd ~/tpu_proj/vla-tpu
python3 -m venv .venv-tpu
source .venv-tpu/bin/activate
pip install --upgrade pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ".[qwen-ref,eval,tracking,checkpoint-gcs]"
```

For `v6e` specifically, prefer the TPU runtime that matches `v2-alpha-tpuv6e`.

## Smoke Tests

```bash
python -m vla_tpu.training.train --config small_debug
python -m vla_tpu.serving.infer --config small_debug
python - <<'PY'
import jax
print(jax.devices())
PY
```

## Suggested Storage Layout

```text
~/tpu_proj/
├── VLA_JAX_TPU/
└── vla-tpu/
```

## Sync Strategy

- Keep the original reference repo untouched as a reference baseline.
- Build TPU-native code in `vla-tpu/`.
- Add external references under `vla-tpu/third_party/`.

## Troubleshooting Notes

### `~/.config/gcloud` is read-only

Use the repo wrapper:

```bash
./scripts/gcloudw.sh ...
```

It sets `CLOUDSDK_CONFIG` to a writable repo-local directory.

### `PERMISSION_DENIED` on another project

This usually means the signed-in account is valid, but the selected project is not accessible to that account.

### `Billing account for project ... is not found`

Billing is not linked yet, or the link has not propagated. Confirm with:

```bash
./scripts/gcloudw.sh beta billing projects describe <PROJECT_ID>
```

### `Insufficient capacity`

This is not a credential issue. It means the requested TPU exists in that zone but no immediate capacity is available.

### `Quota limit ... exceeded`

This is different from capacity exhaustion. The requested TPU shape is larger than the quota assigned to the project in that zone.

### `libtpu` says the TPU is already in use

On a single TPU VM, avoid launching multiple JAX programs at the same time. Run smoke tests sequentially.

### Orbax says `Checkpoint path should be absolute`

Resolve the checkpoint directory to an absolute path before calling Orbax save helpers.
