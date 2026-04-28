# Architecture Notes

## VLA TPU Boundary

From the inspected reference code:

- `policy.prepare_input(...)`
- `backbone(...)`
- `backbone_outputs["backbone_features"]`
- `action_head(...)`
- `action_pred`

This is the cleanest TPU port boundary because it avoids coupling TPU work to the trainer internals too early.

## Planned TPU Port Stages

### Stage 1

Build and validate a TPU-safe minimal policy:

- multimodal observation batch
- backbone adapter
- action head
- optimizer / checkpoint / inference

### Stage 2

Replace the dummy backbone with a Qwen-family adapter:

- first target: `Qwen3-VL`
- second target: `Qwen3.5`

### Stage 3

Increase fidelity:

- meta-query support
- memory augmentation
- world-model stream
- checkpoint conversion from reference format

## Why JAX First

- TPU support is mature and straightforward.
- Compilation and sharding tools are better aligned with TPU-first workflows.
- A staged JAX port lets us validate the action head even if the full Qwen backbone path is still open.

## Fallback Path

If Qwen3-VL or Qwen3.5 turns out to be much more practical in `PyTorch/XLA`, we can keep the same high-level repo structure and swap only the implementation under `models/backbone.py` and `training/`.
