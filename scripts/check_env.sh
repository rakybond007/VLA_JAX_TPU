#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib

mods = ["jax", "flax", "optax", "orbax.checkpoint"]
for name in mods:
    importlib.import_module(name)
    print(f"ok: {name}")

import jax
print("devices:", jax.devices())
PY
