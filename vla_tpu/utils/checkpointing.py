from __future__ import annotations

from pathlib import Path

from flax import core
import jax
import numpy as np
import orbax.checkpoint as ocp


def _checkpoint_path(checkpoint_dir: str) -> str:
    if checkpoint_dir.startswith("gs://"):
        return checkpoint_dir.rstrip("/")
    path = Path(checkpoint_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_checkpoint(checkpoint_dir: str, train_state):
    path_str = _checkpoint_path(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    item = train_state.params if hasattr(train_state, "params") else train_state
    checkpointer.save(path_str, item, force=True)


def restore_checkpoint(checkpoint_dir: str, target_params):
    path_str = _checkpoint_path(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),
        target_params,
    )
    restored = checkpointer.restore(
        path_str,
        args=ocp.args.PyTreeRestore(item=target_params, restore_args=restore_args),
    )

    def _align_shapes(restored_leaf, target_leaf):
        restored_arr = np.asarray(restored_leaf)
        target_arr = np.asarray(target_leaf)
        if restored_arr.shape == target_arr.shape:
            return restored_leaf
        if restored_arr.ndim == target_arr.ndim + 1 and restored_arr.shape[0] == 1:
            squeezed = np.squeeze(restored_arr, axis=0)
            if squeezed.shape == target_arr.shape:
                return squeezed.astype(target_arr.dtype, copy=False)
        return restored_leaf

    aligned = jax.tree_util.tree_map(_align_shapes, restored, target_params)
    if isinstance(target_params, core.FrozenDict):
        return core.freeze(aligned)
    return aligned
