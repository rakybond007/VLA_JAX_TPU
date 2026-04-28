from __future__ import annotations

from flax import core
from flax import traverse_util
from flax.training import train_state
import optax


class TrainState(train_state.TrainState):
    pass


def _ensure_frozendict(tree):
    if isinstance(tree, core.FrozenDict):
        return tree
    return core.freeze(tree)


def _make_optimizer(params, learning_rate, freeze_backbone: bool):
    params = _ensure_frozendict(params)
    if not freeze_backbone:
        return optax.adamw(learning_rate=learning_rate)

    flat = traverse_util.flatten_dict(params)
    label_tree = core.freeze(
        traverse_util.unflatten_dict(
        {
            path: ("frozen" if path and path[0] == "backbone" else "trainable")
            for path in flat
        }
        )
    )
    return optax.multi_transform(
        {
            "trainable": optax.adamw(learning_rate=learning_rate),
            "frozen": optax.set_to_zero(),
        },
        label_tree,
    )


def create_train_state(params, model_apply, learning_rate, freeze_backbone: bool = False):
    params = _ensure_frozendict(params)
    tx = _make_optimizer(params, learning_rate=learning_rate, freeze_backbone=freeze_backbone)
    return TrainState.create(
        apply_fn=model_apply,
        params=params,
        tx=tx,
    )
