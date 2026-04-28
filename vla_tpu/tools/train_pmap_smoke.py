from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import jax_utils
from jax import lax
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.training.state import create_train_state


@dataclass
class Args:
    config: str = "small_debug_jax_qwen3_vl"
    num_steps: int = 2
    global_batch_size: int = 4


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def reshape_for_pmap(batch, num_devices: int):
    return {
        key: value.reshape((num_devices, value.shape[0] // num_devices, *value.shape[1:]))
        for key, value in batch.items()
    }


def main():
    args = tyro.cli(Args)
    num_devices = jax.local_device_count()
    if args.global_batch_size % num_devices != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be divisible by local_device_count={num_devices}"
        )

    config = get_experiment_config(args.config)
    config.train.batch_size = args.global_batch_size
    batch_provider = make_batch_provider(config, train=True)
    batch = batch_provider.next_batch(batch_size=args.global_batch_size)
    sharded_batch = reshape_for_pmap(batch, num_devices)

    model = VLATPUPolicy(config)
    init_variables = model.init(jax.random.PRNGKey(config.train.seed), batch, train=True)
    state = create_train_state(
        params=init_variables["params"],
        model_apply=model.apply,
        learning_rate=config.train.learning_rate,
    )
    state = jax_utils.replicate(state)
    dropout_rng = jax.random.split(jax.random.PRNGKey(config.train.seed + 1), num_devices)

    @partial(jax.pmap, axis_name="data")
    def train_step(train_state, step_batch, step_rng):
        def loss_fn(params):
            outputs = model.apply({"params": params}, step_batch, train=True, rngs={"dropout": step_rng})
            loss = mse_loss(outputs["action_pred"], step_batch["actions"])
            return loss, outputs

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        loss = lax.pmean(loss, axis_name="data")
        grads = lax.pmean(grads, axis_name="data")
        new_state = train_state.apply_gradients(grads=grads)
        return new_state, loss

    print(f"devices={num_devices} global_batch_size={args.global_batch_size}")
    for step in range(1, args.num_steps + 1):
        batch = batch_provider.next_batch(batch_size=args.global_batch_size)
        sharded_batch = reshape_for_pmap(batch, num_devices)
        state, loss = train_step(state, sharded_batch, dropout_rng)
        print(f"step={step} loss={float(jax.device_get(loss)[0]):.6f}")

    final_state = jax_utils.unreplicate(state)
    leaves = jax.tree_util.tree_leaves(final_state.params)
    total_params = sum(x.size for x in leaves)
    print(f"final_param_tensors={len(leaves)} total_params={total_params}")


if __name__ == "__main__":
    main()
