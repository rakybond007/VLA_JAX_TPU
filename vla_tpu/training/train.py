from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import time

from flax import core
from flax import jax_utils
import jax
import jax.numpy as jnp
from jax import lax
import optax
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.models.qwen3_vl_weight_loader import (
    load_hf_qwen3_vl_state_dict,
    load_hf_weights_into_jax_qwen3_vl_pure_backbone,
)
from vla_tpu.training.state import create_train_state
from vla_tpu.utils.checkpointing import restore_checkpoint, save_checkpoint

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


@dataclass
class Args:
    config: str = "small_debug"
    num_steps: int = -1
    batch_size: int = -1
    save_every: int = -1
    checkpoint_dir: str = ""
    dataset_root: str = ""
    num_workers: int = -1
    prefetch_size: int = -1
    use_wandb: bool | None = None
    resume_checkpoint_dir: str = ""


def _maybe_init_wandb(config, args):
    if not config.train.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config but the package is not installed.")

    run_name = f"{config.name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    init_kwargs = {
        "project": config.train.wandb_project,
        "name": run_name,
        "config": {
            "experiment": config.name,
            "learning_rate": config.train.learning_rate,
            "batch_size": config.train.batch_size,
            "num_steps": config.train.num_steps,
            "checkpoint_dir": config.train.checkpoint_dir,
            "backbone_impl": config.backbone.impl,
            "action_dim": config.action_head.action_dim,
            "action_horizon": config.action_head.action_horizon,
            "dataset_type": config.data.dataset_type,
            "dataset_root": config.data.dataset_root,
            "cli_config": args.config,
        },
        "mode": config.train.wandb_mode,
    }
    if config.train.wandb_entity:
        init_kwargs["entity"] = config.train.wandb_entity
    return wandb.init(**init_kwargs)


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def select_loss(outputs, batch):
    if "loss" in outputs:
        return outputs["loss"]
    return mse_loss(outputs["action_pred"], batch["actions"])


def reshape_for_pmap(batch, num_devices: int):
    sharded = {}
    for key, value in batch.items():
        if value.ndim == 0:
            sharded[key] = jnp.broadcast_to(value, (num_devices,))
            continue
        if key == "position_ids":
            if value.shape[1] % num_devices != 0:
                raise ValueError(
                    f"batch field '{key}' with shape {value.shape} is not divisible by local_device_count={num_devices}"
                )
            sharded[key] = value.reshape((value.shape[0], num_devices, value.shape[1] // num_devices, *value.shape[2:]))
            sharded[key] = jnp.transpose(sharded[key], (1, 0, 2, 3))
            continue
        if value.shape[0] % num_devices != 0:
            raise ValueError(
                f"batch field '{key}' with shape {value.shape} is not divisible by local_device_count={num_devices}"
            )
        sharded[key] = value.reshape((num_devices, value.shape[0] // num_devices, *value.shape[1:]))
    return sharded


def take_first_replica(tree):
    def _take(x):
        if hasattr(x, "addressable_shards") and x.addressable_shards:
            return jax.device_get(x.addressable_shards[0].data)
        return jax.device_get(x[0])

    return jax.tree_util.tree_map(_take, tree)


def _maybe_load_pretrained_backbone(config, params):
    if config.backbone.impl != "jax_qwen3_vl_pure":
        return params

    state_dict = load_hf_qwen3_vl_state_dict(config.backbone.model_name, torch_dtype="float32")
    mutable = core.unfreeze(params)
    loaded_backbone, summary = load_hf_weights_into_jax_qwen3_vl_pure_backbone(
        core.freeze(mutable["backbone"]["pure_backbone"]),
        state_dict,
    )
    mutable["backbone"]["pure_backbone"] = core.unfreeze(loaded_backbone)
    print(
        f"loaded pure Qwen3-VL backbone weights: loaded={summary.loaded} "
        f"skipped={summary.skipped} mismatched={summary.mismatched}"
    )
    return core.freeze(mutable)


def main():
    args = tyro.cli(Args)
    config = get_experiment_config(args.config)
    if args.num_steps > 0:
        config.train.num_steps = args.num_steps
    if args.batch_size > 0:
        config.train.batch_size = args.batch_size
    if args.save_every > 0:
        config.train.save_every = args.save_every
    if args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir
    if args.dataset_root:
        config.data.dataset_root = args.dataset_root
    if args.num_workers >= 0:
        config.data.num_workers = args.num_workers
    if args.prefetch_size >= 0:
        config.data.prefetch_size = args.prefetch_size
    if args.use_wandb is not None:
        config.train.use_wandb = args.use_wandb
    run = _maybe_init_wandb(config, args)
    batch_provider = make_batch_provider(config, train=True)
    print(
        f"train_setup config={config.name} dataset_root={config.data.dataset_root} "
        f"num_workers={config.data.num_workers} prefetch_size={config.data.prefetch_size} "
        f"use_wandb={config.train.use_wandb} save_every={config.train.save_every}",
        flush=True,
    )
    print("fetching initial batch...", flush=True)
    init_data_start = time.perf_counter()
    batch = batch_provider.next_batch(batch_size=config.train.batch_size)
    init_data_time = time.perf_counter() - init_data_start
    print(f"initial batch fetched in {init_data_time:.3f}s", flush=True)
    model = VLATPUPolicy(config)
    rng = jax.random.PRNGKey(config.train.seed)
    print("initializing model variables...", flush=True)
    variables = model.init(rng, batch, train=True)
    print("model.init finished", flush=True)
    print("loading pretrained backbone weights...", flush=True)
    if args.resume_checkpoint_dir:
        params = variables["params"]
    else:
        params = _maybe_load_pretrained_backbone(config, variables["params"])
    print("backbone weights ready", flush=True)
    warmup_steps = int(config.train.num_steps * config.train.warmup_ratio)
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.train.learning_rate,
        warmup_steps=max(warmup_steps, 1),
        decay_steps=max(config.train.num_steps, 1),
        end_value=0.0,
    )
    state = create_train_state(
        params=params,
        model_apply=model.apply,
        learning_rate=learning_rate,
        freeze_backbone=config.train.freeze_backbone,
    )
    if args.resume_checkpoint_dir:
        print(f"restoring checkpoint from {args.resume_checkpoint_dir}", flush=True)
        restored_params = restore_checkpoint(args.resume_checkpoint_dir, state.params)
        state = state.replace(params=restored_params)
        print("checkpoint restore finished", flush=True)

    final_params = None
    last_completed_step = 0

    if config.train.use_pmap:
        num_devices = jax.local_device_count()
        if config.train.batch_size % num_devices != 0:
            raise ValueError(
                f"batch_size={config.train.batch_size} must be divisible by local_device_count={num_devices}"
            )

        state = jax_utils.replicate(state)

        @partial(jax.pmap, axis_name="data", donate_argnums=(0,))
        def train_step(train_state, step_batch, step_rng):
            def loss_fn(params):
                outputs = model.apply({"params": params}, step_batch, train=True, rngs={"dropout": step_rng})
                loss = select_loss(outputs, step_batch)
                return loss, outputs

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            loss = lax.pmean(loss, axis_name="data")
            grads = lax.pmean(grads, axis_name="data")
            new_state = train_state.apply_gradients(grads=grads)
            return new_state, loss

        dropout_rng = jax.random.split(jax.random.PRNGKey(config.train.seed + 1), num_devices)
        prev_step_end = time.perf_counter()
        try:
            for step in range(1, config.train.num_steps + 1):
                data_start = time.perf_counter()
                batch = batch_provider.next_batch(batch_size=config.train.batch_size)
                data_time = time.perf_counter() - data_start
                sharded_batch = reshape_for_pmap(batch, num_devices)
                split_rngs = jax.vmap(lambda key: jax.random.split(key, 2))(dropout_rng)
                step_rng = split_rngs[:, 0]
                dropout_rng = split_rngs[:, 1]
                step_start = time.perf_counter()
                state, loss = train_step(state, sharded_batch, step_rng)
                step_time = time.perf_counter() - step_start
                host_loss = float(jax.device_get(loss)[0])
                last_completed_step = step
                if step % config.train.log_every == 0 or step == 1:
                    wall_time = time.perf_counter() - prev_step_end
                    prev_step_end = time.perf_counter()
                    print(
                        f"step={step} loss={host_loss:.6f} "
                        f"data_time={data_time:.3f}s step_time={step_time:.3f}s wall_time={wall_time:.3f}s"
                    )
                    if run is not None:
                        metrics = {
                            "train/loss": host_loss,
                            "train/step": step,
                            "train/data_time_s": data_time,
                            "train/step_time_s": step_time,
                            "train/wall_time_s": wall_time,
                        }
                        if step == 1:
                            metrics["train/init_data_time_s"] = init_data_time
                        wandb.log(metrics, step=step)
                if config.train.save_every > 0 and step % config.train.save_every == 0:
                    periodic_params = take_first_replica(state.params)
                    save_checkpoint(config.train.checkpoint_dir, periodic_params)
                    print(f"saved checkpoint to {config.train.checkpoint_dir} at step={step}", flush=True)
            final_params = take_first_replica(state.params)
        except Exception:
            failure_params = take_first_replica(state.params)
            save_checkpoint(config.train.checkpoint_dir, failure_params)
            print(
                f"saved checkpoint to {config.train.checkpoint_dir} after failure at step={last_completed_step}",
                flush=True,
            )
            raise
    else:
        @partial(jax.jit, donate_argnums=(0,))
        def train_step(train_state, step_batch, step_rng):
            def loss_fn(params):
                outputs = model.apply({"params": params}, step_batch, train=True, rngs={"dropout": step_rng})
                loss = select_loss(outputs, step_batch)
                return loss, outputs

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            new_state = train_state.apply_gradients(grads=grads)
            return new_state, loss

        dropout_rng = jax.random.PRNGKey(config.train.seed + 1)
        prev_step_end = time.perf_counter()
        try:
            for step in range(1, config.train.num_steps + 1):
                data_start = time.perf_counter()
                batch = batch_provider.next_batch(batch_size=config.train.batch_size)
                data_time = time.perf_counter() - data_start
                dropout_rng, step_rng = jax.random.split(dropout_rng)
                step_start = time.perf_counter()
                state, loss = train_step(state, batch, step_rng)
                step_time = time.perf_counter() - step_start
                last_completed_step = step
                if step % config.train.log_every == 0 or step == 1:
                    host_loss = float(loss)
                    wall_time = time.perf_counter() - prev_step_end
                    prev_step_end = time.perf_counter()
                    print(
                        f"step={step} loss={host_loss:.6f} "
                        f"data_time={data_time:.3f}s step_time={step_time:.3f}s wall_time={wall_time:.3f}s"
                    )
                    if run is not None:
                        metrics = {
                            "train/loss": host_loss,
                            "train/step": step,
                            "train/data_time_s": data_time,
                            "train/step_time_s": step_time,
                            "train/wall_time_s": wall_time,
                        }
                        if step == 1:
                            metrics["train/init_data_time_s"] = init_data_time
                        wandb.log(metrics, step=step)
                if config.train.save_every > 0 and step % config.train.save_every == 0:
                    save_checkpoint(config.train.checkpoint_dir, state.params)
                    print(f"saved checkpoint to {config.train.checkpoint_dir} at step={step}", flush=True)
            final_params = state.params
        except Exception:
            save_checkpoint(config.train.checkpoint_dir, state.params)
            print(
                f"saved checkpoint to {config.train.checkpoint_dir} after failure at step={last_completed_step}",
                flush=True,
            )
            raise

    save_checkpoint(config.train.checkpoint_dir, final_params)
    print(f"saved checkpoint to {config.train.checkpoint_dir}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
