from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.serving.websocket_policy_server import PolicyRuntime
from vla_tpu.utils.checkpointing import restore_checkpoint


@dataclass
class Args:
    config: str = "libero_full_jax_qwen3_vl"
    checkpoint_dir: str = ""
    dataset_root: str = ""
    batch_size: int = 1
    run_infer: bool = True


def main() -> None:
    args = tyro.cli(Args)
    t0 = time.time()
    print("devices", jax.devices(), flush=True)

    cfg = get_experiment_config(args.config)
    if args.dataset_root:
        cfg.data.dataset_root = args.dataset_root
    print("config", cfg.name, flush=True)

    t = time.time()
    batch_provider = make_batch_provider(cfg, train=False)
    print(f"batch_provider_s={time.time() - t:.3f}", flush=True)

    t = time.time()
    batch = batch_provider.next_batch(batch_size=args.batch_size)
    print(f"next_batch_s={time.time() - t:.3f}", flush=True)
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"batch[{key}] shape={tuple(value.shape)} dtype={value.dtype}", flush=True)

    model = VLATPUPolicy(cfg)
    t = time.time()
    variables = model.init(jax.random.PRNGKey(0), batch, train=False)
    print(f"model_init_s={time.time() - t:.3f}", flush=True)

    if args.checkpoint_dir:
        t = time.time()
        restored = restore_checkpoint(args.checkpoint_dir, variables["params"])
        variables = {"params": restored}
        print(f"restore_s={time.time() - t:.3f}", flush=True)

    if args.run_infer:
        t = time.time()
        outputs = model.apply(variables, batch, train=False)
        action_pred = jax.device_get(outputs["action_pred"])
        print(f"infer_s={time.time() - t:.3f}", flush=True)
        print(
            f"action_pred shape={action_pred.shape} mean={float(action_pred.mean()):.6f} std={float(action_pred.std()):.6f}",
            flush=True,
        )

        t = time.time()
        runtime = PolicyRuntime(
            config_name=args.config,
            checkpoint_dir=args.checkpoint_dir or None,
            dataset_root=args.dataset_root or None,
        )
        print(f"runtime_init_s={time.time() - t:.3f}", flush=True)
        print(
            f"runtime_ready obs_processor={type(runtime.obs_processor).__name__} "
            f"example_keys={sorted(runtime.example_batch.keys())}",
            flush=True,
        )

    print(f"total_s={time.time() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
