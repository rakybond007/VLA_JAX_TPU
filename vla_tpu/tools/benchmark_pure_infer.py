from __future__ import annotations

import statistics
import time
import logging

import jax
import jax.numpy as jnp
import tyro

from vla_tpu.serving.websocket_policy_server import PolicyRuntime


def main(
    config: str = "libero_full_jax_qwen3_vl_legacy_eval",
    checkpoint_dir: str = "",
    dataset_root: str = "",
    warmup: int = 5,
    iters: int = 30,
    use_jit: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    print("init:construct_runtime_start", flush=True)
    runtime = PolicyRuntime(
        config_name=config,
        checkpoint_dir=checkpoint_dir or None,
        dataset_root=dataset_root or None,
    )
    print("init:construct_runtime_done", flush=True)
    print("init:batch_to_device_start", flush=True)
    batch = {key: jnp.asarray(value) for key, value in runtime.example_batch.items()}
    batch["inference_seed"] = jnp.asarray(0, dtype=jnp.uint32)
    print("init:batch_to_device_done", flush=True)

    if use_jit:
        def _apply_with_variables(variables, input_batch):
            return runtime.model.apply(variables, input_batch, train=False)["action_pred"]

        compiled_apply = jax.jit(_apply_with_variables)

        def apply_fn(input_batch):
            return compiled_apply(runtime.variables, input_batch)
    else:
        apply_fn = runtime.apply
    print(f"benchmark:use_jit={use_jit}", flush=True)

    for idx in range(warmup):
        out = apply_fn(batch)
        out.block_until_ready()
        print(f"warmup_iter={idx + 1}", flush=True)
    print("warmup_done", flush=True)

    elapsed = []
    for idx in range(iters):
        start = time.perf_counter()
        out = apply_fn(batch)
        out.block_until_ready()
        seconds = time.perf_counter() - start
        elapsed.append(seconds)
        print(f"iter={idx + 1} pure_model_forward_sec={seconds:.6f}", flush=True)

    sorted_elapsed = sorted(elapsed)
    print("summary")
    print(f"device_platform={jax.default_backend()}")
    print(f"local_device_count={jax.local_device_count()}")
    print(f"action_pred_shape={tuple(out.shape)}")
    print(f"warmup={warmup} iters={iters}")
    print(f"use_jit={use_jit}")
    print(f"mean_sec={statistics.mean(elapsed):.6f}")
    print(f"median_sec={statistics.median(elapsed):.6f}")
    print(f"min_sec={min(elapsed):.6f}")
    print(f"max_sec={max(elapsed):.6f}")
    print(f"p90_sec={sorted_elapsed[int(0.9 * (len(sorted_elapsed) - 1))]:.6f}")


if __name__ == "__main__":
    tyro.cli(main)
