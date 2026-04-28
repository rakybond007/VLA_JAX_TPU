from __future__ import annotations

from dataclasses import dataclass

import jax
import orbax.checkpoint as ocp
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.utils.checkpointing import _checkpoint_path, restore_checkpoint


@dataclass
class Args:
    config: str = "libero_debug_jax_qwen3_vl"
    src_checkpoint_dir: str = ""
    dst_checkpoint_dir: str = ""
    dataset_root: str = ""
    batch_size: int = 1


def main():
    args = tyro.cli(Args)
    if not args.src_checkpoint_dir:
        raise ValueError("--src-checkpoint-dir is required")
    if not args.dst_checkpoint_dir:
        raise ValueError("--dst-checkpoint-dir is required")

    config = get_experiment_config(args.config)
    if args.dataset_root:
        config.data.dataset_root = args.dataset_root

    provider = make_batch_provider(config, train=True)
    batch = provider.next_batch(batch_size=args.batch_size)
    model = VLATPUPolicy(config)
    variables = model.init(jax.random.PRNGKey(config.train.seed), batch, train=True)
    restored = restore_checkpoint(args.src_checkpoint_dir, variables["params"])
    host_params = jax.device_get(restored)

    dst = _checkpoint_path(args.dst_checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(dst, host_params, force=True)
    print(f"saved portable checkpoint to {dst}", flush=True)


if __name__ == "__main__":
    main()
