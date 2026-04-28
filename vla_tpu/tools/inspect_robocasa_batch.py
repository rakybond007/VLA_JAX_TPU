from __future__ import annotations

from dataclasses import dataclass

import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch


@dataclass
class Args:
    config: str = "robocasa_debug"
    batch_size: int = 2


def main():
    args = tyro.cli(Args)
    config = get_experiment_config(args.config)
    batch = make_batch(config, batch_size=args.batch_size)
    for key, value in batch.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
