from __future__ import annotations

import jax
import jax.numpy as jnp

from vla_tpu.configs.base import ExperimentConfig


def make_dummy_batch(config: ExperimentConfig, batch_size: int):
    data_cfg = config.data
    act_cfg = config.action_head
    backbone_cfg = config.backbone
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    images = jax.random.randint(
        keys[0],
        (
            batch_size,
            data_cfg.num_cameras,
            data_cfg.image_height,
            data_cfg.image_width,
            backbone_cfg.image_channels,
        ),
        minval=0,
        maxval=255,
        dtype=jnp.uint8,
    )
    state = jax.random.normal(keys[1], (batch_size, data_cfg.state_dim), dtype=jnp.float32)
    instruction_tokens = jax.random.randint(
        keys[2],
        (batch_size, data_cfg.instruction_length),
        minval=0,
        maxval=backbone_cfg.text_vocab_size,
        dtype=jnp.int32,
    )
    actions = jax.random.normal(
        keys[3],
        (batch_size, act_cfg.action_horizon, act_cfg.action_dim),
        dtype=jnp.float32,
    )
    return {
        "images": images,
        "state": state,
        "instruction_tokens": instruction_tokens,
        "actions": actions,
    }
