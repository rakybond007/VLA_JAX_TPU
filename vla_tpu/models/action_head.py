from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from vla_tpu.configs.base import ActionHeadConfig
from vla_tpu.models.action_head_flow import FlowMatchingCrossAttnDiTHead
from vla_tpu.models.action_head_flow_legacy import LegacyFlowMatchingCrossAttnDiTHead


class FeedForwardBlock(nn.Module):
    hidden_size: int
    mlp_ratio: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_size * self.mlp_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return residual + x


class CrossAttentionDiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: int
    dropout_rate: float

    @nn.compact
    def __call__(self, query_tokens: jnp.ndarray, memory_tokens: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = query_tokens
        x = nn.LayerNorm()(query_tokens)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        query_tokens = residual + x

        residual = query_tokens
        x = nn.LayerNorm()(query_tokens)
        memory = nn.LayerNorm(name="memory_norm")(memory_tokens)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x, memory)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        query_tokens = residual + x

        return FeedForwardBlock(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="ffn",
        )(query_tokens, train=train)


class ActionHead(nn.Module):
    config: ActionHeadConfig

    def _build_action_queries(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        if "action_query" in batch:
            return batch["action_query"]

        batch_size = batch["state"].shape[0]
        return jnp.zeros(
            (batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=batch["state"].dtype,
        )

    @nn.compact
    def __call__(
        self,
        backbone_outputs: Dict[str, jnp.ndarray],
        state: jnp.ndarray,
        batch: Dict[str, jnp.ndarray],
        train: bool,
    ) -> Dict[str, jnp.ndarray]:
        memory_tokens = nn.Dense(self.config.hidden_size, name="memory_proj")(
            backbone_outputs["backbone_features"]
        )

        action_queries = self._build_action_queries(batch)
        action_tokens = nn.Dense(self.config.hidden_size, name="action_proj")(action_queries)

        state_token = nn.Dense(self.config.hidden_size, name="state_proj")(state)
        state_token = state_token[:, None, :]
        query_tokens = action_tokens + state_token

        pos_embed = self.param(
            "query_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.config.action_horizon, self.config.hidden_size),
        )
        query_tokens = query_tokens + pos_embed[None, :, :]

        for layer_idx in range(self.config.num_layers):
            query_tokens = CrossAttentionDiTBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout_rate=self.config.dropout_rate,
                name=f"block_{layer_idx}",
            )(query_tokens, memory_tokens, train=train)

        query_tokens = nn.LayerNorm(name="final_norm")(query_tokens)
        return {"action_pred": nn.Dense(self.config.action_dim, name="action_out")(query_tokens)}


def build_action_head(config: ActionHeadConfig, name: str | None = None) -> nn.Module:
    if config.impl == "legacy_flow_matching_cross_attn_dit":
        return LegacyFlowMatchingCrossAttnDiTHead(config=config, name=name)
    if config.impl == "flow_matching_cross_attn_dit":
        return FlowMatchingCrossAttnDiTHead(config=config, name=name)
    if config.impl == "cross_attn_regressor":
        return ActionHead(config=config, name=name)
    raise ValueError(f"Unsupported action head impl: {config.impl}")
