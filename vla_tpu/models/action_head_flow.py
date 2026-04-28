from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from vla_tpu.configs.base import ActionHeadConfig


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def _sinusoidal_time_features(timestep_ids: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = dim // 2
    denom = jnp.maximum(half - 1, 1)
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / denom)
    angles = timestep_ids.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb[:, :dim]


class SinusoidalPositionalEncoding(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.embedding_dim // 2
        exponent = -jnp.arange(half_dim, dtype=jnp.float32) * (jnp.log(10000.0) / jnp.maximum(half_dim, 1))
        freqs = timesteps.astype(jnp.float32)[..., None] * jnp.exp(exponent)[None, None, :]
        emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
        if emb.shape[-1] < self.embedding_dim:
            emb = jnp.pad(emb, ((0, 0), (0, 0), (0, self.embedding_dim - emb.shape[-1])))
        return emb[..., : self.embedding_dim]


class SinusoidalSequenceEmbedding(nn.Module):
    embedding_dim: int
    max_seq_length: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        seq_len = hidden_states.shape[1]
        positions = jnp.arange(seq_len, dtype=jnp.float32)[None, :]
        pos_emb = SinusoidalPositionalEncoding(self.embedding_dim, name="sin_pos")(positions)
        return hidden_states + pos_emb


class TimestepEncoder(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, timestep_ids: jnp.ndarray) -> jnp.ndarray:
        x = _sinusoidal_time_features(timestep_ids, 256)
        x = nn.Dense(self.hidden_size, name="time_embed_in")(x)
        x = _silu(x)
        x = nn.Dense(self.hidden_size, name="time_embed_out")(x)
        return x


class AdaLayerNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, temb: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=self.eps, use_scale=False, use_bias=False, name="norm")(x)
        scale_shift = nn.Dense(self.hidden_size * 2, name="linear")(_silu(temb))
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = jax.nn.relu(x)
        x = nn.Dense(self.output_dim, name="fc2")(x)
        return x


class ActionEncoder(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, noisy_actions: jnp.ndarray, timestep_ids: jnp.ndarray) -> jnp.ndarray:
        batch_size, action_horizon, _ = noisy_actions.shape
        action_emb = nn.Dense(self.hidden_size, name="W1")(noisy_actions)
        expanded_t = jnp.broadcast_to(timestep_ids[:, None], (batch_size, action_horizon))
        tau_emb = SinusoidalPositionalEncoding(self.hidden_size, name="pos_encoding")(expanded_t)
        hidden = jnp.concatenate([action_emb, tau_emb], axis=-1)
        hidden = nn.Dense(self.hidden_size, name="W2")(hidden)
        hidden = _silu(hidden)
        hidden = nn.Dense(self.hidden_size, name="W3")(hidden)
        return hidden


class FeedForward(nn.Module):
    hidden_size: int
    mlp_ratio: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        inner_dim = self.hidden_size * self.mlp_ratio
        x = nn.Dense(inner_dim, name="proj_in")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.hidden_size, name="proj_out")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x


class SelfAttentionBlock(nn.Module):
    hidden_size: int
    num_heads: int
    dropout_rate: float
    mlp_ratio: int
    max_num_positional_embeddings: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(hidden_states)
        x = SinusoidalSequenceEmbedding(
            embedding_dim=self.hidden_size,
            max_seq_length=self.max_num_positional_embeddings,
            name="pos_embed",
        )(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
            name="attn",
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        hidden_states = hidden_states + x
        x = nn.LayerNorm(epsilon=1e-5, name="norm2")(hidden_states)
        x = FeedForward(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="ff",
        )(x, train=train)
        return hidden_states + x


class SelfAttentionTransformer(nn.Module):
    hidden_size: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    mlp_ratio: int
    max_num_positional_embeddings: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, train: bool) -> jnp.ndarray:
        for layer_idx in range(self.num_layers):
            hidden_states = SelfAttentionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                mlp_ratio=self.mlp_ratio,
                max_num_positional_embeddings=self.max_num_positional_embeddings,
                name=f"block_{layer_idx}",
            )(hidden_states, train=train)
        return hidden_states


class BackboneMemoryProcessor(nn.Module):
    config: ActionHeadConfig

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, train: bool) -> jnp.ndarray:
        hidden_states = nn.LayerNorm(epsilon=1e-5, name="vl_layer_norm")(hidden_states)
        num_heads = max(1, hidden_states.shape[-1] // 64)
        hidden_states = SelfAttentionTransformer(
            hidden_size=hidden_states.shape[-1],
            num_heads=num_heads,
            num_layers=2,
            dropout_rate=self.config.dropout_rate,
            mlp_ratio=self.config.mlp_ratio,
            max_num_positional_embeddings=max(self.config.action_horizon + 1, hidden_states.shape[1]),
            name="vl_self_attention",
        )(hidden_states, train=train)
        return hidden_states


class CrossAttentionBlock(nn.Module):
    hidden_size: int
    num_heads: int
    dropout_rate: float
    mlp_ratio: int
    max_num_positional_embeddings: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        x = AdaLayerNorm(self.hidden_size, name="norm1")(hidden_states, temb)
        x = SinusoidalSequenceEmbedding(
            embedding_dim=self.hidden_size,
            max_seq_length=self.max_num_positional_embeddings,
            name="pos_embed",
        )(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
            name="attn",
        )(x, encoder_hidden_states)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        hidden_states = hidden_states + x

        x = nn.LayerNorm(epsilon=1e-5, name="norm3")(hidden_states)
        x = FeedForward(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="ff",
        )(x, train=train)
        return hidden_states + x


class CrossAttentionDiT(nn.Module):
    config: ActionHeadConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        timestep_ids: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        temb = TimestepEncoder(self.config.hidden_size, name="timestep_encoder")(timestep_ids)
        for layer_idx in range(self.config.num_layers):
            hidden_states = CrossAttentionBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
                mlp_ratio=self.config.mlp_ratio,
                max_num_positional_embeddings=self.config.action_horizon + 1,
                name=f"block_{layer_idx}",
            )(hidden_states, encoder_hidden_states, temb, train=train)

        shift_scale = nn.Dense(self.config.hidden_size * 2, name="proj_out_1")(_silu(temb))
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        hidden_states = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False, name="norm_out")(hidden_states)
        hidden_states = hidden_states * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return hidden_states


class FlowMatchingCrossAttnDiTHead(nn.Module):
    config: ActionHeadConfig

    def setup(self) -> None:
        self.memory_processor = BackboneMemoryProcessor(self.config, name="memory_processor")
        self.state_encoder = MLP(
            input_dim=0,
            hidden_dim=self.config.hidden_size,
            output_dim=self.config.hidden_size,
            name="state_encoder",
        )
        self.action_encoder = ActionEncoder(self.config.hidden_size, name="action_encoder")
        self.action_decoder = MLP(
            input_dim=0,
            hidden_dim=self.config.hidden_size,
            output_dim=self.config.action_dim,
            name="action_decoder",
        )
        if self.config.add_action_pos_embed:
            self.action_pos_embed = nn.Embed(
                num_embeddings=max(self.config.action_horizon, 1),
                features=self.config.hidden_size,
                name="action_pos_embed",
            )
        else:
            self.action_pos_embed = None
        self.dit = CrossAttentionDiT(self.config, name="dit")

    def _sample_t(self, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        alpha = self.config.flow_beta_alpha
        beta = self.config.flow_beta_beta
        u1 = jax.random.gamma(rng, alpha, (batch_size,), dtype=jnp.float32)
        u2 = jax.random.gamma(jax.random.fold_in(rng, 1), beta, (batch_size,), dtype=jnp.float32)
        sample = u1 / jnp.maximum(u1 + u2, 1e-6)
        return (self.config.flow_noise_s - sample) / self.config.flow_noise_s

    def _predict_velocity(
        self,
        memory_tokens: jnp.ndarray,
        state: jnp.ndarray,
        noisy_actions: jnp.ndarray,
        timestep_ids: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        state_token = self.state_encoder(state)[:, None, :]
        action_tokens = self.action_encoder(noisy_actions, timestep_ids)
        if self.action_pos_embed is not None:
            pos_ids = jnp.arange(self.config.action_horizon, dtype=jnp.int32)
            pos_emb = self.action_pos_embed(pos_ids)[None, :, :]
            action_tokens = action_tokens + pos_emb

        sa_tokens = jnp.concatenate([state_token, action_tokens], axis=1)
        hidden = self.dit(
            hidden_states=sa_tokens,
            encoder_hidden_states=memory_tokens,
            timestep_ids=timestep_ids,
            train=train,
        )
        pred = self.action_decoder(hidden)
        return pred[:, -self.config.action_horizon :, :]

    @nn.compact
    def __call__(
        self,
        backbone_outputs: Dict[str, jnp.ndarray],
        state: jnp.ndarray,
        batch: Dict[str, jnp.ndarray],
        train: bool,
    ) -> Dict[str, jnp.ndarray]:
        memory_tokens = self.memory_processor(backbone_outputs["backbone_features"], train=train)
        actions = batch.get("actions")
        action_mask = batch.get(
            "action_mask",
            jnp.ones((state.shape[0], self.config.action_horizon, self.config.action_dim), dtype=state.dtype),
        )

        if train:
            rng = self.make_rng("dropout")
            noise = jax.random.normal(rng, actions.shape, dtype=actions.dtype)
            t = self._sample_t(rng, actions.shape[0])
            t_expanded = t[:, None, None]
            noisy_actions = (1.0 - t_expanded) * noise + t_expanded * actions
            velocity = actions - noise
            timestep_ids = jnp.asarray(
                jnp.floor(t * self.config.num_timestep_buckets),
                dtype=jnp.int32,
            )
            pred_velocity = self._predict_velocity(
                memory_tokens=memory_tokens,
                state=state,
                noisy_actions=noisy_actions,
                timestep_ids=timestep_ids,
                train=True,
            )
            loss = jnp.sum(((pred_velocity - velocity) ** 2) * action_mask) / jnp.maximum(jnp.sum(action_mask), 1.0)
            return {
                "action_pred": noisy_actions + pred_velocity,
                "pred_velocity": pred_velocity,
                "flow_matching_loss": loss,
                "loss": loss,
            }

        inference_seed = batch.get("inference_seed")
        if inference_seed is None:
            seed_key = jax.random.PRNGKey(0)
        else:
            seed_value = jnp.asarray(inference_seed).reshape(-1)[0].astype(jnp.uint32)
            seed_key = jax.random.PRNGKey(seed_value)

        current = jax.random.normal(
            seed_key,
            (state.shape[0], self.config.action_horizon, self.config.action_dim),
            dtype=state.dtype,
        )
        num_steps = max(1, self.config.num_inference_steps)
        dt = 1.0 / num_steps
        pred_velocity = jnp.zeros_like(current)
        for step in range(num_steps):
            t = jnp.full((state.shape[0],), step * dt, dtype=state.dtype)
            timestep_ids = jnp.asarray(jnp.floor(t * self.config.num_timestep_buckets), dtype=jnp.int32)
            pred_velocity = self._predict_velocity(
                memory_tokens=memory_tokens,
                state=state,
                noisy_actions=current,
                timestep_ids=timestep_ids,
                train=False,
            )
            current = current + dt * pred_velocity

        return {
            "action_pred": current,
            "pred_velocity": pred_velocity,
        }
