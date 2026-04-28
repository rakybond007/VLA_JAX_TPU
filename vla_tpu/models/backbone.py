from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from vla_tpu.configs.base import BackboneConfig
from vla_tpu.models.qwen3_vl_jax import JAXQwen3VLPureBackbone, JAXQwen3VLFullAdapter


class MLPBlock(nn.Module):
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


class EncoderBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = residual + x
        return MLPBlock(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="mlp",
        )(x, train=train)


class DummyQwenAdapter(nn.Module):
    """A small TPU-safe stand-in that preserves the backbone output contract."""

    config: BackboneConfig

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], train: bool = False) -> Dict[str, jnp.ndarray]:
        images = batch["images"].astype(jnp.float32) / 255.0
        state = batch["state"]
        instruction_tokens = batch["instruction_tokens"]

        bsz = images.shape[0]
        image_features = images.reshape(bsz, -1)
        image_features = nn.Dense(self.config.hidden_size, name="image_proj")(image_features)

        text_emb = nn.Embed(
            num_embeddings=self.config.text_vocab_size,
            features=self.config.hidden_size,
            name="token_embed",
        )(instruction_tokens)
        text_features = text_emb.mean(axis=1)

        state_features = nn.Dense(self.config.hidden_size, name="state_proj")(state)
        fused = image_features + text_features + state_features
        fused = nn.LayerNorm()(fused)

        token_bank = nn.Dense(
            self.config.n_visual_tokens * self.config.hidden_size,
            name="feature_bank",
        )(fused)
        token_bank = token_bank.reshape(
            bsz,
            self.config.n_visual_tokens,
            self.config.hidden_size,
        )
        return {
            "backbone_features": token_bank,
            "backbone_attention_mask": jnp.ones((bsz, self.config.n_visual_tokens), dtype=jnp.int32),
        }


class PureQwen3VLAdapter(nn.Module):
    """Thin adapter that exposes pure Qwen3-VL hidden states as action-head memory."""

    config: BackboneConfig

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], train: bool = False) -> Dict[str, jnp.ndarray]:
        outputs = JAXQwen3VLPureBackbone(self.config, name="pure_backbone")(
            batch,
            train=train,
            return_intermediates=False,
        )
        return {
            "backbone_features": outputs["hidden_states"],
            "backbone_attention_mask": outputs["backbone_attention_mask"],
            "visual_token_mask": outputs["visual_token_mask"],
        }


class ExperimentalJAXQwenAdapter(nn.Module):
    """A TPU-safe experimental JAX backbone with patch, text, and state fusion."""

    config: BackboneConfig

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], train: bool = False) -> Dict[str, jnp.ndarray]:
        images = batch["images"].astype(jnp.float32) / 255.0
        state = batch["state"].astype(jnp.float32)
        instruction_tokens = batch["instruction_tokens"]

        batch_size, num_cameras, height, width, channels = images.shape
        patch = self.config.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by patch_size={self.config.patch_size}"
            )

        flat_images = images.reshape(batch_size * num_cameras, height, width, channels)
        patch_tokens = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(patch, patch),
            strides=(patch, patch),
            padding="VALID",
            name="patch_embed",
        )(flat_images)
        patch_tokens = patch_tokens.reshape(batch_size, num_cameras, -1, self.config.hidden_size)
        image_tokens = patch_tokens.reshape(batch_size, -1, self.config.hidden_size)

        num_image_tokens = image_tokens.shape[1]
        image_pos = self.param(
            "image_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_image_tokens, self.config.hidden_size),
        )
        image_tokens = image_tokens + image_pos

        text_emb = nn.Embed(
            num_embeddings=self.config.text_vocab_size,
            features=self.config.hidden_size,
            name="token_embed",
        )(instruction_tokens)
        text_pos = self.param(
            "text_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.config.max_text_tokens, self.config.hidden_size),
        )
        text_tokens = text_emb + text_pos[:, : text_emb.shape[1], :]

        state_token = nn.Dense(self.config.hidden_size, name="state_proj")(state)[:, None, :]
        state_pos = self.param(
            "state_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.hidden_size),
        )
        state_token = state_token + state_pos

        tokens = jnp.concatenate([state_token, text_tokens, image_tokens], axis=1)
        for layer_idx in range(self.config.num_layers):
            tokens = EncoderBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout_rate=self.config.dropout_rate,
                name=f"encoder_{layer_idx}",
            )(tokens, train=train)
        tokens = nn.LayerNorm(name="final_norm")(tokens)

        if tokens.shape[1] > self.config.n_visual_tokens:
            tokens = tokens[:, -self.config.n_visual_tokens :, :]
        elif tokens.shape[1] < self.config.n_visual_tokens:
            pad = self.config.n_visual_tokens - tokens.shape[1]
            tokens = jnp.pad(tokens, ((0, 0), (0, pad), (0, 0)))

        return {
            "backbone_features": tokens,
            "backbone_attention_mask": jnp.ones((batch_size, self.config.n_visual_tokens), dtype=jnp.int32),
        }


class Qwen3RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        x32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        normed = x32 * lax.rsqrt(var + self.epsilon)
        return (normed * scale).astype(x.dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(
    x: jnp.ndarray,
    positions: jnp.ndarray,
    max_wavelength: int,
    scaling_factor: float,
) -> jnp.ndarray:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        max_wavelength ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    scaled_pos = positions.astype(jnp.float32) / scaling_factor
    freqs = jnp.einsum("t,d->td", scaled_pos, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[None, :, None, :]
    sin = jnp.sin(emb)[None, :, None, :]
    return (x * cos) + (_rotate_half(x) * sin)


class Qwen3GroupedAttention(nn.Module):
    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    rope_max_wavelength: int
    rope_scaling_factor: float
    dropout_rate: float
    layer_norm_epsilon: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray, train: bool) -> jnp.ndarray:
        head_dim = self.hidden_size // self.num_heads
        q = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="q_proj",
        )(x)
        k = nn.DenseGeneral(
            features=(self.num_key_value_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="k_proj",
        )(x)
        v = nn.DenseGeneral(
            features=(self.num_key_value_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="v_proj",
        )(x)

        q = Qwen3RMSNorm(self.layer_norm_epsilon, name="q_norm")(q)
        k = Qwen3RMSNorm(self.layer_norm_epsilon, name="k_norm")(k)

        positions = jnp.arange(x.shape[1], dtype=jnp.int32)
        q = apply_rope(q, positions, self.rope_max_wavelength, self.rope_scaling_factor)
        k = apply_rope(k, positions, self.rope_max_wavelength, self.rope_scaling_factor)

        repeats = self.num_heads // self.num_key_value_heads
        k = jnp.repeat(k, repeats, axis=2)
        v = jnp.repeat(v, repeats, axis=2)

        scale = head_dim ** -0.5
        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k) * scale
        attn_logits = jnp.where(attention_mask[:, None, :, :], attn_logits, -1e30)
        attn = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(x.dtype)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not train)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn, v)
        attn_out = nn.DenseGeneral(
            features=self.hidden_size,
            axis=(-2, -1),
            use_bias=False,
            name="o_proj",
        )(attn_out)
        attn_out = nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=not train)
        return attn_out


class Qwen3MLP(nn.Module):
    hidden_size: int
    mlp_ratio: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        intermediate_dim = self.hidden_size * self.mlp_ratio
        gate = nn.Dense(intermediate_dim, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(intermediate_dim, use_bias=False, name="up_proj")(x)
        gate = nn.silu(gate.astype(jnp.float32)).astype(x.dtype)
        out = gate * up
        out = nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)
        return out


class Qwen3DecoderBlock(nn.Module):
    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    mlp_ratio: int
    dropout_rate: float
    rope_max_wavelength: int
    rope_scaling_factor: float
    layer_norm_epsilon: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x
        x = Qwen3RMSNorm(self.layer_norm_epsilon, name="input_layernorm")(x)
        x = Qwen3GroupedAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            name="self_attn",
        )(x, attention_mask=attention_mask, train=train)
        x = residual + x

        residual = x
        x = Qwen3RMSNorm(self.layer_norm_epsilon, name="post_attention_layernorm")(x)
        x = Qwen3MLP(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="mlp",
        )(x, train=train)
        return residual + x


class TokenResampler(nn.Module):
    hidden_size: int
    num_tokens: int
    num_heads: int

    @nn.compact
    def __call__(self, memory_tokens: jnp.ndarray, train: bool) -> jnp.ndarray:
        batch_size = memory_tokens.shape[0]
        queries = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_tokens, self.hidden_size),
        )
        queries = jnp.broadcast_to(queries[None, :, :], (batch_size, self.num_tokens, self.hidden_size))
        pooled = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            dropout_rate=0.0,
            deterministic=not train,
            name="cross_attn",
        )(queries, memory_tokens)
        return Qwen3RMSNorm(name="out_norm")(queries + pooled)


class JAXQwen3Adapter(nn.Module):
    """Qwen3-style Flax port for the text tower with visual/state prefix tokens."""

    config: BackboneConfig

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], train: bool = False) -> Dict[str, jnp.ndarray]:
        images = batch["images"].astype(jnp.float32) / 255.0
        state = batch["state"].astype(jnp.float32)
        instruction_tokens = batch["instruction_tokens"]

        batch_size, num_cameras, height, width, channels = images.shape
        patch = self.config.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by patch_size={self.config.patch_size}"
            )
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.config.num_heads % self.config.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")

        flat_images = images.reshape(batch_size * num_cameras, height, width, channels)
        patch_tokens = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=(patch, patch),
            strides=(patch, patch),
            padding="VALID",
            name="patch_embed",
        )(flat_images)
        patch_tokens = patch_tokens.reshape(batch_size, num_cameras, -1, self.config.hidden_size)
        image_tokens = patch_tokens.reshape(batch_size, -1, self.config.hidden_size)

        state_token = nn.Dense(self.config.hidden_size, use_bias=False, name="state_proj")(state)[:, None, :]
        text_tokens = nn.Embed(
            num_embeddings=self.config.text_vocab_size,
            features=self.config.hidden_size,
            name="token_embed",
        )(instruction_tokens)

        prefix_tokens = jnp.concatenate([state_token, image_tokens], axis=1)
        tokens = jnp.concatenate([prefix_tokens, text_tokens], axis=1)

        prefix_valid = jnp.ones((batch_size, prefix_tokens.shape[1]), dtype=bool)
        text_valid = instruction_tokens != 0
        token_valid = jnp.concatenate([prefix_valid, text_valid], axis=1)
        causal = jnp.tril(jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=bool))
        attention_mask = token_valid[:, None, :] & token_valid[:, :, None] & causal[None, :, :]

        for layer_idx in range(self.config.num_layers):
            tokens = Qwen3DecoderBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout_rate=self.config.dropout_rate,
                rope_max_wavelength=self.config.rope_max_wavelength,
                rope_scaling_factor=self.config.rope_scaling_factor,
                layer_norm_epsilon=self.config.layer_norm_epsilon,
                name=f"decoder_{layer_idx}",
            )(tokens, attention_mask=attention_mask, train=train)
        tokens = Qwen3RMSNorm(self.config.layer_norm_epsilon, name="final_norm")(tokens)
        memory_tokens = TokenResampler(
            hidden_size=self.config.hidden_size,
            num_tokens=self.config.n_visual_tokens,
            num_heads=self.config.num_heads,
            name="resampler",
        )(tokens, train=train)
        return {
            "backbone_features": memory_tokens,
            "backbone_attention_mask": jnp.ones((batch_size, self.config.n_visual_tokens), dtype=jnp.int32),
        }


def build_backbone(config: BackboneConfig, name: str | None = None) -> nn.Module:
    if config.impl == "dummy_qwen_adapter":
        return DummyQwenAdapter(config, name=name)
    if config.impl == "jax_qwen_experimental":
        return ExperimentalJAXQwenAdapter(config, name=name)
    if config.impl == "jax_qwen3_adapter":
        return JAXQwen3Adapter(config, name=name)
    if config.impl == "jax_qwen3_vl_pure":
        return PureQwen3VLAdapter(config, name=name)
    if config.impl == "jax_qwen3_vl_full":
        return JAXQwen3VLFullAdapter(config, name=name)
    raise ValueError(f"Unsupported backbone impl: {config.impl}")
