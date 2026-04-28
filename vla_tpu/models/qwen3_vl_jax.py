from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from vla_tpu.configs.base import BackboneConfig


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        x32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        normed = x32 * lax.rsqrt(var + self.epsilon)
        return (normed * scale).astype(x.dtype)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rope_frequencies(positions: jnp.ndarray, dim: int, theta: float) -> jnp.ndarray:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.einsum("...i,d->...id", positions.astype(jnp.float32), inv_freq)
    return jnp.concatenate([freqs, freqs], axis=-1)


def apply_rope_1d(x: jnp.ndarray, positions: jnp.ndarray, theta: float, scaling_factor: float) -> jnp.ndarray:
    freqs = rope_frequencies(positions[..., None] / scaling_factor, x.shape[-1], theta).squeeze(-2)
    cos = jnp.cos(freqs)[None, :, None, :]
    sin = jnp.sin(freqs)[None, :, None, :]
    return (x * cos) + (rotate_half(x) * sin)


def apply_interleaved_mrope(freqs: jnp.ndarray, mrope_section: tuple[int, int, int]) -> jnp.ndarray:
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        freqs_t = freqs_t.at[..., slice(offset, length, 3)].set(freqs[dim, ..., slice(offset, length, 3)])
    return freqs_t


def build_text_position_embeddings(
    head_dim: int,
    position_ids: jnp.ndarray,
    rope_theta: float,
    mrope_section: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.einsum("cbs,d->cbsd", position_ids.astype(jnp.float32), inv_freq)
    freqs = apply_interleaved_mrope(freqs, mrope_section)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def get_vision_position_ids(
    start_position: int,
    grid_thw: jnp.ndarray,
    spatial_merge_size: int,
) -> jnp.ndarray:
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = jnp.tile(jnp.arange(start_position, start_position + llm_grid_w), llm_grid_h * llm_grid_t)
    position_height = jnp.repeat(jnp.arange(start_position, start_position + llm_grid_h), llm_grid_w * llm_grid_t)
    position_temporal = jnp.full((image_seq_length,), start_position, dtype=jnp.int32)
    return jnp.stack([position_temporal, position_height, position_width], axis=0)


def get_rope_index(
    input_ids: jnp.ndarray,
    mm_token_type_ids: jnp.ndarray,
    image_grid_thw: jnp.ndarray,
    attention_mask: jnp.ndarray,
    spatial_merge_size: int,
) -> jnp.ndarray:
    batch_size, seq_len = input_ids.shape
    position_ids = jnp.zeros((3, batch_size, seq_len), dtype=jnp.int32)
    for batch_idx in range(batch_size):
        current_types = mm_token_type_ids[batch_idx][attention_mask[batch_idx].astype(bool)]
        current_pos = 0
        pos_list = []
        start = 0
        img_idx = 0
        while start < current_types.shape[0]:
            token_type = int(current_types[start])
            end = start
            while end < current_types.shape[0] and int(current_types[end]) == token_type:
                end += 1
            if token_type == 0:
                text_len = end - start
                pos_list.append(
                    jnp.broadcast_to(
                        jnp.arange(text_len, dtype=jnp.int32)[None, :] + current_pos,
                        (3, text_len),
                    )
                )
                current_pos += text_len
            else:
                vision_pos = get_vision_position_ids(
                    current_pos,
                    image_grid_thw[img_idx],
                    spatial_merge_size,
                )
                pos_list.append(vision_pos)
                current_pos += max(int(image_grid_thw[img_idx, 1]), int(image_grid_thw[img_idx, 2])) // spatial_merge_size
                img_idx += 1
            start = end
        llm_positions = jnp.concatenate(pos_list, axis=1)
        valid_idx = jnp.where(attention_mask[batch_idx].astype(bool), size=llm_positions.shape[1], fill_value=0)[0]
        position_ids = position_ids.at[:, batch_idx, valid_idx].set(llm_positions)
    return position_ids


def apply_rope_2d(x: jnp.ndarray, row_ids: jnp.ndarray, col_ids: jnp.ndarray, theta: float) -> jnp.ndarray:
    half_dim = x.shape[-1] // 2
    row_freqs = rope_frequencies(row_ids, half_dim, theta)
    col_freqs = rope_frequencies(col_ids, half_dim, theta)
    freqs = jnp.concatenate([row_freqs, col_freqs], axis=-1)
    cos = jnp.cos(freqs)[None, :, None, :]
    sin = jnp.sin(freqs)[None, :, None, :]
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_vision(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


class VisionMLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.intermediate_size, name="linear_fc1", precision=lax.Precision.HIGHEST)(x)
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Dense(self.hidden_size, name="linear_fc2", precision=lax.Precision.HIGHEST)(x)
        return x


class VisionAttention(nn.Module):
    hidden_size: int
    num_heads: int
    rope_theta: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, row_ids: jnp.ndarray, col_ids: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.hidden_size // self.num_heads
        qkv = nn.DenseGeneral(
            features=(3, self.num_heads, head_dim),
            axis=-1,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        q = apply_rope_2d(q, row_ids, col_ids, self.rope_theta)
        k = apply_rope_2d(k, row_ids, col_ids, self.rope_theta)

        scale = head_dim ** -0.5
        attn = jnp.einsum("bthd,bshd->bhts", q, k) * scale
        attn = nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhts,bshd->bthd", attn, v)
        return nn.DenseGeneral(
            features=self.hidden_size,
            axis=(-2, -1),
            name="proj",
        )(out)


class VisionBlock(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, row_ids: jnp.ndarray, col_ids: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm(epsilon=1e-6, name="norm1")(x)
        x = VisionAttention(
            hidden_size=self.config.vision_hidden_size,
            num_heads=self.config.vision_num_heads,
            rope_theta=self.config.vision_rope_theta,
            name="attn",
        )(x, row_ids=row_ids, col_ids=col_ids)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=1e-6, name="norm2")(x)
        x = VisionMLP(
            hidden_size=self.config.vision_hidden_size,
            intermediate_size=self.config.vision_intermediate_size,
            name="mlp",
        )(x)
        return residual + x


class VisionPatchMerger(nn.Module):
    config: BackboneConfig
    use_postshuffle_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, grid_h: int, grid_w: int) -> jnp.ndarray:
        merge = self.config.vision_spatial_merge_size
        hidden = self.config.vision_hidden_size
        if self.use_postshuffle_norm:
            x = x.reshape(x.shape[0], grid_h // merge, merge, grid_w // merge, merge, hidden)
            x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
            x = x.reshape(x.shape[0], -1, hidden * merge * merge)
            norm_dim = hidden * merge * merge
            x = nn.LayerNorm(epsilon=1e-6, name="norm")(x.reshape(-1, norm_dim)).reshape(x.shape[0], -1, norm_dim)
        else:
            x = nn.LayerNorm(epsilon=1e-6, name="norm")(x.reshape(-1, hidden)).reshape(x.shape[0], grid_h * grid_w, hidden)
            x = x.reshape(x.shape[0], grid_h // merge, merge, grid_w // merge, merge, hidden)
            x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
            x = x.reshape(x.shape[0], -1, hidden * merge * merge)
            norm_dim = hidden * merge * merge
        x = nn.Dense(norm_dim, name="linear_fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.hidden_size, name="linear_fc2")(x)
        return x


class ExactVisionPatchMerger(nn.Module):
    config: BackboneConfig
    use_postshuffle_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        merge_hidden = self.config.vision_hidden_size * (self.config.vision_spatial_merge_size**2)
        if self.use_postshuffle_norm:
            x = x.reshape(-1, merge_hidden)
            x = nn.LayerNorm(epsilon=1e-6, name="norm")(x)
        else:
            x = nn.LayerNorm(epsilon=1e-6, name="norm")(x)
            x = x.reshape(-1, merge_hidden)
        x = nn.Dense(merge_hidden, name="linear_fc1")(x)
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Dense(self.config.hidden_size, name="linear_fc2")(x)
        return x


class ExactVisionRotaryEmbedding(nn.Module):
    dim: int
    theta: float

    def __call__(self, seqlen: int) -> jnp.ndarray:
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        return jnp.outer(seq, inv_freq)


def _build_vision_rotary_embeddings(
    grid_thw: jnp.ndarray,
    spatial_merge_size: int,
    rotary_table: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    embeddings = _build_vision_rotary_raw_embeddings(
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        rotary_table=rotary_table,
    )
    emb = jnp.concatenate([embeddings, embeddings], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _build_vision_rotary_raw_embeddings(
    grid_thw: jnp.ndarray,
    spatial_merge_size: int,
    rotary_table: jnp.ndarray,
) -> jnp.ndarray:
    grid_list = np.asarray(grid_thw).tolist()
    max_hw = max(max(h, w) for _, h, w in grid_list)
    freq_table = rotary_table[:max_hw]

    pos_ids = []
    merge_size = spatial_merge_size
    for num_frames, height, width in grid_list:
        merged_h = height // merge_size
        merged_w = width // merge_size
        block_rows = np.arange(merged_h)
        block_cols = np.arange(merged_w)
        intra_row = np.arange(merge_size)
        intra_col = np.arange(merge_size)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = np.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
        col_idx = np.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
        coords = np.stack((row_idx, col_idx), axis=-1)
        if num_frames > 1:
            coords = np.tile(coords, (num_frames, 1))
        pos_ids.append(coords)

    pos_ids = np.concatenate(pos_ids, axis=0)
    embeddings = np.asarray(freq_table)[pos_ids].reshape(pos_ids.shape[0], -1)
    return jnp.asarray(embeddings, dtype=jnp.float32)


class VisionModel(nn.Module):
    config: BackboneConfig

    def _interpolate_pos_embed(self, grid_h: int, grid_w: int) -> jnp.ndarray:
        base_side = int(self.config.vision_num_position_embeddings**0.5)
        table = self.param(
            "pos_embed_table",
            nn.initializers.normal(stddev=0.02),
            (base_side, base_side, self.config.vision_hidden_size),
        )
        resized = jax.image.resize(
            table,
            shape=(grid_h, grid_w, self.config.vision_hidden_size),
            method="bilinear",
        )
        return resized.reshape(grid_h * grid_w, self.config.vision_hidden_size)

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        batch_size, num_cameras, height, width, channels = images.shape
        patch = self.config.vision_patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by vision_patch_size={patch}"
            )

        flat_images = images.reshape(batch_size * num_cameras, height, width, channels)
        patch_tokens = nn.Conv(
            features=self.config.vision_hidden_size,
            kernel_size=(patch, patch),
            strides=(patch, patch),
            padding="VALID",
            name="patch_embed",
        )(flat_images)
        grid_h, grid_w = patch_tokens.shape[1], patch_tokens.shape[2]
        tokens = patch_tokens.reshape(batch_size * num_cameras, grid_h * grid_w, self.config.vision_hidden_size)
        pos_embed = self._interpolate_pos_embed(grid_h, grid_w)
        tokens = tokens + pos_embed[None, :, :]

        row_ids = jnp.repeat(jnp.arange(grid_h), grid_w)
        col_ids = jnp.tile(jnp.arange(grid_w), grid_h)

        deepstack_features = []
        for idx in range(self.config.vision_depth):
            tokens = VisionBlock(self.config, name=f"block_{idx}")(tokens, row_ids=row_ids, col_ids=col_ids)
            if idx in self.config.deepstack_visual_indexes:
                merged = VisionPatchMerger(
                    self.config,
                    use_postshuffle_norm=True,
                    name=f"deepstack_merger_{idx}",
                )(tokens, grid_h=grid_h, grid_w=grid_w)
                deepstack_features.append(merged.reshape(batch_size, -1, self.config.hidden_size))

        merged_tokens = VisionPatchMerger(self.config, name="merger")(tokens, grid_h=grid_h, grid_w=grid_w)
        merged_tokens = merged_tokens.reshape(batch_size, -1, self.config.hidden_size)
        return merged_tokens, deepstack_features


class ExactVisionPatchEmbed(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=0.02),
            (
                2,
                self.config.vision_patch_size,
                self.config.vision_patch_size,
                self.config.image_channels,
                self.config.vision_hidden_size,
            ),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.config.vision_hidden_size,))
        x = pixel_values.reshape(
            -1,
            self.config.image_channels,
            2,
            self.config.vision_patch_size,
            self.config.vision_patch_size,
        )
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        return jnp.einsum("nthwc,thwco->no", x, kernel) + bias


class ExactVisionModel(nn.Module):
    config: BackboneConfig

    def _interpolate_pos_embed_exact(self, grid_thw: jnp.ndarray) -> jnp.ndarray:
        grid_list = np.asarray(grid_thw).tolist()
        grid_ts = [row[0] for row in grid_list]
        grid_hs = [row[1] for row in grid_list]
        grid_ws = [row[2] for row in grid_list]
        base_side = int(self.config.vision_num_position_embeddings**0.5)
        table = self.param(
            "pos_embed_table",
            nn.initializers.normal(stddev=0.02),
            (base_side * base_side, self.config.vision_hidden_size),
        )
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, h, w in grid_list:
            h_idxs = np.linspace(0, base_side - 1, h)
            w_idxs = np.linspace(0, base_side - 1, w)
            h_floor = h_idxs.astype(np.int32)
            w_floor = w_idxs.astype(np.int32)
            h_ceil = np.clip(h_floor + 1, 0, base_side - 1)
            w_ceil = np.clip(w_floor + 1, 0, base_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * base_side
            base_h_ceil = h_ceil * base_side
            indices = [
                (base_h[:, None] + w_floor[None, :]).reshape(-1),
                (base_h[:, None] + w_ceil[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_floor[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_ceil[None, :]).reshape(-1),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                (dh[:, None] * dw[None, :]).reshape(-1),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = np.asarray(idx_list, dtype=np.int32)
        weight_tensor = jnp.asarray(weight_list, dtype=table.dtype)
        pos_embeds = table[idx_tensor] * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        splits = [h * w for h, w in zip(grid_hs, grid_ws)]
        split_embeds = []
        start = 0
        for split in splits:
            split_embeds.append(patch_pos_embeds[start : start + split])
            start += split
        permuted = []
        merge = self.config.vision_spatial_merge_size
        for pos_embed, t, h, w in zip(split_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = jnp.repeat(pos_embed, t, axis=0)
            pos_embed = pos_embed.reshape(t, h // merge, merge, w // merge, merge, -1)
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5)).reshape(-1, self.config.vision_hidden_size)
            permuted.append(pos_embed)
        return jnp.concatenate(permuted, axis=0)

    @nn.compact
    def __call__(
        self,
        pixel_values: jnp.ndarray,
        grid_thw: jnp.ndarray,
        return_intermediates: bool = False,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]] | dict[str, jnp.ndarray | list[jnp.ndarray]]:
        hidden_states = ExactVisionPatchEmbed(self.config, name="patch_embed")(pixel_values)
        if self.config.fixed_image_grid_thw:
            base_grid = np.asarray(self.config.fixed_image_grid_thw, dtype=np.int32)
            repeats = grid_thw.shape[0] // base_grid.shape[0]
            grid_np = np.tile(base_grid, (repeats, 1))
        else:
            grid_np = np.asarray(grid_thw)
        pos_embeds = self._interpolate_pos_embed_exact(grid_np)
        patch_tokens = hidden_states
        hidden_states = hidden_states + pos_embeds
        patch_pos_tokens = hidden_states
        head_dim = self.config.vision_hidden_size // self.config.vision_num_heads
        max_grid_hw = int(np.max(grid_np[:, 1:]))
        if self.config.fixed_image_grid_thw:
            rotary_freq = 1.0 / (
                self.config.vision_rope_theta
                ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2))
            )
            rotary_table_np = np.outer(np.arange(max_grid_hw, dtype=np.float32), rotary_freq)
            position_embeddings = _build_vision_rotary_embeddings(
                grid_np,
                self.config.vision_spatial_merge_size,
                rotary_table_np,
            )
        else:
            rotary_table = ExactVisionRotaryEmbedding(
                dim=head_dim // 2,
                theta=self.config.vision_rope_theta,
                name="rotary_pos_emb",
            )(max_grid_hw)
            grid_thw_jnp = jnp.asarray(grid_np, dtype=jnp.int32)
            position_embeddings = _build_vision_rotary_embeddings(
                grid_thw_jnp,
                self.config.vision_spatial_merge_size,
                rotary_table,
            )
        lengths = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0])
        cu_seqlens_np = np.pad(np.cumsum(lengths, dtype=np.int32), (1, 0))
        cu_seqlens = cu_seqlens_np if self.config.fixed_image_grid_thw else jnp.asarray(cu_seqlens_np, dtype=jnp.int32)
        deepstack_features = []
        block_hidden_states = []
        for idx in range(self.config.vision_depth):
            hidden_states = ExactVisionBlock(self.config, name=f"block_{idx}")(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if return_intermediates:
                block_hidden_states.append(hidden_states)
            if idx in self.config.deepstack_visual_indexes:
                merged = ExactVisionPatchMerger(
                    self.config,
                    use_postshuffle_norm=True,
                    name=f"deepstack_merger_{idx}",
                )(hidden_states)
                deepstack_features.append(merged[None, :, :])
        pre_merge_hidden = hidden_states
        merged_tokens = ExactVisionPatchMerger(self.config, name="merger")(hidden_states)
        merged_tokens = merged_tokens[None, :, :]
        if return_intermediates:
            return {
                "pooler_output": merged_tokens,
                "deepstack_features": deepstack_features,
                "patch_tokens": patch_tokens,
                "pos_embeds": pos_embeds,
                "patch_pos_tokens": patch_pos_tokens,
                "last_hidden_state": pre_merge_hidden,
                "block_hidden_states": block_hidden_states,
            }
        return merged_tokens, deepstack_features


class ExactVisionAttention(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        seq_length = hidden_states.shape[0]
        head_dim = self.hidden_size // self.num_heads
        qkv = nn.DenseGeneral(
            features=(3, self.num_heads, head_dim),
            axis=-1,
            name="qkv",
            precision=lax.Precision.HIGHEST,
        )(hidden_states)
        q, k, v = jnp.split(qkv, 3, axis=1)
        q = jnp.squeeze(q, axis=1)
        k = jnp.squeeze(k, axis=1)
        v = jnp.squeeze(v, axis=1)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        outputs = []
        scale = head_dim ** -0.5
        for start, end in zip(np.asarray(cu_seqlens[:-1]), np.asarray(cu_seqlens[1:])):
            qs = q[start:end].transpose(1, 0, 2)[None, ...]
            ks = k[start:end].transpose(1, 0, 2)[None, ...]
            vs = v[start:end].transpose(1, 0, 2)[None, ...]
            attn_logits = jnp.matmul(qs, jnp.swapaxes(ks, -1, -2), precision=lax.Precision.HIGHEST) * scale
            attn = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(hidden_states.dtype)
            out = jnp.matmul(attn, vs, precision=lax.Precision.HIGHEST).transpose(0, 2, 1, 3).reshape(end - start, -1)
            outputs.append(out)

        attn_output = jnp.concatenate(outputs, axis=0).reshape(seq_length, -1)
        return nn.Dense(self.hidden_size, name="proj", precision=lax.Precision.HIGHEST)(attn_output)


class ExactVisionBlock(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        residual = hidden_states
        hidden_states = nn.LayerNorm(epsilon=1e-6, name="norm1")(hidden_states)
        hidden_states = ExactVisionAttention(
            hidden_size=self.config.vision_hidden_size,
            num_heads=self.config.vision_num_heads,
            name="attn",
        )(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = nn.LayerNorm(epsilon=1e-6, name="norm2")(hidden_states)
        hidden_states = VisionMLP(
            hidden_size=self.config.vision_hidden_size,
            intermediate_size=self.config.vision_intermediate_size,
            name="mlp",
        )(hidden_states)
        return residual + hidden_states


class GroupedAttention(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        head_dim = self.config.hidden_size // self.config.num_heads
        q = nn.Dense(
            self.config.num_heads * head_dim,
            use_bias=False,
            name="q_proj",
        )(x).reshape(x.shape[0], x.shape[1], self.config.num_heads, head_dim)
        k = nn.Dense(
            self.config.num_key_value_heads * head_dim,
            use_bias=False,
            name="k_proj",
        )(x).reshape(x.shape[0], x.shape[1], self.config.num_key_value_heads, head_dim)
        v = nn.Dense(
            self.config.num_key_value_heads * head_dim,
            use_bias=False,
            name="v_proj",
        )(x).reshape(x.shape[0], x.shape[1], self.config.num_key_value_heads, head_dim)

        q = RMSNorm(self.config.layer_norm_epsilon, name="q_norm")(q).transpose(0, 2, 1, 3)
        k = RMSNorm(self.config.layer_norm_epsilon, name="k_norm")(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        q = (q * cos[:, None, :, :]) + (rotate_half(q) * sin[:, None, :, :])
        k = (k * cos[:, None, :, :]) + (rotate_half(k) * sin[:, None, :, :])

        repeats = self.config.num_heads // self.config.num_key_value_heads
        k = jnp.repeat(k, repeats, axis=1)
        v = jnp.repeat(v, repeats, axis=1)

        scale = head_dim ** -0.5
        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn_logits = attn_logits + attention_mask
        attn = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return nn.Dense(
            self.config.hidden_size,
            use_bias=False,
            name="o_proj",
        )(out)


class TextMLP(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        intermediate = self.config.hidden_size * self.config.mlp_ratio
        gate = nn.Dense(intermediate, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(intermediate, use_bias=False, name="up_proj")(x)
        gate = nn.silu(gate.astype(jnp.float32)).astype(x.dtype)
        out = gate * up
        return nn.Dense(self.config.hidden_size, use_bias=False, name="down_proj")(out)


class TextDecoderLayer(nn.Module):
    config: BackboneConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        residual = x
        x = RMSNorm(self.config.layer_norm_epsilon, name="input_layernorm")(x)
        x = GroupedAttention(self.config, name="self_attn")(
            x,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        x = residual + x

        residual = x
        x = RMSNorm(self.config.layer_norm_epsilon, name="post_attention_layernorm")(x)
        x = TextMLP(self.config, name="mlp")(x)
        return residual + x


class TokenResampler(nn.Module):
    hidden_size: int
    num_tokens: int
    num_heads: int

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        batch_size = tokens.shape[0]
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
            name="cross_attn",
        )(queries, tokens)
        return RMSNorm(name="out_norm")(queries + pooled)


def _scatter_visual_tokens(
    base_tokens: jnp.ndarray,
    image_token_mask: jnp.ndarray,
    visual_tokens: jnp.ndarray,
) -> jnp.ndarray:
    batch_size, seq_len, hidden_size = base_tokens.shape
    flat_tokens = base_tokens.reshape(batch_size * seq_len, hidden_size)
    flat_mask = image_token_mask.reshape(batch_size * seq_len)
    num_visual = visual_tokens.shape[0] * visual_tokens.shape[1]
    flat_visual = visual_tokens.reshape(num_visual, hidden_size)
    scatter_idx = jnp.nonzero(flat_mask, size=num_visual, fill_value=0)[0]
    flat_tokens = flat_tokens.at[scatter_idx].set(flat_visual)
    return flat_tokens.reshape(batch_size, seq_len, hidden_size)


def _deepstack_process(
    hidden_states: jnp.ndarray,
    visual_pos_masks: jnp.ndarray,
    visual_embeds: jnp.ndarray,
) -> jnp.ndarray:
    batch_size, seq_len, hidden_size = hidden_states.shape
    flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_size)
    flat_mask = visual_pos_masks.reshape(batch_size * seq_len)
    visual_embeds = visual_embeds.reshape(-1, hidden_size)
    scatter_idx = jnp.nonzero(flat_mask, size=visual_embeds.shape[0], fill_value=0)[0]
    flat_hidden = flat_hidden.at[scatter_idx].add(visual_embeds)
    return flat_hidden.reshape(batch_size, seq_len, hidden_size)


class JAXQwen3VLPureBackbone(nn.Module):
    """Pure Qwen3-VL-style backbone: image + text only, no VLA-specific bridge layers."""

    config: BackboneConfig

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        train: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        del train
        instruction_tokens = batch["input_ids"] if "input_ids" in batch else batch["instruction_tokens"]
        attention_mask = batch["attention_mask"].astype(bool)
        mm_token_type_ids = batch["mm_token_type_ids"].astype(jnp.int32)
        if self.config.fixed_image_grid_thw:
            base_grid = np.asarray(self.config.fixed_image_grid_thw, dtype=np.int32)
            num_rows = batch["image_grid_thw"].shape[0]
            repeats = num_rows // base_grid.shape[0]
            image_grid_thw = jnp.asarray(np.tile(base_grid, (repeats, 1)), dtype=jnp.int32)
        else:
            image_grid_thw = batch["image_grid_thw"].astype(jnp.int32)

        if "pixel_values" in batch:
            visual_tokens, deepstack_visual_embeds = ExactVisionModel(self.config, name="visual")(
                batch["pixel_values"].astype(jnp.float32),
                image_grid_thw,
            )
        else:
            images = batch["images"].astype(jnp.float32) / 255.0
            visual_tokens, deepstack_visual_embeds = VisionModel(self.config, name="visual")(images)
        text_tokens = nn.Embed(
            num_embeddings=self.config.text_vocab_size,
            features=self.config.hidden_size,
            name="token_embed",
        )(instruction_tokens)
        image_token_id = batch["image_token_id"].astype(jnp.int32) if "image_token_id" in batch else None
        if image_token_id is None:
            image_token_mask = batch["image_token_mask"].astype(bool)
        else:
            image_token_mask = instruction_tokens == image_token_id.reshape(())

        tokens = _scatter_visual_tokens(text_tokens, image_token_mask, visual_tokens)
        visual_pos_masks = image_token_mask

        token_valid = attention_mask
        causal = jnp.tril(jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=bool))
        keep_mask = token_valid[:, None, :] & token_valid[:, :, None] & causal[None, :, :]
        attention_mask = jnp.where(
            keep_mask,
            jnp.zeros((), dtype=tokens.dtype),
            jnp.full((), -1e30, dtype=tokens.dtype),
        )[:, None, :, :]
        if "position_ids" in batch:
            position_ids = batch["position_ids"].astype(jnp.int32)
        else:
            position_ids = get_rope_index(
                instruction_tokens,
                mm_token_type_ids,
                image_grid_thw,
                token_valid.astype(jnp.int32),
                self.config.vision_spatial_merge_size,
            )
        position_embeddings = build_text_position_embeddings(
            self.config.hidden_size // self.config.num_heads,
            position_ids,
            self.config.rope_max_wavelength,
            self.config.mrope_section,
        )

        hidden_states = tokens
        for layer_idx in range(self.config.num_layers):
            hidden_states = TextDecoderLayer(self.config, name=f"decoder_{layer_idx}")(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = _deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = RMSNorm(self.config.layer_norm_epsilon, name="final_norm")(hidden_states)
        outputs = {
            "hidden_states": hidden_states,
            "backbone_attention_mask": token_valid.astype(jnp.int32),
            "visual_token_mask": image_token_mask.astype(jnp.int32),
        }
        if return_intermediates:
            outputs["visual_tokens"] = visual_tokens
        return outputs


class JAXQwen3VLFullAdapter(nn.Module):
    """Qwen3-VL-style full JAX backbone with vision encoder and text decoder."""

    config: BackboneConfig

    @nn.compact
    def __call__(
        self,
        batch: Dict[str, jnp.ndarray],
        train: bool = False,
        return_intermediates: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        del train
        images = batch["images"].astype(jnp.float32) / 255.0
        state = batch["state"].astype(jnp.float32)
        instruction_tokens = batch["instruction_tokens"]

        visual_tokens, deepstack_visual_embeds = VisionModel(self.config, name="visual")(images)
        state_token = nn.Dense(self.config.hidden_size, use_bias=False, name="state_proj")(state)[:, None, :]
        text_tokens = nn.Embed(
            num_embeddings=self.config.text_vocab_size,
            features=self.config.hidden_size,
            name="token_embed",
        )(instruction_tokens)

        prefix_tokens = jnp.concatenate([state_token, visual_tokens], axis=1)
        tokens = jnp.concatenate([prefix_tokens, text_tokens], axis=1)

        prefix_valid = jnp.ones((tokens.shape[0], prefix_tokens.shape[1]), dtype=bool)
        text_valid = instruction_tokens != 0
        token_valid = jnp.concatenate([prefix_valid, text_valid], axis=1)
        causal = jnp.tril(jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=bool))
        keep_mask = token_valid[:, None, :] & token_valid[:, :, None] & causal[None, :, :]
        attention_mask = jnp.where(
            keep_mask,
            jnp.zeros((), dtype=tokens.dtype),
            jnp.full((), -1e30, dtype=tokens.dtype),
        )[:, None, :, :]
        position_ids = jnp.broadcast_to(
            jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, None, :],
            (3, tokens.shape[0], tokens.shape[1]),
        )
        position_embeddings = build_text_position_embeddings(
            self.config.hidden_size // self.config.num_heads,
            position_ids,
            self.config.rope_max_wavelength,
            self.config.mrope_section,
        )

        visual_start = 1
        visual_end = 1 + visual_tokens.shape[1]
        visual_mask = jnp.zeros((tokens.shape[0], tokens.shape[1], 1), dtype=tokens.dtype)
        visual_mask = visual_mask.at[:, visual_start:visual_end, :].set(1.0)

        hidden_states = tokens
        for layer_idx in range(self.config.num_layers):
            hidden_states = TextDecoderLayer(self.config, name=f"decoder_{layer_idx}")(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            if layer_idx < len(deepstack_visual_embeds):
                deepstack = deepstack_visual_embeds[layer_idx]
                hidden_states = hidden_states + jnp.pad(
                    deepstack,
                    ((0, 0), (visual_start, hidden_states.shape[1] - visual_end), (0, 0)),
                ) * visual_mask

        hidden_states = RMSNorm(self.config.layer_norm_epsilon, name="final_norm")(hidden_states)
        memory_tokens = TokenResampler(
            hidden_size=self.config.hidden_size,
            num_tokens=self.config.n_visual_tokens,
            num_heads=self.config.num_heads,
            name="resampler",
        )(hidden_states)
        outputs = {
            "backbone_features": memory_tokens,
            "backbone_attention_mask": jnp.ones((tokens.shape[0], self.config.n_visual_tokens), dtype=jnp.int32),
        }
        if return_intermediates:
            outputs["hidden_states"] = hidden_states
            outputs["visual_tokens"] = visual_tokens
            outputs["prefix_tokens"] = prefix_tokens
        return outputs
