from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import core

from vla_tpu.configs.base import BackboneConfig

try:
    import torch
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    AutoConfig = None
    Qwen3VLForConditionalGeneration = None


@dataclass
class WeightLoadSummary:
    loaded: int
    skipped: int
    mismatched: int
    skipped_keys: list[str]
    mismatched_keys: list[str]


def backbone_config_from_hf_qwen3_vl(
    model_name: str,
    *,
    n_visual_tokens: int = 192,
    max_text_tokens: int = 512,
) -> BackboneConfig:
    if AutoConfig is None:
        raise ImportError("transformers is required for HF-backed Qwen3-VL config loading.")

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    vision_cfg = cfg.vision_config
    text_cfg = cfg.text_config
    return BackboneConfig(
        name="jax_qwen3_vl_full",
        impl="jax_qwen3_vl_full",
        model_name=model_name,
        hidden_size=text_cfg.hidden_size,
        n_visual_tokens=n_visual_tokens,
        text_vocab_size=text_cfg.vocab_size,
        patch_size=vision_cfg.patch_size,
        num_layers=text_cfg.num_hidden_layers,
        num_heads=text_cfg.num_attention_heads,
        num_key_value_heads=text_cfg.num_key_value_heads,
        mlp_ratio=text_cfg.intermediate_size // text_cfg.hidden_size,
        max_text_tokens=max_text_tokens,
        rope_max_wavelength=text_cfg.rope_parameters["rope_theta"],
        rope_scaling_factor=1.0,
        mrope_section=tuple(text_cfg.rope_parameters.get("mrope_section", [24, 20, 20])),
        layer_norm_epsilon=text_cfg.rms_norm_eps,
        vision_hidden_size=vision_cfg.hidden_size,
        vision_intermediate_size=vision_cfg.intermediate_size,
        vision_num_heads=vision_cfg.num_heads,
        vision_depth=vision_cfg.depth,
        vision_patch_size=vision_cfg.patch_size,
        vision_spatial_merge_size=vision_cfg.spatial_merge_size,
        vision_num_position_embeddings=vision_cfg.num_position_embeddings,
        vision_rope_theta=float(getattr(vision_cfg, "rope_theta", 10000.0)),
        deepstack_visual_indexes=tuple(vision_cfg.deepstack_visual_indexes),
    )


def load_hf_qwen3_vl_state_dict(model_name: str, torch_dtype: str = "float32") -> dict[str, Any]:
    if Qwen3VLForConditionalGeneration is None:
        raise ImportError("transformers + torch are required for HF-backed Qwen3-VL state loading.")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        _attn_implementation="sdpa",
        torch_dtype=getattr(torch, torch_dtype),
    )
    return model.state_dict()


def _as_np_tensor(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _reshape_dense_general_in(weight: Any, *shape: int) -> np.ndarray:
    return _as_np_tensor(weight).T.reshape(shape)


def _reshape_dense_general_out(weight: Any, *shape: int) -> np.ndarray:
    return _as_np_tensor(weight).T.reshape(shape)


def _set_param(
    params: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
    summary: dict[str, list[str] | int],
) -> None:
    cursor = params
    for key in path[:-1]:
        cursor = cursor[key]
    leaf_key = path[-1]
    current = cursor[leaf_key]
    array = np.asarray(value, dtype=np.float32)
    if current.shape != array.shape:
        summary["mismatched"] += 1
        summary["mismatched_keys"].append(".".join(path))
        return
    cursor[leaf_key] = jnp.asarray(array, dtype=current.dtype)
    summary["loaded"] += 1


def _skip(summary: dict[str, list[str] | int], key: str) -> None:
    summary["skipped"] += 1
    summary["skipped_keys"].append(key)


def load_hf_weights_into_jax_qwen3_vl(params: core.FrozenDict[str, Any], state_dict: dict[str, Any]) -> tuple[core.FrozenDict[str, Any], WeightLoadSummary]:
    mutable = core.unfreeze(params)
    summary: dict[str, list[str] | int] = {
        "loaded": 0,
        "skipped": 0,
        "mismatched": 0,
        "skipped_keys": [],
        "mismatched_keys": [],
    }

    # Vision patch embed: HF uses Conv3d with temporal_patch_size=2, while our JAX port uses Conv2d.
    # We average the temporal kernel to recover a single-image compatible 2D patch kernel.
    patch_weight = _as_np_tensor(state_dict["model.visual.patch_embed.proj.weight"])
    if mutable["visual"]["patch_embed"]["kernel"].ndim == 5:
        patch_weight = np.transpose(patch_weight, (2, 3, 4, 1, 0))
    else:
        patch_weight = np.transpose(patch_weight.mean(axis=2), (2, 3, 1, 0))
    _set_param(mutable, ("visual", "patch_embed", "kernel"), patch_weight, summary)
    _set_param(mutable, ("visual", "patch_embed", "bias"), state_dict["model.visual.patch_embed.proj.bias"], summary)

    pos_weight = _as_np_tensor(state_dict["model.visual.pos_embed.weight"])
    current_pos = mutable["visual"]["pos_embed_table"]
    if current_pos.ndim == 3:
        side = int(pos_weight.shape[0] ** 0.5)
        pos_weight = pos_weight.reshape(side, side, pos_weight.shape[1])
    _set_param(mutable, ("visual", "pos_embed_table"), pos_weight, summary)

    vision_hidden = mutable["visual"]["patch_embed"]["bias"].shape[0]
    vision_heads = mutable["visual"]["block_0"]["attn"]["qkv"]["bias"].shape[1]
    vision_head_dim = mutable["visual"]["block_0"]["attn"]["qkv"]["bias"].shape[2]

    for idx in range(len([k for k in mutable["visual"] if k.startswith("block_")])):
        prefix = f"model.visual.blocks.{idx}"
        block_prefix = ("visual", f"block_{idx}")
        _set_param(mutable, block_prefix + ("norm1", "scale"), state_dict[f"{prefix}.norm1.weight"], summary)
        _set_param(mutable, block_prefix + ("norm1", "bias"), state_dict[f"{prefix}.norm1.bias"], summary)
        _set_param(mutable, block_prefix + ("norm2", "scale"), state_dict[f"{prefix}.norm2.weight"], summary)
        _set_param(mutable, block_prefix + ("norm2", "bias"), state_dict[f"{prefix}.norm2.bias"], summary)
        _set_param(
            mutable,
            block_prefix + ("attn", "qkv", "kernel"),
            _reshape_dense_general_in(
                state_dict[f"{prefix}.attn.qkv.weight"],
                vision_hidden,
                3,
                vision_heads,
                vision_head_dim,
            ),
            summary,
        )
        _set_param(
            mutable,
            block_prefix + ("attn", "qkv", "bias"),
            _as_np_tensor(state_dict[f"{prefix}.attn.qkv.bias"]).reshape(3, vision_heads, vision_head_dim),
            summary,
        )
        _set_param(
            mutable,
            block_prefix + ("attn", "proj", "kernel"),
            (
                _reshape_dense_general_out(
                    state_dict[f"{prefix}.attn.proj.weight"],
                    vision_heads,
                    vision_head_dim,
                    vision_hidden,
                )
                if mutable["visual"][f"block_{idx}"]["attn"]["proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.attn.proj.weight"]).T
            ),
            summary,
        )
        _set_param(mutable, block_prefix + ("attn", "proj", "bias"), state_dict[f"{prefix}.attn.proj.bias"], summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc1", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.linear_fc1.weight"]).T, summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc1", "bias"), state_dict[f"{prefix}.mlp.linear_fc1.bias"], summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc2", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.linear_fc2.weight"]).T, summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc2", "bias"), state_dict[f"{prefix}.mlp.linear_fc2.bias"], summary)

    deepstack_keys = sorted(
        (k for k in mutable["visual"] if k.startswith("deepstack_merger_")),
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    for idx, key in enumerate(deepstack_keys):
        prefix = f"model.visual.deepstack_merger_list.{idx}"
        _set_param(mutable, ("visual", key, "norm", "scale"), state_dict[f"{prefix}.norm.weight"], summary)
        _set_param(mutable, ("visual", key, "norm", "bias"), state_dict[f"{prefix}.norm.bias"], summary)
        _set_param(mutable, ("visual", key, "linear_fc1", "kernel"), _as_np_tensor(state_dict[f"{prefix}.linear_fc1.weight"]).T, summary)
        _set_param(mutable, ("visual", key, "linear_fc1", "bias"), state_dict[f"{prefix}.linear_fc1.bias"], summary)
        _set_param(mutable, ("visual", key, "linear_fc2", "kernel"), _as_np_tensor(state_dict[f"{prefix}.linear_fc2.weight"]).T, summary)
        _set_param(mutable, ("visual", key, "linear_fc2", "bias"), state_dict[f"{prefix}.linear_fc2.bias"], summary)

    _set_param(mutable, ("visual", "merger", "norm", "scale"), state_dict["model.visual.merger.norm.weight"], summary)
    _set_param(mutable, ("visual", "merger", "norm", "bias"), state_dict["model.visual.merger.norm.bias"], summary)
    _set_param(mutable, ("visual", "merger", "linear_fc1", "kernel"), _as_np_tensor(state_dict["model.visual.merger.linear_fc1.weight"]).T, summary)
    _set_param(mutable, ("visual", "merger", "linear_fc1", "bias"), state_dict["model.visual.merger.linear_fc1.bias"], summary)
    _set_param(mutable, ("visual", "merger", "linear_fc2", "kernel"), _as_np_tensor(state_dict["model.visual.merger.linear_fc2.weight"]).T, summary)
    _set_param(mutable, ("visual", "merger", "linear_fc2", "bias"), state_dict["model.visual.merger.linear_fc2.bias"], summary)

    _set_param(mutable, ("token_embed", "embedding"), state_dict["model.language_model.embed_tokens.weight"], summary)
    text_hidden = mutable["token_embed"]["embedding"].shape[1]
    q_kernel_shape = mutable["decoder_0"]["self_attn"]["q_proj"]["kernel"].shape
    k_kernel_shape = mutable["decoder_0"]["self_attn"]["k_proj"]["kernel"].shape
    if len(q_kernel_shape) == 3:
        text_heads = q_kernel_shape[1]
        text_head_dim = q_kernel_shape[2]
    else:
        text_heads = state_dict["model.language_model.layers.0.self_attn.q_proj.weight"].shape[0] // text_hidden
        text_head_dim = q_kernel_shape[1] // text_heads
    if len(k_kernel_shape) == 3:
        kv_heads = k_kernel_shape[1]
    else:
        kv_heads = state_dict["model.language_model.layers.0.self_attn.k_proj.weight"].shape[0] // text_head_dim
    mlp_intermediate = mutable["decoder_0"]["mlp"]["gate_proj"]["kernel"].shape[1]

    for idx in range(len([k for k in mutable if k.startswith("decoder_")])):
        prefix = f"model.language_model.layers.{idx}"
        layer = f"decoder_{idx}"
        _set_param(mutable, (layer, "input_layernorm", "scale"), state_dict[f"{prefix}.input_layernorm.weight"], summary)
        _set_param(mutable, (layer, "post_attention_layernorm", "scale"), state_dict[f"{prefix}.post_attention_layernorm.weight"], summary)
        _set_param(
            mutable,
            (layer, "self_attn", "q_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.q_proj.weight"],
                    text_hidden,
                    text_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["q_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.q_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "k_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.k_proj.weight"],
                    text_hidden,
                    kv_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["k_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.k_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "v_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.v_proj.weight"],
                    text_hidden,
                    kv_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["v_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.v_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "o_proj", "kernel"),
            (
                _reshape_dense_general_out(
                    state_dict[f"{prefix}.self_attn.o_proj.weight"],
                    text_heads,
                    text_head_dim,
                    text_hidden,
                )
                if mutable[layer]["self_attn"]["o_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.o_proj.weight"]).T
            ),
            summary,
        )
        _set_param(mutable, (layer, "self_attn", "q_norm", "scale"), state_dict[f"{prefix}.self_attn.q_norm.weight"], summary)
        _set_param(mutable, (layer, "self_attn", "k_norm", "scale"), state_dict[f"{prefix}.self_attn.k_norm.weight"], summary)
        _set_param(mutable, (layer, "mlp", "gate_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.gate_proj.weight"]).T, summary)
        _set_param(mutable, (layer, "mlp", "up_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.up_proj.weight"]).T, summary)
        _set_param(mutable, (layer, "mlp", "down_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.down_proj.weight"]).T, summary)

    _set_param(mutable, ("final_norm", "scale"), state_dict["model.language_model.norm.weight"], summary)

    _skip(summary, "state_proj")
    _skip(summary, "resampler")

    return core.freeze(mutable), WeightLoadSummary(
        loaded=int(summary["loaded"]),
        skipped=int(summary["skipped"]),
        mismatched=int(summary["mismatched"]),
        skipped_keys=list(summary["skipped_keys"]),
        mismatched_keys=list(summary["mismatched_keys"]),
    )


def load_hf_weights_into_jax_qwen3_vl_pure_backbone(
    params: core.FrozenDict[str, Any], state_dict: dict[str, Any]
) -> tuple[core.FrozenDict[str, Any], WeightLoadSummary]:
    mutable = core.unfreeze(params)
    summary: dict[str, list[str] | int] = {
        "loaded": 0,
        "skipped": 0,
        "mismatched": 0,
        "skipped_keys": [],
        "mismatched_keys": [],
    }

    patch_weight = _as_np_tensor(state_dict["model.visual.patch_embed.proj.weight"])
    if mutable["visual"]["patch_embed"]["kernel"].ndim == 5:
        patch_weight = np.transpose(patch_weight, (2, 3, 4, 1, 0))
    else:
        patch_weight = np.transpose(patch_weight.mean(axis=2), (2, 3, 1, 0))
    _set_param(mutable, ("visual", "patch_embed", "kernel"), patch_weight, summary)
    _set_param(mutable, ("visual", "patch_embed", "bias"), state_dict["model.visual.patch_embed.proj.bias"], summary)

    pos_weight = _as_np_tensor(state_dict["model.visual.pos_embed.weight"])
    current_pos = mutable["visual"]["pos_embed_table"]
    if current_pos.ndim == 3:
        side = int(pos_weight.shape[0] ** 0.5)
        pos_weight = pos_weight.reshape(side, side, pos_weight.shape[1])
    _set_param(mutable, ("visual", "pos_embed_table"), pos_weight, summary)

    vision_hidden = mutable["visual"]["patch_embed"]["bias"].shape[0]
    vision_heads = mutable["visual"]["block_0"]["attn"]["qkv"]["bias"].shape[1]
    vision_head_dim = mutable["visual"]["block_0"]["attn"]["qkv"]["bias"].shape[2]

    for idx in range(len([k for k in mutable["visual"] if k.startswith("block_")])):
        prefix = f"model.visual.blocks.{idx}"
        block_prefix = ("visual", f"block_{idx}")
        _set_param(mutable, block_prefix + ("norm1", "scale"), state_dict[f"{prefix}.norm1.weight"], summary)
        _set_param(mutable, block_prefix + ("norm1", "bias"), state_dict[f"{prefix}.norm1.bias"], summary)
        _set_param(mutable, block_prefix + ("norm2", "scale"), state_dict[f"{prefix}.norm2.weight"], summary)
        _set_param(mutable, block_prefix + ("norm2", "bias"), state_dict[f"{prefix}.norm2.bias"], summary)
        _set_param(
            mutable,
            block_prefix + ("attn", "qkv", "kernel"),
            _reshape_dense_general_in(
                state_dict[f"{prefix}.attn.qkv.weight"],
                vision_hidden,
                3,
                vision_heads,
                vision_head_dim,
            ),
            summary,
        )
        _set_param(
            mutable,
            block_prefix + ("attn", "qkv", "bias"),
            _as_np_tensor(state_dict[f"{prefix}.attn.qkv.bias"]).reshape(3, vision_heads, vision_head_dim),
            summary,
        )
        _set_param(
            mutable,
            block_prefix + ("attn", "proj", "kernel"),
            (
                _reshape_dense_general_out(
                    state_dict[f"{prefix}.attn.proj.weight"],
                    vision_heads,
                    vision_head_dim,
                    vision_hidden,
                )
                if mutable["visual"][f"block_{idx}"]["attn"]["proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.attn.proj.weight"]).T
            ),
            summary,
        )
        _set_param(mutable, block_prefix + ("attn", "proj", "bias"), state_dict[f"{prefix}.attn.proj.bias"], summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc1", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.linear_fc1.weight"]).T, summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc1", "bias"), state_dict[f"{prefix}.mlp.linear_fc1.bias"], summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc2", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.linear_fc2.weight"]).T, summary)
        _set_param(mutable, block_prefix + ("mlp", "linear_fc2", "bias"), state_dict[f"{prefix}.mlp.linear_fc2.bias"], summary)

    deepstack_keys = sorted(
        (k for k in mutable["visual"] if k.startswith("deepstack_merger_")),
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    for idx, key in enumerate(deepstack_keys):
        prefix = f"model.visual.deepstack_merger_list.{idx}"
        _set_param(mutable, ("visual", key, "norm", "scale"), state_dict[f"{prefix}.norm.weight"], summary)
        _set_param(mutable, ("visual", key, "norm", "bias"), state_dict[f"{prefix}.norm.bias"], summary)
        _set_param(mutable, ("visual", key, "linear_fc1", "kernel"), _as_np_tensor(state_dict[f"{prefix}.linear_fc1.weight"]).T, summary)
        _set_param(mutable, ("visual", key, "linear_fc1", "bias"), state_dict[f"{prefix}.linear_fc1.bias"], summary)
        _set_param(mutable, ("visual", key, "linear_fc2", "kernel"), _as_np_tensor(state_dict[f"{prefix}.linear_fc2.weight"]).T, summary)
        _set_param(mutable, ("visual", key, "linear_fc2", "bias"), state_dict[f"{prefix}.linear_fc2.bias"], summary)

    _set_param(mutable, ("visual", "merger", "norm", "scale"), state_dict["model.visual.merger.norm.weight"], summary)
    _set_param(mutable, ("visual", "merger", "norm", "bias"), state_dict["model.visual.merger.norm.bias"], summary)
    _set_param(mutable, ("visual", "merger", "linear_fc1", "kernel"), _as_np_tensor(state_dict["model.visual.merger.linear_fc1.weight"]).T, summary)
    _set_param(mutable, ("visual", "merger", "linear_fc1", "bias"), state_dict["model.visual.merger.linear_fc1.bias"], summary)
    _set_param(mutable, ("visual", "merger", "linear_fc2", "kernel"), _as_np_tensor(state_dict["model.visual.merger.linear_fc2.weight"]).T, summary)
    _set_param(mutable, ("visual", "merger", "linear_fc2", "bias"), state_dict["model.visual.merger.linear_fc2.bias"], summary)

    _set_param(mutable, ("token_embed", "embedding"), state_dict["model.language_model.embed_tokens.weight"], summary)
    text_hidden = mutable["token_embed"]["embedding"].shape[1]
    q_kernel_shape = mutable["decoder_0"]["self_attn"]["q_proj"]["kernel"].shape
    k_kernel_shape = mutable["decoder_0"]["self_attn"]["k_proj"]["kernel"].shape
    if len(q_kernel_shape) == 3:
        text_heads = q_kernel_shape[1]
        text_head_dim = q_kernel_shape[2]
    else:
        text_heads = state_dict["model.language_model.layers.0.self_attn.q_proj.weight"].shape[0] // text_hidden
        text_head_dim = q_kernel_shape[1] // text_heads
    if len(k_kernel_shape) == 3:
        kv_heads = k_kernel_shape[1]
    else:
        kv_heads = state_dict["model.language_model.layers.0.self_attn.k_proj.weight"].shape[0] // text_head_dim

    for idx in range(len([k for k in mutable if k.startswith("decoder_")])):
        prefix = f"model.language_model.layers.{idx}"
        layer = f"decoder_{idx}"
        _set_param(mutable, (layer, "input_layernorm", "scale"), state_dict[f"{prefix}.input_layernorm.weight"], summary)
        _set_param(mutable, (layer, "post_attention_layernorm", "scale"), state_dict[f"{prefix}.post_attention_layernorm.weight"], summary)
        _set_param(
            mutable,
            (layer, "self_attn", "q_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.q_proj.weight"],
                    text_hidden,
                    text_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["q_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.q_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "k_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.k_proj.weight"],
                    text_hidden,
                    kv_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["k_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.k_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "v_proj", "kernel"),
            (
                _reshape_dense_general_in(
                    state_dict[f"{prefix}.self_attn.v_proj.weight"],
                    text_hidden,
                    kv_heads,
                    text_head_dim,
                )
                if mutable[layer]["self_attn"]["v_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.v_proj.weight"]).T
            ),
            summary,
        )
        _set_param(
            mutable,
            (layer, "self_attn", "o_proj", "kernel"),
            (
                _reshape_dense_general_out(
                    state_dict[f"{prefix}.self_attn.o_proj.weight"],
                    text_heads,
                    text_head_dim,
                    text_hidden,
                )
                if mutable[layer]["self_attn"]["o_proj"]["kernel"].ndim == 3
                else _as_np_tensor(state_dict[f"{prefix}.self_attn.o_proj.weight"]).T
            ),
            summary,
        )
        _set_param(mutable, (layer, "self_attn", "q_norm", "scale"), state_dict[f"{prefix}.self_attn.q_norm.weight"], summary)
        _set_param(mutable, (layer, "self_attn", "k_norm", "scale"), state_dict[f"{prefix}.self_attn.k_norm.weight"], summary)
        _set_param(mutable, (layer, "mlp", "gate_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.gate_proj.weight"]).T, summary)
        _set_param(mutable, (layer, "mlp", "up_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.up_proj.weight"]).T, summary)
        _set_param(mutable, (layer, "mlp", "down_proj", "kernel"), _as_np_tensor(state_dict[f"{prefix}.mlp.down_proj.weight"]).T, summary)

    _set_param(mutable, ("final_norm", "scale"), state_dict["model.language_model.norm.weight"], summary)

    return core.freeze(mutable), WeightLoadSummary(
        loaded=int(summary["loaded"]),
        skipped=int(summary["skipped"]),
        mismatched=int(summary["mismatched"]),
        skipped_keys=list(summary["skipped_keys"]),
        mismatched_keys=list(summary["mismatched_keys"]),
    )
