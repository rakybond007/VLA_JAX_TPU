from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image
import tyro
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from vla_tpu.models.qwen_processor import QwenProcessorAdapter, QwenProcessorConfig
from vla_tpu.models.qwen3_vl_jax import (
    ExactVisionModel,
    ExactVisionPatchEmbed,
    JAXQwen3VLPureBackbone,
    ExactVisionRotaryEmbedding,
    _build_vision_rotary_embeddings,
    _build_vision_rotary_raw_embeddings,
)
from vla_tpu.models.qwen3_vl_weight_loader import (
    backbone_config_from_hf_qwen3_vl,
    load_hf_qwen3_vl_state_dict,
    load_hf_weights_into_jax_qwen3_vl_pure_backbone,
)


@dataclass
class Args:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    prompt: str = "Describe this image briefly."
    image_size: int = 224
    color_value: int = 128
    input_prefix: str = "qwen_3_vl_"
    output_path: str = "outputs/qwen3_vl_vision_stages_cpu.json"


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, float | list[int]]:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    cosine = float(np.dot(flat_a, flat_b) / denom) if denom else 0.0
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "mse": float(np.mean((a - b) ** 2)),
        "cosine": cosine,
        "mean_delta": float(abs(a.mean() - b.mean())),
        "std_delta": float(abs(a.std() - b.std())),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def _write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = tyro.cli(Args)
    image = Image.new("RGB", (args.image_size, args.image_size), color=(args.color_value,) * 3)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    processor = QwenProcessorAdapter(
        QwenProcessorConfig(
            model_name=args.model_name,
            input_prefix=args.input_prefix,
        )
    )
    prefixed = processor.build_prefixed_inputs(messages)
    pixel_values = prefixed[f"{args.input_prefix}pixel_values"]
    image_grid_thw = prefixed[f"{args.input_prefix}image_grid_thw"]
    input_ids = prefixed[f"{args.input_prefix}input_ids"]
    attention_mask = prefixed[f"{args.input_prefix}attention_mask"]
    mm_token_type_ids = prefixed[f"{args.input_prefix}mm_token_type_ids"]
    hf_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    cfg = backbone_config_from_hf_qwen3_vl(args.model_name, n_visual_tokens=192, max_text_tokens=int(input_ids.shape[1]))
    backbone = JAXQwen3VLPureBackbone(cfg)
    vision_token_count = int(np.prod(image_grid_thw.detach().cpu().numpy()[:, 1:], axis=1).sum() // (cfg.vision_spatial_merge_size**2))
    dummy_instruction_tokens = np.full((1, vision_token_count), hf_cfg.image_token_id, dtype=np.int32)
    init_batch = {
        "images": jnp.asarray(np.asarray(image, dtype=np.uint8)[None, None, ...]),
        "pixel_values": jnp.asarray(pixel_values.detach().cpu().numpy().astype(np.float32)),
        "instruction_tokens": jnp.asarray(dummy_instruction_tokens),
        "attention_mask": jnp.ones((1, vision_token_count), dtype=jnp.int32),
        "image_token_mask": jnp.ones((1, vision_token_count), dtype=jnp.int32),
        "image_token_id": jnp.asarray(np.array(hf_cfg.image_token_id, dtype=np.int32)),
        "mm_token_type_ids": jnp.ones((1, vision_token_count), dtype=jnp.int32),
        "image_grid_thw": jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
    }
    init_vars = backbone.init(jax.random.PRNGKey(0), init_batch, return_intermediates=True)
    state_dict = load_hf_qwen3_vl_state_dict(args.model_name)
    params, summary = load_hf_weights_into_jax_qwen3_vl_pure_backbone(init_vars["params"], state_dict)

    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        _attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    with torch.no_grad():
        hf_patch = hf_model.model.visual.patch_embed(pixel_values.float()).detach().cpu().numpy().astype(np.float32)
        hf_pos = hf_model.model.visual.fast_pos_embed_interpolate(image_grid_thw).detach().cpu().numpy().astype(np.float32)
        hf_patch_pos = hf_patch + hf_pos
        hf_rotary_raw = hf_model.model.visual.rot_pos_emb(image_grid_thw).detach().cpu().numpy().astype(np.float32)
        hf_rotary_emb = np.concatenate([hf_rotary_raw, hf_rotary_raw], axis=-1)
        hf_rotary_cos = np.cos(hf_rotary_emb).astype(np.float32)
        hf_rotary_sin = np.sin(hf_rotary_emb).astype(np.float32)
        hf_block_hidden_states_t: list[torch.Tensor] = []
        hooks = [
            block.register_forward_hook(
                lambda _module, _inputs, output, bucket=hf_block_hidden_states_t: bucket.append(output.detach())
            )
            for block in hf_model.model.visual.blocks
        ]
        hf_visual_outputs = hf_model.model.visual(
            pixel_values.float(),
            grid_thw=image_grid_thw,
            return_dict=False,
        )
        for hook in hooks:
            hook.remove()
        hf_last_hidden_t = hf_visual_outputs[0]
        hf_pooler_t = hf_visual_outputs[1]
        hf_last_hidden = hf_last_hidden_t.detach().cpu().numpy().astype(np.float32)
        hf_pooler = hf_pooler_t.detach().cpu().numpy().astype(np.float32)[None, ...]
        hf_block_hidden_states = [t.cpu().numpy().astype(np.float32) for t in hf_block_hidden_states_t]

    jax_patch = ExactVisionPatchEmbed(cfg).apply(
        {"params": params["visual"]["patch_embed"]},
        jnp.asarray(pixel_values.detach().cpu().numpy().astype(np.float32)),
    )
    head_dim = cfg.vision_hidden_size // cfg.vision_num_heads
    rotary_table = ExactVisionRotaryEmbedding(
        dim=head_dim // 2,
        theta=cfg.rope_max_wavelength,
    ).apply({}, int(jnp.max(jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32))[:, 1:])))
    jax_rotary_raw = np.asarray(
        jax.device_get(
            _build_vision_rotary_raw_embeddings(
                jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
                cfg.vision_spatial_merge_size,
                rotary_table,
            )
        )
    ).astype(np.float32)
    jax_rotary_cos, jax_rotary_sin = _build_vision_rotary_embeddings(
        jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
        cfg.vision_spatial_merge_size,
        rotary_table,
    )
    jax_visual = ExactVisionModel(cfg).apply(
        {"params": params["visual"]},
        jnp.asarray(pixel_values.detach().cpu().numpy().astype(np.float32)),
        jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
        return_intermediates=True,
    )

    jax_patch = np.asarray(jax.device_get(jax_patch)).astype(np.float32)
    jax_pos = np.asarray(jax.device_get(jax_visual["pos_embeds"])).astype(np.float32)
    jax_patch_pos = jax_patch + jax_pos
    jax_patch_pos_model = np.asarray(jax.device_get(jax_visual["patch_pos_tokens"])).astype(np.float32)
    jax_last_hidden = np.asarray(jax.device_get(jax_visual["last_hidden_state"])).astype(np.float32)
    jax_pooler_np = np.asarray(jax.device_get(jax_visual["pooler_output"])).astype(np.float32)
    jax_rotary_cos = np.asarray(jax.device_get(jax_rotary_cos)).astype(np.float32)
    jax_rotary_sin = np.asarray(jax.device_get(jax_rotary_sin)).astype(np.float32)
    jax_block_hidden_states = [
        np.asarray(jax.device_get(t)).astype(np.float32) for t in jax_visual["block_hidden_states"]
    ]

    payload = {
        "weight_load_summary": {
            "loaded": summary.loaded,
            "mismatched": summary.mismatched,
        },
        "patch_compare": _compare(hf_patch, jax_patch),
        "pos_compare": _compare(hf_pos, jax_pos),
        "patch_pos_compare": _compare(hf_patch_pos, jax_patch_pos),
        "patch_pos_model_compare": _compare(hf_patch_pos, jax_patch_pos_model),
        "vision_rotary_raw_compare": _compare(hf_rotary_raw, jax_rotary_raw),
        "vision_rotary_cos_compare": _compare(hf_rotary_cos, jax_rotary_cos),
        "vision_rotary_sin_compare": _compare(hf_rotary_sin, jax_rotary_sin),
        "vision_last_hidden_compare": _compare(hf_last_hidden, jax_last_hidden),
        "vision_pooler_compare": _compare(hf_pooler, jax_pooler_np),
        "vision_block_compares": [
            {"block_index": idx, **_compare(hf_block_hidden_states[idx], jax_block_hidden_states[idx])}
            for idx in range(min(len(hf_block_hidden_states), len(jax_block_hidden_states)))
        ],
    }
    _write_json(args.output_path, payload)


if __name__ == "__main__":
    main()
