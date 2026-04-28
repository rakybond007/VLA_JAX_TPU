from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tyro

from vla_tpu.models.qwen_processor import QwenProcessorAdapter, QwenProcessorConfig
from vla_tpu.models.qwen3_vl_jax import (
    ExactVisionModel,
    JAXQwen3VLPureBackbone,
    RMSNorm,
    TextDecoderLayer,
    _deepstack_process,
    _scatter_visual_tokens,
    build_text_position_embeddings,
    get_rope_index,
)
from vla_tpu.models.qwen3_vl_weight_loader import (
    backbone_config_from_hf_qwen3_vl,
    load_hf_qwen3_vl_state_dict,
    load_hf_weights_into_jax_qwen3_vl_pure_backbone,
)

try:
    import torch
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from transformers.masking_utils import create_causal_mask
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    AutoConfig = None
    Qwen3VLForConditionalGeneration = None
    create_causal_mask = None


@dataclass
class Args:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    prompt: str = "Describe this image briefly."
    image_size: int = 224
    color_value: int = 128
    n_visual_tokens: int = 192
    max_text_tokens: int = 512
    input_prefix: str = "qwen_3_vl_"
    output_path: str = "outputs/qwen3_vl_decoder_parity_cpu.json"


def _stats(array: np.ndarray) -> dict[str, float | list[int]]:
    arr = array.astype(np.float32)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "l2_norm": float(np.linalg.norm(arr.reshape(-1))),
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(np.square(diff)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, float | list[int]]:
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "mse": _mse(a, b),
        "cosine": _cosine(a, b),
        "mean_delta": float(abs(a.astype(np.float32).mean() - b.astype(np.float32).mean())),
        "std_delta": float(abs(a.astype(np.float32).std() - b.astype(np.float32).std())),
        "max_abs": float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))),
    }


def _build_messages(image_size: int, color_value: int, prompt: str) -> tuple[Image.Image, list[dict]]:
    image = Image.new("RGB", (image_size, image_size), color=(color_value,) * 3)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return image, messages


def _write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _explicit_causal_bias(attention_mask: np.ndarray, dtype: np.dtype) -> np.ndarray:
    token_valid = attention_mask.astype(bool)
    seq_len = token_valid.shape[1]
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    keep = token_valid[:, None, :] & token_valid[:, :, None] & causal[None, :, :]
    return np.where(
        keep[:, None, :, :],
        np.array(0.0, dtype=dtype),
        np.array(-1e30, dtype=dtype),
    ).astype(dtype)


def _run_jax_decoder(
    params: dict,
    cfg,
    inputs_embeds: np.ndarray,
    attention_bias: np.ndarray,
    position_embeddings: tuple[np.ndarray, np.ndarray],
    deepstack_visual_embeds: list[np.ndarray] | None,
    visual_pos_masks: np.ndarray | None,
) -> np.ndarray:
    hidden = jnp.asarray(inputs_embeds)
    attn = jnp.asarray(attention_bias)
    pos_emb = (jnp.asarray(position_embeddings[0]), jnp.asarray(position_embeddings[1]))

    for layer_idx in range(cfg.num_layers):
        hidden = TextDecoderLayer(cfg).apply(
            {"params": params[f"decoder_{layer_idx}"]},
            hidden,
            attention_mask=attn,
            position_embeddings=pos_emb,
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden = _deepstack_process(
                hidden,
                jnp.asarray(visual_pos_masks),
                jnp.asarray(deepstack_visual_embeds[layer_idx]),
            )

    hidden = RMSNorm(cfg.layer_norm_epsilon).apply({"params": params["final_norm"]}, hidden)
    return np.asarray(jax.device_get(hidden))


def main() -> None:
    args = tyro.cli(Args)
    if (
        Qwen3VLForConditionalGeneration is None
        or AutoConfig is None
        or create_causal_mask is None
        or torch is None
    ):
        raise ImportError("transformers + torch are required for decoder parity comparison.")

    image, messages = _build_messages(args.image_size, args.color_value, args.prompt)
    processor = QwenProcessorAdapter(
        QwenProcessorConfig(
            model_name=args.model_name,
            input_prefix=args.input_prefix,
        )
    )
    prefixed = processor.build_prefixed_inputs(messages)
    input_ids = prefixed[f"{args.input_prefix}input_ids"]
    attention_mask = prefixed[f"{args.input_prefix}attention_mask"]
    mm_token_type_ids = prefixed[f"{args.input_prefix}mm_token_type_ids"]
    image_grid_thw = prefixed[f"{args.input_prefix}image_grid_thw"]
    pixel_values = prefixed[f"{args.input_prefix}pixel_values"]

    hf_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    cfg = backbone_config_from_hf_qwen3_vl(
        args.model_name,
        n_visual_tokens=args.n_visual_tokens,
        max_text_tokens=max(args.max_text_tokens, int(input_ids.shape[1])),
    )

    model = JAXQwen3VLPureBackbone(cfg)
    init_batch = {
        "images": jnp.asarray(np.asarray(image, dtype=np.uint8)[None, None, ...]),
        "pixel_values": jnp.asarray(pixel_values.detach().cpu().numpy().astype(np.float32)),
        "instruction_tokens": jnp.asarray(input_ids.detach().cpu().numpy().astype(np.int32)),
        "attention_mask": jnp.asarray(attention_mask.detach().cpu().numpy().astype(np.int32)),
        "image_token_mask": jnp.asarray((input_ids == hf_cfg.image_token_id).detach().cpu().numpy().astype(np.int32)),
        "image_token_id": jnp.asarray(np.array(hf_cfg.image_token_id, dtype=np.int32)),
        "mm_token_type_ids": jnp.asarray(mm_token_type_ids.detach().cpu().numpy().astype(np.int32)),
        "image_grid_thw": jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
    }
    init_vars = model.init(jax.random.PRNGKey(0), init_batch, return_intermediates=True)
    state_dict = load_hf_qwen3_vl_state_dict(args.model_name)
    loaded_params, summary = load_hf_weights_into_jax_qwen3_vl_pure_backbone(init_vars["params"], state_dict)

    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        _attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    with torch.no_grad():
        inputs_embeds = hf_model.model.get_input_embeddings()(input_ids)
        image_outputs = hf_model.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = hf_model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = [embed.detach().cpu().numpy() for embed in image_outputs.deepstack_features]

        position_ids = hf_model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        text_position_ids = position_ids[0]
        hf_attention_bias = create_causal_mask(
            config=hf_model.model.language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        hf_cos, hf_sin = hf_model.model.language_model.rotary_emb(inputs_embeds, position_ids)

        hf_no_ds = hf_model.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        ).last_hidden_state
        hf_with_ds = hf_model.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=[torch.from_numpy(x).to(inputs_embeds.device, inputs_embeds.dtype) for x in deepstack_visual_embeds],
        ).last_hidden_state

    jax_visual_tokens, jax_deepstack = ExactVisionModel(cfg).apply(
        {"params": loaded_params["visual"]},
        jnp.asarray(pixel_values.detach().cpu().numpy().astype(np.float32)),
        jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
    )
    instruction_tokens_np = input_ids.detach().cpu().numpy().astype(np.int32)
    text_embed_table = np.asarray(jax.device_get(loaded_params["token_embed"]["embedding"]))
    jax_text_embeds = text_embed_table[instruction_tokens_np]
    image_token_mask_np = (instruction_tokens_np == hf_cfg.image_token_id)
    jax_inputs_embeds = _scatter_visual_tokens(
        jnp.asarray(jax_text_embeds),
        jnp.asarray(image_token_mask_np),
        jax_visual_tokens,
    )
    jax_position_ids = get_rope_index(
        jnp.asarray(instruction_tokens_np),
        jnp.asarray(mm_token_type_ids.detach().cpu().numpy().astype(np.int32)),
        jnp.asarray(image_grid_thw.detach().cpu().numpy().astype(np.int32)),
        jnp.asarray(attention_mask.detach().cpu().numpy().astype(np.int32)),
        cfg.vision_spatial_merge_size,
    )
    jax_cos, jax_sin = build_text_position_embeddings(
        cfg.hidden_size // cfg.num_heads,
        jax_position_ids,
        cfg.rope_max_wavelength,
        cfg.mrope_section,
    )

    hf_inputs_embeds_np = inputs_embeds.detach().cpu().numpy().astype(np.float32)
    if hf_attention_bias is None:
        hf_attention_bias_np = _explicit_causal_bias(
            attention_mask.detach().cpu().numpy().astype(np.int32),
            np.float32,
        )
        hf_attention_bias_kind = "implicit_sdpa_causal"
    else:
        hf_attention_bias_np = hf_attention_bias.detach().cpu().numpy().astype(np.float32)
        hf_attention_bias_kind = "explicit_bias"
    hf_cos_np = hf_cos.detach().cpu().numpy().astype(np.float32)
    hf_sin_np = hf_sin.detach().cpu().numpy().astype(np.float32)
    jax_inputs_embeds_np = np.asarray(jax.device_get(jax_inputs_embeds)).astype(np.float32)
    jax_attention_bias_np = hf_attention_bias_np
    jax_cos_np = np.asarray(jax.device_get(jax_cos)).astype(np.float32)
    jax_sin_np = np.asarray(jax.device_get(jax_sin)).astype(np.float32)

    jax_no_ds = _run_jax_decoder(
        loaded_params,
        cfg,
        inputs_embeds=jax_inputs_embeds_np,
        attention_bias=jax_attention_bias_np,
        position_embeddings=(jax_cos_np, jax_sin_np),
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    )
    jax_with_ds = _run_jax_decoder(
        loaded_params,
        cfg,
        inputs_embeds=jax_inputs_embeds_np,
        attention_bias=jax_attention_bias_np,
        position_embeddings=(jax_cos_np, jax_sin_np),
        deepstack_visual_embeds=[np.asarray(jax.device_get(x)).astype(np.float32) for x in jax_deepstack],
        visual_pos_masks=visual_pos_masks.detach().cpu().numpy().astype(bool),
    )

    hf_no_ds_np = hf_no_ds.detach().cpu().numpy().astype(np.float32)
    hf_with_ds_np = hf_with_ds.detach().cpu().numpy().astype(np.float32)

    payload = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "weight_load_summary": {
            "loaded": summary.loaded,
            "skipped": summary.skipped,
            "mismatched": summary.mismatched,
        },
        "hf_attention_bias_kind": hf_attention_bias_kind,
        "inputs_embeds_compare": _compare(hf_inputs_embeds_np, jax_inputs_embeds_np),
        "attention_bias_compare": _compare(hf_attention_bias_np, jax_attention_bias_np),
        "rotary_cos_compare": _compare(hf_cos_np, jax_cos_np),
        "rotary_sin_compare": _compare(hf_sin_np, jax_sin_np),
        "vision_pooler_compare": _compare(
            np.asarray(torch.cat(image_outputs.pooler_output, dim=0).detach().cpu().numpy(), dtype=np.float32)[None, ...],
            np.asarray(jax.device_get(jax_visual_tokens)).astype(np.float32),
        ),
        "deepstack_visual_compare": [
            _compare(
                np.asarray(hf_embed, dtype=np.float32)[None, ...],
                np.asarray(jax.device_get(jax_embed)).astype(np.float32),
            )
            for hf_embed, jax_embed in zip(deepstack_visual_embeds, jax_deepstack)
        ],
        "decoder_no_deepstack_compare": _compare(hf_no_ds_np, jax_no_ds),
        "decoder_with_deepstack_compare": _compare(hf_with_ds_np, jax_with_ds),
        "deepstack_delta_compare": _compare(hf_with_ds_np - hf_no_ds_np, jax_with_ds - jax_no_ds),
        "hf_no_deepstack_stats": _stats(hf_no_ds_np),
        "jax_no_deepstack_stats": _stats(jax_no_ds),
        "hf_with_deepstack_stats": _stats(hf_with_ds_np),
        "jax_with_deepstack_stats": _stats(jax_with_ds),
    }
    _write_json(args.output_path, payload)


if __name__ == "__main__":
    main()
