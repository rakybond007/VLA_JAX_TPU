from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tyro

from vla_tpu.models.qwen_processor import QwenProcessorAdapter, QwenProcessorConfig
from vla_tpu.models.qwen_xla_reference import QwenXLAConfig, QwenXLABackbone
from vla_tpu.models.qwen3_vl_jax import JAXQwen3VLPureBackbone
from vla_tpu.models.qwen3_vl_weight_loader import (
    backbone_config_from_hf_qwen3_vl,
    load_hf_qwen3_vl_state_dict,
    load_hf_weights_into_jax_qwen3_vl_pure_backbone,
)

try:
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - optional dependency path
    AutoConfig = None


@dataclass
class Args:
    mode: str = "reference"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    prompt: str = "Describe this image briefly."
    image_size: int = 224
    color_value: int = 128
    n_visual_tokens: int = 192
    max_text_tokens: int = 512
    select_layer: int = -1
    input_prefix: str = "qwen_3_vl_"
    reference_path: str = "outputs/qwen3_vl_reference_stats.json"
    jax_path: str = "outputs/qwen3_vl_jax_stats.json"
    output_path: str = "outputs/qwen3_vl_weight_loading_compare.json"
    reference_hidden_path: str = "outputs/qwen3_vl_reference_hidden.npy"
    jax_hidden_path: str = "outputs/qwen3_vl_jax_hidden.npy"


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


def _masked_stats(array: np.ndarray, mask: np.ndarray | None) -> dict[str, float | list[int]] | None:
    if mask is None:
        return None
    flat = array[mask]
    if flat.size == 0:
        return None
    return _stats(flat)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(np.square(diff)))


def _write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


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


def run_reference(args: Args) -> None:
    image, messages = _build_messages(args.image_size, args.color_value, args.prompt)
    processor = QwenProcessorAdapter(
        QwenProcessorConfig(
            model_name=args.model_name,
            input_prefix=args.input_prefix,
        )
    )
    prefixed = processor.build_prefixed_inputs(messages)
    reference = QwenXLABackbone(
        QwenXLAConfig(
            backbone_type="qwen3_vl",
            model_name=args.model_name,
            input_prefix=args.input_prefix,
            select_layer=args.select_layer,
            project_to_dim=None,
        )
    )
    ref_outputs = reference.extract_backbone_features(prefixed)
    ref_features = ref_outputs["backbone_features"].detach().float().cpu().numpy()
    ref_image_mask = None
    if ref_outputs.get("image_mask") is not None:
        ref_image_mask = ref_outputs["image_mask"].detach().cpu().numpy().astype(bool)
    np.save(args.reference_hidden_path, ref_features.astype(np.float32))

    payload = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "image_size": args.image_size,
        "color_value": args.color_value,
        "instruction_tokens_shape": list(prefixed[f"{args.input_prefix}input_ids"].shape),
        "reference_last_hidden": _stats(ref_features),
        "reference_image_positions": _masked_stats(ref_features, ref_image_mask),
    }
    _write_json(args.reference_path, payload)


def run_jax(args: Args) -> None:
    image, messages = _build_messages(args.image_size, args.color_value, args.prompt)
    processor = QwenProcessorAdapter(
        QwenProcessorConfig(
            model_name=args.model_name,
            input_prefix=args.input_prefix,
        )
    )
    prefixed = processor.build_prefixed_inputs(messages)
    instruction_tokens = prefixed[f"{args.input_prefix}input_ids"].detach().cpu().numpy().astype(np.int32)
    attention_mask = prefixed[f"{args.input_prefix}attention_mask"].detach().cpu().numpy().astype(np.int32)
    mm_token_type_ids = prefixed[f"{args.input_prefix}mm_token_type_ids"].detach().cpu().numpy().astype(np.int32)
    image_grid_thw = prefixed[f"{args.input_prefix}image_grid_thw"].detach().cpu().numpy().astype(np.int32)
    pixel_values = prefixed[f"{args.input_prefix}pixel_values"].detach().cpu().numpy().astype(np.float32)
    if AutoConfig is None:
        raise ImportError("transformers is required to fetch Qwen3-VL config.")
    hf_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    image_token_mask = (instruction_tokens == hf_cfg.image_token_id).astype(np.int32)
    image_np = np.asarray(image, dtype=np.uint8)[None, None, ...]

    cfg = backbone_config_from_hf_qwen3_vl(
        args.model_name,
        n_visual_tokens=args.n_visual_tokens,
        max_text_tokens=max(args.max_text_tokens, instruction_tokens.shape[1]),
    )
    model = JAXQwen3VLPureBackbone(cfg)
    batch = {
        "images": jnp.asarray(image_np),
        "pixel_values": jnp.asarray(pixel_values),
        "instruction_tokens": jnp.asarray(instruction_tokens),
        "attention_mask": jnp.asarray(attention_mask),
        "image_token_mask": jnp.asarray(image_token_mask),
        "image_token_id": jnp.asarray(np.array(hf_cfg.image_token_id, dtype=np.int32)),
        "mm_token_type_ids": jnp.asarray(mm_token_type_ids),
        "image_grid_thw": jnp.asarray(image_grid_thw),
    }
    init_vars = model.init(jax.random.PRNGKey(0), batch, return_intermediates=True)
    state_dict = load_hf_qwen3_vl_state_dict(args.model_name)
    loaded_params, summary = load_hf_weights_into_jax_qwen3_vl_pure_backbone(init_vars["params"], state_dict)
    outputs = model.apply({"params": loaded_params}, batch, return_intermediates=True)

    jax_hidden = np.asarray(jax.device_get(outputs["hidden_states"]))
    jax_visual = np.asarray(jax.device_get(outputs["visual_tokens"]))
    np.save(args.jax_hidden_path, jax_hidden.astype(np.float32))
    payload = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "instruction_tokens_shape": list(instruction_tokens.shape),
        "weight_load_summary": asdict(summary),
        "jax_hidden_states": _stats(jax_hidden),
        "jax_visual_tokens": _stats(jax_visual),
    }
    _write_json(args.jax_path, payload)


def run_compare(args: Args) -> None:
    reference_payload = json.loads(Path(args.reference_path).read_text())
    jax_payload = json.loads(Path(args.jax_path).read_text())
    reference_hidden = np.load(args.reference_hidden_path)
    jax_hidden = np.load(args.jax_hidden_path)

    seq_min = min(reference_hidden.shape[1], jax_hidden.shape[1])
    dim_min = min(reference_hidden.shape[2], jax_hidden.shape[2])
    reference_aligned = reference_hidden[:, :seq_min, :dim_min]
    jax_aligned = jax_hidden[:, :seq_min, :dim_min]

    payload = {
        "model_name": args.model_name,
        "prompt": args.prompt,
        "reference_path": args.reference_path,
        "jax_path": args.jax_path,
        "weight_load_summary": jax_payload["weight_load_summary"],
        "reference_last_hidden": reference_payload["reference_last_hidden"],
        "reference_image_positions": reference_payload["reference_image_positions"],
        "jax_hidden_states": jax_payload["jax_hidden_states"],
        "jax_visual_tokens": jax_payload["jax_visual_tokens"],
        "comparison": {
            "aligned_shape": [int(reference_aligned.shape[0]), int(reference_aligned.shape[1]), int(reference_aligned.shape[2])],
            "hidden_mean_delta": float(
                abs(
                    reference_payload["reference_last_hidden"]["mean"]
                    - jax_payload["jax_hidden_states"]["mean"]
                )
            ),
            "hidden_std_delta": float(
                abs(
                    reference_payload["reference_last_hidden"]["std"]
                    - jax_payload["jax_hidden_states"]["std"]
                )
            ),
            "hidden_mse": _mse(reference_aligned, jax_aligned),
            "hidden_cosine": _cosine(reference_aligned, jax_aligned),
            "mean_stat_cosine": _cosine(
                np.array([reference_payload["reference_last_hidden"]["mean"]], dtype=np.float32),
                np.array([jax_payload["jax_hidden_states"]["mean"]], dtype=np.float32),
            ),
        },
    }
    _write_json(args.output_path, payload)


def main() -> None:
    args = tyro.cli(Args)
    if args.mode == "reference":
        run_reference(args)
        return
    if args.mode == "jax":
        run_jax(args)
        return
    if args.mode == "compare":
        run_compare(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
