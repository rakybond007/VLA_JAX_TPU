from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

from PIL import Image
import torch
import tyro

from vla_tpu.models.qwen_processor import QwenProcessorAdapter, QwenProcessorConfig
from vla_tpu.models.qwen_xla_reference import QwenXLAConfig, QwenXLABackbone


@dataclass
class Args:
    backbone_type: str = "qwen3_vl"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    input_prefix: str = "qwen_3_vl_"
    prompt: str = "Describe these images briefly."
    image_counts: str = "1,2,3"
    batch_sizes: str = "1"
    image_size: int = 224
    select_layer: int = -1
    project_to_dim: int = 1536
    warmup_steps: int = 10
    measure_steps: int = 10
    fixed_image_slots: int | None = None


def _build_messages(
    num_images: int,
    image_size: int,
    prompt: str,
    fixed_image_slots: int | None = None,
):
    colors = [
        (128, 128, 128),
        (64, 128, 192),
        (192, 96, 64),
    ]
    content = []
    total_slots = fixed_image_slots if fixed_image_slots is not None else num_images
    for idx in range(total_slots):
        if idx < num_images:
            color = colors[idx % len(colors)]
        else:
            # Keep a static image-slot count while marking unused views with a blank frame.
            color = (0, 0, 0)
        image = Image.new("RGB", (image_size, image_size), color=color)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _parse_counts(spec: str):
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _repeat_batch(batch: dict[str, torch.Tensor], batch_size: int):
    if batch_size == 1:
        return batch

    repeated = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            repeated[key] = value
            continue

        suffix = key.split("_")[-1]
        if suffix in {"ids", "mask"} and value.ndim >= 2 and value.shape[0] == 1:
            repeated[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
            continue

        if key.endswith("pixel_values") or key.endswith("pixel_values_videos"):
            repeated[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
            continue

        if key.endswith("image_grid_thw") or key.endswith("video_grid_thw"):
            repeated[key] = value.repeat(batch_size, 1)
            continue

        if value.ndim >= 1 and value.shape[0] == 1:
            repeated[key] = value.repeat(batch_size, *([1] * (value.ndim - 1)))
            continue

        repeated[key] = value

    return repeated


def main():
    args = tyro.cli(Args)
    counts = _parse_counts(args.image_counts)
    batch_sizes = _parse_counts(args.batch_sizes)

    processor = QwenProcessorAdapter(
        QwenProcessorConfig(
            model_name=args.model_name,
            input_prefix=args.input_prefix,
        )
    )
    backbone = QwenXLABackbone(
        QwenXLAConfig(
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            input_prefix=args.input_prefix,
            select_layer=args.select_layer,
            project_to_dim=args.project_to_dim,
        )
    )

    print("device:", next(backbone.model.parameters()).device)
    print("dtype:", next(backbone.model.parameters()).dtype)
    print("model:", args.model_name)
    print("backbone_type:", args.backbone_type)
    print("image_size:", args.image_size)
    print("batch_sizes:", batch_sizes)
    print("warmup_steps:", args.warmup_steps)
    print("measure_steps:", args.measure_steps)
    print("fixed_image_slots:", args.fixed_image_slots)

    for num_images in counts:
        messages = _build_messages(
            num_images,
            args.image_size,
            args.prompt,
            fixed_image_slots=args.fixed_image_slots,
        )
        base_batch = processor.build_prefixed_inputs(messages)

        print(f"images={num_images}")
        for batch_size in batch_sizes:
            print(f"  batch_size={batch_size}")
            batch = _repeat_batch(base_batch, batch_size)

            try:
                start = time.perf_counter()
                outputs = backbone.extract_backbone_features(batch)
                first_latency = time.perf_counter() - start

                for _ in range(args.warmup_steps):
                    outputs = backbone.extract_backbone_features(batch)

                latencies = []
                for _ in range(args.measure_steps):
                    start = time.perf_counter()
                    outputs = backbone.extract_backbone_features(batch)
                    latencies.append(time.perf_counter() - start)
            except RuntimeError as exc:
                print(f"    status: failed")
                print(f"    error_type: {type(exc).__name__}")
                print(f"    error: {exc}")
                break

            print("    backbone_features:", tuple(outputs["backbone_features"].shape))
            if outputs.get("backbone_attention_mask") is not None:
                print("    attention_mask:", tuple(outputs["backbone_attention_mask"].shape))
            print(f"    first_call_s: {first_latency:.6f}")
            print(f"    mean_repeat_s: {statistics.mean(latencies):.6f}")
            print(f"    min_repeat_s: {min(latencies):.6f}")
            print(f"    max_repeat_s: {max(latencies):.6f}")
            print(f"    hz_estimate: {1.0 / statistics.mean(latencies):.3f}")

    print("torch tensor device:", torch.tensor(0, device=next(backbone.model.parameters()).device).device)


if __name__ == "__main__":
    main()
