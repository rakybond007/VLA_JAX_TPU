from __future__ import annotations

from dataclasses import dataclass

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
    prompt: str = "Describe this image briefly."
    select_layer: int = -1
    project_to_dim: int = 1536


def main():
    args = tyro.cli(Args)

    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
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
    batch = processor.build_prefixed_inputs(messages)
    backbone = QwenXLABackbone(
        QwenXLAConfig(
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            input_prefix=args.input_prefix,
            select_layer=args.select_layer,
            project_to_dim=args.project_to_dim,
        )
    )
    outputs = backbone.extract_backbone_features(batch)

    print("device:", next(backbone.model.parameters()).device)
    print("dtype:", next(backbone.model.parameters()).dtype)
    print("backbone_features:", tuple(outputs["backbone_features"].shape))
    if outputs.get("backbone_attention_mask") is not None:
        print("attention_mask:", tuple(outputs["backbone_attention_mask"].shape))
    if outputs.get("image_mask") is not None:
        print("image_mask:", tuple(outputs["image_mask"].shape))
    print("projected_dim:", backbone.output_dim)
    print("torch tensor device:", torch.tensor(0, device=next(backbone.model.parameters()).device).device)


if __name__ == "__main__":
    main()
