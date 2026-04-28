from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tyro

from vla_tpu.models.qwen_reference import QwenReferenceBackbone, QwenReferenceConfig


@dataclass
class Args:
    backbone_type: str = "qwen3_vl"
    model_name: str = "Qwen/Qwen3-VL"
    input_prefix: str = "qwen_3_vl_"
    select_layer: int = -1
    project_to_dim: int = 1536
    device: str = "cpu"


def main():
    args = tyro.cli(Args)
    backbone = QwenReferenceBackbone(
        QwenReferenceConfig(
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            input_prefix=args.input_prefix,
            select_layer=args.select_layer,
            project_to_dim=args.project_to_dim,
            device=args.device,
        )
    )

    print("Loaded backbone:", args.backbone_type, args.model_name)
    print("Output dim:", backbone.output_dim)
    print("Input prefix:", args.input_prefix)
    print("Reference-only adapter ready.")
    print("Provide processor-generated inputs to `extract_backbone_features(...)`.")
    _ = np.zeros((1,), dtype=np.int32)


if __name__ == "__main__":
    main()
