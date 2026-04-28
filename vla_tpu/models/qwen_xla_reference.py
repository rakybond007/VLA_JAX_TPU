from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

try:
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    xm = None
    Qwen3VLForConditionalGeneration = None
    Qwen3_5ForConditionalGeneration = None


BackboneKind = Literal["qwen3_vl", "qwen3_5"]


@dataclass
class QwenXLAConfig:
    backbone_type: BackboneKind
    model_name: str
    input_prefix: str
    select_layer: int | tuple[int, ...] = -1
    project_to_dim: int | None = 1536
    torch_dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


class QwenXLABackbone:
    """PyTorch/XLA-backed reference runner for Qwen backbones on TPU."""

    def __init__(self, config: QwenXLAConfig):
        if torch is None or xm is None:
            raise ImportError(
                "Qwen XLA backbones require torch, torch_xla and transformers on the TPU VM."
            )

        self.config = config
        self.device = xm.xla_device()
        self.model = self._load_model(config)
        self.model.eval()
        self.model.to(self.device)

        hidden_size = self.model.model.language_model.config.hidden_size
        if config.project_to_dim is None:
            self.projector = None
            self.output_dim = hidden_size
        else:
            self.projector = torch.nn.Linear(hidden_size, config.project_to_dim, bias=True)
            self.projector.to(self.device, dtype=self._torch_dtype(config.torch_dtype))
            self.output_dim = config.project_to_dim

    def _load_model(self, config: QwenXLAConfig):
        kwargs = {
            "_attn_implementation": config.attention_implementation,
            "torch_dtype": self._torch_dtype(config.torch_dtype),
            "trust_remote_code": True,
        }
        if config.backbone_type == "qwen3_vl":
            return Qwen3VLForConditionalGeneration.from_pretrained(config.model_name, **kwargs)
        if config.backbone_type == "qwen3_5":
            return Qwen3_5ForConditionalGeneration.from_pretrained(config.model_name, **kwargs)
        raise ValueError(f"Unsupported backbone_type: {config.backbone_type}")

    @staticmethod
    def _torch_dtype(name: str):
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported torch dtype: {name}")
        return mapping[name]

    def _extract_prefixed_inputs(self, batch: Dict[str, torch.Tensor]):
        prefix = self.config.input_prefix
        extracted = {
            key.removeprefix(prefix): value
            for key, value in batch.items()
            if key.startswith(prefix)
        }
        if not extracted:
            raise ValueError(f"No inputs found with prefix '{prefix}'")
        return extracted

    def _normalize_inputs(self, qwen_input: Dict[str, torch.Tensor]):
        normalized = {}
        for key, value in qwen_input.items():
            if isinstance(value, torch.Tensor):
                normalized[key] = value.to(self.device)
            else:
                normalized[key] = value

        if "pixel_values" in normalized and isinstance(normalized["pixel_values"], torch.Tensor):
            if normalized["pixel_values"].ndim == 3:
                pv = normalized["pixel_values"]
                normalized["pixel_values"] = pv.reshape(-1, pv.shape[-1])

        if "image_grid_thw" in normalized and isinstance(normalized["image_grid_thw"], torch.Tensor):
            if normalized["image_grid_thw"].ndim == 3:
                grid = normalized["image_grid_thw"]
                normalized["image_grid_thw"] = grid.reshape(-1, 3)

        return normalized

    def _select_hidden_states(self, hidden_states: Iterable[torch.Tensor]):
        indices = self.config.select_layer
        if isinstance(indices, int):
            indices = (indices,)
        selected = [hidden_states[idx] for idx in indices]

        if self.projector is not None:
            target_dtype = next(self.projector.parameters()).dtype
            selected = [self.projector(feat.to(target_dtype)) for feat in selected]

        return selected[0] if len(selected) == 1 else selected

    def extract_backbone_features(self, batch: Dict[str, torch.Tensor]):
        with torch.no_grad():
            qwen_input = self._normalize_inputs(self._extract_prefixed_inputs(batch))
            outputs = self.model(**qwen_input, output_hidden_states=True, return_dict=True)
            features = self._select_hidden_states(outputs.hidden_states)

            result = {
                "backbone_features": features,
                "backbone_attention_mask": qwen_input.get("attention_mask"),
            }
            if "input_ids" in qwen_input:
                image_token_id = self.model.model.config.image_token_id
                result["image_mask"] = qwen_input["input_ids"] == image_token_id

            xm.mark_step()
            return result
