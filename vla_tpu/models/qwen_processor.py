from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoProcessor
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    AutoProcessor = None
    AutoConfig = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - optional dependency path
    process_vision_info = None


@dataclass
class QwenProcessorConfig:
    model_name: str
    input_prefix: str
    do_resize: bool = False


class QwenProcessorAdapter:
    """Small wrapper around the official Qwen processor stack.

    This follows the common pattern:
    - `AutoProcessor.from_pretrained(...)`
    - `qwen_vl_utils.process_vision_info(...)`
    """

    def __init__(self, config: QwenProcessorConfig):
        if AutoProcessor is None or process_vision_info is None:
            raise ImportError(
                "Qwen processor support requires optional deps. "
                "Install with `pip install -e .[qwen-ref]`."
            )
        self.config = config
        model_source = self._resolve_processor_source(config.model_name)
        try:
            self.processor = AutoProcessor.from_pretrained(model_source)
        except OSError:
            fallback_source = self._resolve_local_snapshot(config.model_name)
            if fallback_source is None or str(fallback_source) == str(model_source):
                raise
            self.processor = AutoProcessor.from_pretrained(str(fallback_source))
        self.processor.tokenizer.padding_side = "left"
        if AutoConfig is not None:
            model_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
            self.image_token_id = getattr(model_cfg, "image_token_id", None)
            if self.image_token_id is None and hasattr(model_cfg, "text_config"):
                self.image_token_id = getattr(model_cfg.text_config, "image_token_id", None)
        else:
            self.image_token_id = None

    @staticmethod
    def _resolve_local_snapshot(model_name: str) -> Path | None:
        if "/" not in model_name:
            return None
        owner, repo = model_name.split("/", 1)
        hub_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{repo}"
        if not hub_root.exists():
            return None
        refs_main = hub_root / "refs" / "main"
        if refs_main.exists():
            snapshot = hub_root / "snapshots" / refs_main.read_text().strip()
            if snapshot.exists() and (snapshot / "preprocessor_config.json").exists():
                return snapshot
        snapshots_dir = hub_root / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(
                path
                for path in snapshots_dir.iterdir()
                if path.is_dir() and (path / "preprocessor_config.json").exists()
            )
            if snapshots:
                return snapshots[-1]
        return None

    @classmethod
    def _resolve_processor_source(cls, model_name: str) -> str:
        local_snapshot = cls._resolve_local_snapshot(model_name)
        if local_snapshot is not None:
            return str(local_snapshot)
        return model_name

    def build_prefixed_inputs(self, messages: list[dict[str, Any]]):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=getattr(self.processor.image_processor, "patch_size", None),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=self.config.do_resize,
            return_tensors="pt",
            **video_kwargs,
        )
        return {
            f"{self.config.input_prefix}{key}": value
            for key, value in encoded.items()
        }

    def build_batch_inputs(self, messages_list: list[list[dict[str, Any]]]):
        if torch is None:
            raise ImportError("Batch Qwen processor support requires torch.")

        text_list = []
        image_inputs = []
        video_inputs = []
        video_kwargs = {}
        for messages in messages_list:
            sample_image_inputs, sample_video_inputs, sample_video_kwargs = process_vision_info(
                messages,
                image_patch_size=getattr(self.processor.image_processor, "patch_size", None),
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            text_list.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            if sample_image_inputs:
                image_inputs.extend(sample_image_inputs)
            if sample_video_inputs:
                video_inputs.extend(sample_video_inputs)
            video_kwargs.update(sample_video_kwargs)

        encoded = self.processor(
            text=text_list,
            images=image_inputs or None,
            videos=video_inputs or None,
            do_resize=self.config.do_resize,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        return encoded
