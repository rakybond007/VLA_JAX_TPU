from __future__ import annotations

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

from vla_tpu.configs.base import ExperimentConfig
from vla_tpu.models.action_head import build_action_head
from vla_tpu.models.backbone import build_backbone


class VLATPUPolicy(nn.Module):
    config: ExperimentConfig

    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], train: bool = False) -> Dict[str, jnp.ndarray]:
        backbone_outputs = build_backbone(self.config.backbone, name="backbone")(batch, train=train)
        action_outputs = build_action_head(self.config.action_head, name="action_head")(
            backbone_outputs=backbone_outputs,
            state=batch["state"],
            batch=batch,
            train=train,
        )
        if not isinstance(action_outputs, dict):
            action_outputs = {"action_pred": action_outputs}
        outputs = {
            "backbone_features": backbone_outputs["backbone_features"],
            "action_pred": action_outputs["action_pred"],
        }
        for key, value in action_outputs.items():
            if key not in outputs:
                outputs[key] = value
        return outputs
