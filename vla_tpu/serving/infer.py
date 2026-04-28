from __future__ import annotations

from dataclasses import dataclass

from flax import core
import jax
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.models.qwen3_vl_weight_loader import (
    load_hf_qwen3_vl_state_dict,
    load_hf_weights_into_jax_qwen3_vl_pure_backbone,
)


@dataclass
class Args:
    config: str = "small_debug"
    batch_size: int = 1


def main():
    args = tyro.cli(Args)
    config = get_experiment_config(args.config)
    batch_provider = make_batch_provider(config, train=False)
    batch = batch_provider.next_batch(batch_size=args.batch_size)
    model = VLATPUPolicy(config)
    variables = model.init(jax.random.PRNGKey(0), batch, train=False)
    if config.backbone.impl == "jax_qwen3_vl_pure":
        state_dict = load_hf_qwen3_vl_state_dict(config.backbone.model_name, torch_dtype="float32")
        mutable = core.unfreeze(variables["params"])
        loaded_backbone, summary = load_hf_weights_into_jax_qwen3_vl_pure_backbone(
            core.freeze(mutable["backbone"]["pure_backbone"]),
            state_dict,
        )
        mutable["backbone"]["pure_backbone"] = core.unfreeze(loaded_backbone)
        variables = {"params": core.freeze(mutable)}
        print(
            f"loaded pure Qwen3-VL backbone weights: loaded={summary.loaded} "
            f"skipped={summary.skipped} mismatched={summary.mismatched}"
        )
    outputs = model.apply(variables, batch, train=False)
    print("backbone_features shape:", outputs["backbone_features"].shape)
    print("action_pred shape:", outputs["action_pred"].shape)


if __name__ == "__main__":
    main()
