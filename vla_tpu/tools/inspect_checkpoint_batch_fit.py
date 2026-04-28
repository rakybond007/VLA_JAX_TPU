from __future__ import annotations

from dataclasses import dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.utils.checkpointing import restore_checkpoint


@dataclass
class Args:
    config: str = "libero_full_jax_qwen3_vl"
    checkpoint_dir: str = ""
    dataset_root: str = ""
    batch_size: int = 2
    num_batches: int = 2
    train_split: bool = False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def main() -> None:
    args = tyro.cli(Args)
    config = get_experiment_config(args.config)
    if args.dataset_root:
        config.data.dataset_root = args.dataset_root
    if args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir

    model = VLATPUPolicy(config)
    batch_provider = make_batch_provider(config, train=args.train_split)
    first_batch = batch_provider.next_batch(batch_size=args.batch_size)
    variables = model.init(jax.random.PRNGKey(0), first_batch, train=False)
    restored_params = restore_checkpoint(config.train.checkpoint_dir, variables["params"])

    mses = []
    maes = []
    cosines = []
    pred_means = []
    pred_stds = []
    gt_means = []
    gt_stds = []
    pred_abs_means = []
    gt_abs_means = []
    first_pred_t0 = None
    first_gt_t0 = None
    pred_shape = None
    gt_shape = None

    for batch_idx in range(max(1, args.num_batches)):
        if batch_idx == 0:
            batch = first_batch
        else:
            batch = batch_provider.next_batch(batch_size=args.batch_size)
        outputs = model.apply({"params": restored_params}, batch, train=False)
        pred = np.asarray(jax.device_get(outputs["action_pred"]))
        gt = np.asarray(batch["actions"])
        diff = pred - gt

        if first_pred_t0 is None:
            first_pred_t0 = pred[0, 0].tolist()
            first_gt_t0 = gt[0, 0].tolist()
            pred_shape = list(pred.shape)
            gt_shape = list(gt.shape)

        mses.append(float(np.mean(diff ** 2)))
        maes.append(float(np.mean(np.abs(diff))))
        cosines.append(_cosine(pred, gt))
        pred_means.append(float(pred.mean()))
        pred_stds.append(float(pred.std()))
        gt_means.append(float(gt.mean()))
        gt_stds.append(float(gt.std()))
        pred_abs_means.append(float(np.mean(np.abs(pred))))
        gt_abs_means.append(float(np.mean(np.abs(gt))))

    report = {
        "num_batches": max(1, args.num_batches),
        "pred_shape": pred_shape,
        "gt_shape": gt_shape,
        "mse": float(np.mean(mses)),
        "mae": float(np.mean(maes)),
        "cosine": float(np.mean(cosines)),
        "pred_mean": float(np.mean(pred_means)),
        "pred_std": float(np.mean(pred_stds)),
        "gt_mean": float(np.mean(gt_means)),
        "gt_std": float(np.mean(gt_stds)),
        "pred_abs_mean": float(np.mean(pred_abs_means)),
        "gt_abs_mean": float(np.mean(gt_abs_means)),
        "first_pred_t0": first_pred_t0,
        "first_gt_t0": first_gt_t0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
