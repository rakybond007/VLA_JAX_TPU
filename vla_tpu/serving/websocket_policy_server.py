from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import websockets
import websockets.asyncio.server
import websockets.frames

from vla_tpu.configs.base import get_experiment_config
from vla_tpu.data.factory import make_batch_provider
from vla_tpu.data.lerobot_dataset import LeRobotObservationProcessor
from vla_tpu.models.policy import VLATPUPolicy
from vla_tpu.utils.checkpointing import restore_checkpoint
from vla_tpu.utils.msgpack_numpy import Packer, unpackb


class PolicyRuntime:
    def __init__(self, config_name: str, checkpoint_dir: str | None = None, dataset_root: str | None = None):
        logging.info("runtime:init_config_start")
        self.config = get_experiment_config(config_name)
        if dataset_root:
            self.config.data.dataset_root = dataset_root
        self.checkpoint_dir = checkpoint_dir or self.config.train.checkpoint_dir
        logging.info("runtime:init_batch_provider_start")
        self.batch_provider = make_batch_provider(self.config, train=False)
        logging.info("runtime:example_batch_start")
        self.example_batch = self.batch_provider.next_batch(batch_size=1)
        logging.info("runtime:model_init_start")
        self.model = VLATPUPolicy(self.config)
        self.variables = self.model.init(jax.random.PRNGKey(0), self.example_batch, train=False)
        logging.info("runtime:restore_checkpoint_start path=%s", self.checkpoint_dir)
        restored_params = restore_checkpoint(self.checkpoint_dir, self.variables["params"])
        self.variables = {"params": restored_params}
        logging.info("runtime:restore_checkpoint_done")
        self.obs_processor = LeRobotObservationProcessor(self.config)
        self._infer_seed = 0
        # Pure Qwen3-VL serving currently uses dynamic vision grid metadata and a NumPy-based
        # exact positional interpolation path, so keep eval inference eager for correctness.
        if self.config.backbone.impl == "jax_qwen3_vl_pure":
            self.apply = self._apply
        else:
            self.apply = jax.jit(self._apply)

    def _apply(self, batch):
        return self.model.apply(self.variables, batch, train=False)["action_pred"]

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        logging.info("infer:start")
        batch = self.obs_processor.process(observation)
        batch["inference_seed"] = np.asarray(self._infer_seed, dtype=np.uint32)
        self._infer_seed += 1
        logging.info("infer:processed_obs elapsed=%.3f", time.time() - start)
        batch = {key: jnp.asarray(value) for key, value in batch.items()}
        action_pred = self.apply(batch)
        logging.info("infer:model_apply elapsed=%.3f", time.time() - start)
        action = jax.device_get(action_pred)[0]
        logging.info("infer:device_get elapsed=%.3f", time.time() - start)
        unprocessed = self.obs_processor.unprocess_action(action)
        horizon = action.shape[0]
        pos = unprocessed["action.eef_pos_delta"]
        rot = unprocessed["action.eef_rot_delta"]
        grip = unprocessed["action.gripper_close"].reshape(-1)
        return {
            "actions": action,
            "actions_unprocessed": unprocessed,
            "action.eef_pos_delta": pos,
            "action.eef_rot_delta": rot,
            "action.gripper_close": grip,
            "action_horizon": horizon,
        }


class WebsocketPolicyServer:
    def __init__(self, runtime: PolicyRuntime, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.metadata = {
            "action_horizon": runtime.config.action_head.action_horizon,
            "available_policies": [runtime.config.name],
            "dataset_type": runtime.config.data.dataset_type,
            "camera_keys": list(runtime.config.data.camera_keys[: runtime.config.data.num_cameras]),
        }

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info("Websocket policy server listening on %s:%s", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        packer = Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                obs = unpackb(await websocket.recv())
                logging.info("handler:received_message")
                if isinstance(obs, dict) and "observation" in obs:
                    obs = obs["observation"]
                action = self.runtime.infer(obs)
                await websocket.send(packer.pack(action))
                logging.info("handler:sent_response")
            except websockets.ConnectionClosed:
                break
            except Exception as exc:
                await websocket.send(str(exc))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                raise


@dataclass
class Args:
    config: str = "libero_full"
    checkpoint_dir: str = ""
    dataset_root: str = ""
    host: str = "0.0.0.0"
    port: int = 8000


def main():
    args = tyro.cli(Args)
    runtime = PolicyRuntime(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir or None,
        dataset_root=args.dataset_root or None,
    )
    server = WebsocketPolicyServer(runtime=runtime, host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
