from __future__ import annotations

import collections
from dataclasses import dataclass
import inspect
import logging
import math
import pathlib
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import tyro

from vla_tpu.utils.msgpack_numpy import Packer, unpackb


def _append_local_eval_dependency_paths() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "third_party/libero",
        repo_root / ".venv-local-libero-client/lib/python3.10/site-packages",
    ]
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)

try:
    from PIL import Image
    import websockets.sync.client
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:  # pragma: no cover
    _append_local_eval_dependency_paths()
    try:
        from PIL import Image
        import websockets.sync.client
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        Image = None
        benchmark = None
        get_libero_path = None
        OffScreenRenderEnv = None
        websockets = None


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _require_eval_deps() -> None:
    if Image is None or benchmark is None or get_libero_path is None or OffScreenRenderEnv is None or websockets is None:
        raise ImportError(
            "LIBERO eval dependencies are missing. Install websocket/imageio deps and LIBERO itself."
        )


class WebsocketClientPolicy:
    def __init__(self, host: str, port: int):
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        self._metadata = unpackb(self._ws.recv())

    def infer(self, obs: dict) -> dict:
        start = time.time()
        logging.info("client:send_request")
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        logging.info("client:recv_response elapsed=%.3f", time.time() - start)
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    @property
    def metadata(self) -> dict:
        return self._metadata


class LocalRuntimePolicy:
    def __init__(self, config: str, checkpoint_dir: str, dataset_root: str):
        from vla_tpu.serving.websocket_policy_server import PolicyRuntime

        self._runtime = PolicyRuntime(
            config_name=config,
            checkpoint_dir=checkpoint_dir or None,
            dataset_root=dataset_root or None,
        )
        self._metadata = {
            "action_horizon": self._runtime.config.action_head.action_horizon,
            "available_policies": [self._runtime.config.name],
            "dataset_type": self._runtime.config.data.dataset_type,
            "camera_keys": list(self._runtime.config.data.camera_keys[: self._runtime.config.data.num_cameras]),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.infer(obs)

    @property
    def metadata(self) -> dict:
        return self._metadata


def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[:2] == (size, size):
        return img.astype(np.uint8)
    pil = Image.fromarray(img)
    width, height = pil.size
    ratio = max(width / size, height / size)
    resized = pil.resize((int(width / ratio), int(height / ratio)), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), 0)
    left = (size - resized.size[0]) // 2
    top = (size - resized.size[1]) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def _quat2euler_rpy(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


@dataclass
class Args:
    policy_mode: str = "websocket"
    config: str = "libero_full_jax_qwen3_vl_legacy_eval"
    checkpoint_dir: str = ""
    dataset_root: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_idx: int = -1
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    video_out_path: str = "outputs/libero_rollouts"
    seed: int = 7
    state_rot_format: str = "axis_angle"
    image_flip_180: bool = False
    image_resize_with_pad: bool = False
    video_flip_180: bool = False


def eval_libero(args: Args) -> None:
    _require_eval_deps()
    _patch_torch_load_for_libero()
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.policy_mode == "websocket":
        client = WebsocketClientPolicy(args.host, args.port)
    elif args.policy_mode == "local":
        client = LocalRuntimePolicy(
            config=args.config,
            checkpoint_dir=args.checkpoint_dir,
            dataset_root=args.dataset_root,
        )
    else:
        raise ValueError(f"Unsupported policy_mode: {args.policy_mode}")
    logging.info("client:connected metadata=%s", client.metadata)

    if args.task_suite_name.startswith("libero_spatial"):
        max_steps = 220
    elif args.task_suite_name.startswith("libero_object"):
        max_steps = 280
    elif args.task_suite_name.startswith("libero_goal"):
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    else:
        max_steps = 400

    total_episodes = 0
    total_successes = 0
    for task_id in range(task_suite.n_tasks):
        if args.task_idx != -1 and task_id != args.task_idx:
            continue

        task = task_suite.get_task(task_id)
        logging.info("eval:task_id=%s task_language=%s", task_id, task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_segment = task_description.replace(" ", "_")
        task_episodes = 0
        task_successes = 0

        for episode_idx in range(args.num_trials_per_task):
            failure_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_failure.mp4"
            success_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_success.mp4"
            if failure_path.exists() or success_path.exists():
                total_episodes += 1
                task_episodes += 1
                if success_path.exists():
                    total_successes += 1
                    task_successes += 1
                logging.info("task=%s episode=%s skipped_existing=true", task_segment, episode_idx)
                continue

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            logging.info("eval:episode_start task=%s episode=%s", task_segment, episode_idx)
            action_plan = collections.deque()
            replay_images = []
            done = False

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue

                img = np.asarray(obs["agentview_image"])
                wrist_img = np.asarray(obs["robot0_eye_in_hand_image"])
                if args.image_flip_180:
                    img = np.ascontiguousarray(img[::-1, ::-1])
                    wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1])
                else:
                    img = np.ascontiguousarray(img)
                    wrist_img = np.ascontiguousarray(wrist_img)
                if args.image_resize_with_pad:
                    img = _resize_with_pad(img, args.resize_size)
                    wrist_img = _resize_with_pad(wrist_img, args.resize_size)
                replay_frame = np.ascontiguousarray(img[::-1, ::-1]) if args.video_flip_180 else img
                replay_images.append(replay_frame)

                if not action_plan:
                    logging.info("eval:request_action t=%s", t)
                    element = {
                        "video.front_view": np.array([img]),
                        "video.left_wrist_view": np.array([wrist_img]),
                        "state.eef_pos_absolute": obs["robot0_eef_pos"],
                        "state.eef_rot_absolute": (
                            _quat2euler_rpy(obs["robot0_eef_quat"])
                            if args.state_rot_format == "euler_rpy"
                            else _quat2axisangle(obs["robot0_eef_quat"])
                        ),
                        "state.gripper_close": obs["robot0_gripper_qpos"],
                        "annotation.human.action.task_description": [str(task_description)],
                    }
                    action_dict = client.infer(element)
                    action_chunk = np.concatenate(
                        [
                            np.asarray(action_dict["action.eef_pos_delta"], dtype=np.float32),
                            np.asarray(action_dict["action.eef_rot_delta"], dtype=np.float32),
                            np.asarray(action_dict["action.gripper_close"], dtype=np.float32).reshape(-1, 1),
                        ],
                        axis=-1,
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])
                    logging.info("eval:received_action_chunk len=%s", len(action_chunk))

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    total_successes += 1
                    task_successes += 1
                    break

            total_episodes += 1
            task_episodes += 1

            suffix = "success" if done else "failure"
            out_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=30)
            logging.info("task=%s episode=%s success=%s saved=%s", task_segment, episode_idx, done, out_path)

        if task_episodes > 0:
            logging.info(
                "task=%s task_success_rate=%.4f total_success_rate=%.4f",
                task_segment,
                task_successes / task_episodes,
                total_successes / total_episodes,
            )

    result_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    with result_path.open("w") as f:
        f.write(f"Total success rate: {total_successes / max(1, total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")
    logging.info("results saved to %s", result_path)


def main():
    eval_libero(tyro.cli(Args))


def _patch_torch_load_for_libero() -> None:
    import torch
    import numpy as np

    signature = inspect.signature(torch.load)
    if "weights_only" not in signature.parameters:
        return

    original_load = torch.load
    original_serialization_load = torch.serialization.load
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load
    torch.serialization.load = compat_load


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
