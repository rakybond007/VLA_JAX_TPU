from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image as PILImage

from vla_tpu.configs.base import ExperimentConfig
from vla_tpu.models.qwen_processor import QwenProcessorAdapter, QwenProcessorConfig

try:
    import cv2
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency path
    cv2 = None
    pd = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency path
    Image = None


DEFAULT_ROTATION_BOUNDS = {
    "rotation_6d": {
        "min": np.full((6,), -1.0, dtype=np.float32),
        "max": np.full((6,), 1.0, dtype=np.float32),
    },
}


@dataclass(frozen=True)
class EpisodeRef:
    episode_index: int
    task: str
    length: int


def _require_lerobot_deps() -> None:
    if pd is None:
        raise ImportError(
            "LeRobot loading requires optional deps. "
            "Install with `pip install -e .[robocasa-data]`."
        )


def _tokenize_instruction(text: str, vocab_size: int, length: int) -> np.ndarray:
    tokens = np.zeros((length,), dtype=np.int32)
    for idx, piece in enumerate(text.lower().split()[:length]):
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        tokens[idx] = int.from_bytes(digest[:4], "little") % vocab_size
    return tokens


def _crop_bounds(
    height: int,
    width: int,
    scale: float,
    train: bool,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    crop_h = max(1, min(height, int(round(height * scale))))
    crop_w = max(1, min(width, int(round(width * scale))))
    if train:
        top = int(rng.integers(0, max(1, height - crop_h + 1)))
        left = int(rng.integers(0, max(1, width - crop_w + 1)))
    else:
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
    return top, top + crop_h, left, left + crop_w


def _adjust_hsv(
    frame: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if brightness <= 0 and contrast <= 0 and saturation <= 0 and hue <= 0:
        return frame

    out = frame.astype(np.float32) / 255.0
    if brightness > 0:
        out *= float(rng.uniform(1.0 - brightness, 1.0 + brightness))
    if contrast > 0:
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * float(rng.uniform(1.0 - contrast, 1.0 + contrast)) + mean

    if saturation > 0 or hue > 0:
        if cv2 is None:
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(np.clip(out * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        if saturation > 0:
            hsv[..., 1] *= float(rng.uniform(1.0 - saturation, 1.0 + saturation))
        if hue > 0:
            hsv[..., 0] += float(rng.uniform(-hue, hue) * 180.0)
        hsv[..., 0] %= 180.0
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _euler_rpy_to_rotation_6d(euler: np.ndarray) -> np.ndarray:
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    rot = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float32)
    rot[..., 0, 0] = cy * cp
    rot[..., 0, 1] = cy * sp * sr - sy * cr
    rot[..., 0, 2] = cy * sp * cr + sy * sr
    rot[..., 1, 0] = sy * cp
    rot[..., 1, 1] = sy * sp * sr + cy * cr
    rot[..., 1, 2] = sy * sp * cr - cy * sr
    rot[..., 2, 0] = -sp
    rot[..., 2, 1] = cp * sr
    rot[..., 2, 2] = cp * cr
    return rot[..., :, :2].reshape(euler.shape[:-1] + (6,))


def _axis_angle_to_rotation_6d(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = np.maximum(angle, 1e-8)
    axis = axis_angle / safe_angle

    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    ca = np.cos(angle)
    sa = np.sin(angle)
    one_minus_ca = 1.0 - ca

    rot = np.empty(axis_angle.shape[:-1] + (3, 3), dtype=np.float32)
    rot[..., 0, 0] = (ca + x * x * one_minus_ca)[..., 0]
    rot[..., 0, 1] = (x * y * one_minus_ca - z * sa)[..., 0]
    rot[..., 0, 2] = (x * z * one_minus_ca + y * sa)[..., 0]
    rot[..., 1, 0] = (y * x * one_minus_ca + z * sa)[..., 0]
    rot[..., 1, 1] = (ca + y * y * one_minus_ca)[..., 0]
    rot[..., 1, 2] = (y * z * one_minus_ca - x * sa)[..., 0]
    rot[..., 2, 0] = (z * x * one_minus_ca - y * sa)[..., 0]
    rot[..., 2, 1] = (z * y * one_minus_ca + x * sa)[..., 0]
    rot[..., 2, 2] = (ca + z * z * one_minus_ca)[..., 0]

    zero_angle = (angle[..., 0] < 1e-8)
    if np.any(zero_angle):
        rot[zero_angle] = np.eye(3, dtype=np.float32)
    return rot[..., :, :2].reshape(axis_angle.shape[:-1] + (6,))


def _normalize(values: np.ndarray, mode: str, stats: dict[str, list[float]]) -> np.ndarray:
    if mode == "min_max":
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        mask = min_v != max_v
        out = np.zeros_like(values, dtype=np.float32)
        out[..., mask] = (values[..., mask] - min_v[mask]) / (max_v[mask] - min_v[mask])
        out[..., mask] = 2.0 * out[..., mask] - 1.0
        return out
    if mode == "mean_std":
        mean_v = np.asarray(stats["mean"], dtype=np.float32)
        std_v = np.asarray(stats["std"], dtype=np.float32)
        mask = std_v != 0
        out = np.zeros_like(values, dtype=np.float32)
        out[..., mask] = (values[..., mask] - mean_v[mask]) / std_v[mask]
        return out
    if mode == "binary":
        return (values > 0.5).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _denormalize(values: np.ndarray, mode: str, stats: dict[str, list[float]]) -> np.ndarray:
    if mode == "min_max":
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        mask = min_v != max_v
        out = values.astype(np.float32).copy()
        out[..., mask] = (out[..., mask] + 1.0) * 0.5
        out[..., mask] = out[..., mask] * (max_v[mask] - min_v[mask]) + min_v[mask]
        return out
    if mode == "mean_std":
        mean_v = np.asarray(stats["mean"], dtype=np.float32)
        std_v = np.asarray(stats["std"], dtype=np.float32)
        mask = std_v != 0
        out = values.astype(np.float32).copy()
        out[..., mask] = out[..., mask] * std_v[mask] + mean_v[mask]
        return out
    if mode == "binary":
        return (values > 0).astype(np.float32)
    raise ValueError(f"Unsupported denormalization mode: {mode}")


class LeRobotDataset:
    def __init__(self, config: ExperimentConfig):
        _require_lerobot_deps()

        self.config = config
        self.data_cfg = config.data
        self.action_cfg = config.action_head
        self.backbone_cfg = config.backbone
        self.dataset_root = Path(self.data_cfg.dataset_root).expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.info = json.loads((self.dataset_root / "meta/info.json").read_text())
        self.modality = json.loads((self.dataset_root / "meta/modality.json").read_text())
        self.stats = json.loads((self.dataset_root / "meta/stats.json").read_text())
        self.episodes = self._load_episodes()[: self.data_cfg.max_episodes]
        if self.data_cfg.fixed_episode_indices:
            fixed_set = set(int(x) for x in self.data_cfg.fixed_episode_indices)
            filtered = [episode for episode in self.episodes if int(episode.episode_index) in fixed_set]
            if not filtered:
                raise ValueError(
                    f"No episodes found for fixed_episode_indices={sorted(fixed_set)} "
                    f"under dataset root {self.dataset_root}"
                )
            self.episodes = filtered
        self.tasks = self._load_tasks()
        self.sample_refs = self._build_sample_refs(self.episodes)
        self._episode_cache: dict[int, Any] = {}
        self.rng = np.random.default_rng(self.data_cfg.seed)
        self.qwen_processor = None
        if self.backbone_cfg.impl == "jax_qwen3_vl_pure":
            self.qwen_processor = QwenProcessorAdapter(
                QwenProcessorConfig(
                    model_name=self.backbone_cfg.model_name,
                    input_prefix="",
                    do_resize=False,
                )
            )

    def _load_episodes(self) -> list[EpisodeRef]:
        episodes = []
        with (self.dataset_root / "meta/episodes.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                task = row["tasks"][0] if row.get("tasks") else ""
                episodes.append(
                    EpisodeRef(
                        episode_index=int(row["episode_index"]),
                        task=task,
                        length=int(row["length"]),
                    )
                )
        return episodes

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.dataset_root / "meta/tasks.jsonl"
        if not tasks_path.exists():
            return {}
        tasks: dict[int, str] = {}
        with tasks_path.open() as f:
            for line in f:
                row = json.loads(line)
                tasks[int(row["task_index"])] = str(row["task"])
        return tasks

    def _build_sample_refs(self, episodes: list[EpisodeRef]) -> list[tuple[int, int]]:
        refs: list[tuple[int, int]] = []
        stride = max(1, self.data_cfg.episode_stride)
        for episode in episodes:
            for start in range(0, episode.length, stride):
                refs.append((episode.episode_index, start))
                if len(refs) >= self.data_cfg.max_samples:
                    return refs
        return refs

    def __len__(self) -> int:
        return len(self.sample_refs)

    def _episode_path(self, episode_index: int) -> Path:
        chunk_size = int(self.info["chunks_size"])
        episode_chunk = episode_index // chunk_size
        return self.dataset_root / self.info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )

    def _video_path(self, episode_index: int, camera_key: str) -> Path:
        chunk_size = int(self.info["chunks_size"])
        episode_chunk = episode_index // chunk_size
        original_key = self.modality["video"][camera_key].get("original_key", camera_key)
        return self.dataset_root / self.info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=original_key,
        )

    def _load_episode_df(self, episode_index: int):
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            return cached
        df = pd.read_parquet(self._episode_path(episode_index))
        self._episode_cache = {episode_index: df}
        return df

    def _retrieve_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        absolute: bool,
    ) -> np.ndarray:
        max_length = array.shape[0]
        front_mask = step_indices < 0
        back_mask = step_indices >= max_length
        valid_indices = np.clip(step_indices, 0, max_length - 1)
        values = array[valid_indices]
        if not absolute:
            values = values.copy()
            values[front_mask | back_mask] = 0
        return values.astype(np.float32)

    def _read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        if cv2 is None:
            raise ImportError("Reading LeRobot videos requires opencv-python-headless.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _transform_frame(
        self,
        frame: np.ndarray,
        crop_bounds: tuple[int, int, int, int],
        train: bool,
    ) -> np.ndarray:
        top, bottom, left, right = crop_bounds
        frame = frame[top:bottom, left:right]
        if cv2 is not None:
            frame = cv2.resize(
                frame,
                (self.data_cfg.image_width, self.data_cfg.image_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            if Image is None:
                raise ImportError("Image resize requires either cv2 or Pillow.")
            frame = np.asarray(
                Image.fromarray(frame).resize(
                    (self.data_cfg.image_width, self.data_cfg.image_height),
                    resample=Image.BILINEAR,
                ),
                dtype=np.uint8,
            )
        if train and self.data_cfg.use_color_jitter:
            frame = _adjust_hsv(
                frame,
                brightness=self.data_cfg.color_jitter_brightness,
                contrast=self.data_cfg.color_jitter_contrast,
                saturation=self.data_cfg.color_jitter_saturation,
                hue=self.data_cfg.color_jitter_hue,
                rng=self.rng,
            )
        return frame.astype(np.uint8)

    def _slice_state_or_action(
        self,
        df,
        base_index: int,
        source_key: str,
        keys: tuple[str, ...],
        normalization_modes: dict[str, str],
        rotation_targets: dict[str, str],
        delta_indices: tuple[int, ...],
    ) -> np.ndarray:
        array = np.stack(df[source_key]).astype(np.float32)
        parts = []
        for key in keys:
            _, short_key = key.split(".", 1)
            meta = self.modality["state" if source_key == "observation.state" else "action"][short_key]
            start = int(meta["start"])
            end = int(meta["end"])
            absolute = bool(meta["absolute"])
            indices = np.asarray(delta_indices, dtype=np.int32) + base_index
            values = self._retrieve_and_pad(array[:, start:end], indices, absolute=absolute)

            rotation_type = meta.get("rotation_type")
            if rotation_targets.get(key) == "rotation_6d":
                if rotation_type != "euler_angles_rpy":
                    raise ValueError(f"Unsupported rotation conversion for {key}: {rotation_type}")
                values = _euler_rpy_to_rotation_6d(values)
                bounds = DEFAULT_ROTATION_BOUNDS["rotation_6d"]
                values = _normalize(values, "min_max", bounds)
            elif key in normalization_modes:
                stats = self.stats[source_key]
                sliced_stats = {
                    stat_name: stat_values[start:end]
                    for stat_name, stat_values in stats.items()
                    if isinstance(stat_values, list)
                }
                values = _normalize(values, normalization_modes[key], sliced_stats)

            parts.append(values)

        merged = np.concatenate(parts, axis=-1).astype(np.float32)
        if merged.shape[0] == 1:
            return merged[0]
        return merged.reshape(-1, merged.shape[-1])

    def _get_instruction(self, df, base_index: int, episode: EpisodeRef) -> str:
        task_key = self.data_cfg.language_key
        _, short_key = task_key.split(".", 1)
        annotation_meta = self.modality.get("annotation", {})
        original_key = annotation_meta.get(short_key, {}).get("original_key")
        if original_key and original_key in df.columns:
            task_index = int(df.iloc[base_index][original_key])
            if task_index in self.tasks:
                return self.tasks[task_index]
        return episode.task

    def __getitem__(self, index: int, train: bool = False) -> dict[str, np.ndarray]:
        episode_index, base_index = self.sample_refs[index]
        episode = next(ep for ep in self.episodes if ep.episode_index == episode_index)
        df = self._load_episode_df(episode_index)

        state = self._slice_state_or_action(
            df=df,
            base_index=base_index,
            source_key="observation.state",
            keys=self.data_cfg.state_keys,
            normalization_modes=self.data_cfg.state_normalization_modes,
            rotation_targets=self.data_cfg.state_rotation_targets,
            delta_indices=self.data_cfg.observation_indices,
        )
        actions = self._slice_state_or_action(
            df=df,
            base_index=base_index,
            source_key="action",
            keys=self.data_cfg.action_keys,
            normalization_modes=self.data_cfg.action_normalization_modes,
            rotation_targets={},
            delta_indices=self.data_cfg.action_indices,
        )

        frame_indices = np.asarray(df["frame_index"], dtype=np.int32)
        obs_indices = np.asarray(self.data_cfg.observation_indices, dtype=np.int32) + base_index
        obs_indices = np.clip(obs_indices, 0, len(frame_indices) - 1)

        sample_frame = self._read_frame(self._video_path(episode_index, self.data_cfg.camera_keys[0]), int(frame_indices[obs_indices[0]]))
        crop_bounds = _crop_bounds(
            height=sample_frame.shape[0],
            width=sample_frame.shape[1],
            scale=self.data_cfg.video_crop_scale,
            train=train,
            rng=self.rng,
        )

        images = []
        for camera_key in self.data_cfg.camera_keys[: self.data_cfg.num_cameras]:
            frames = []
            video_path = self._video_path(episode_index, camera_key)
            for step_index in obs_indices:
                frame = self._read_frame(video_path, int(frame_indices[step_index]))
                frame = self._transform_frame(frame, crop_bounds=crop_bounds, train=train)
                frames.append(frame)
            images.append(np.stack(frames, axis=0))

        images_np = np.stack(images, axis=0)
        if images_np.shape[1] == 1:
            images_np = images_np[:, 0]
        else:
            images_np = images_np.reshape(-1, *images_np.shape[-3:])

        instruction = self._get_instruction(df, base_index, episode)
        if train:
            action_query = actions.astype(np.float32) + self.rng.normal(
                loc=0.0,
                scale=0.05,
                size=actions.shape,
            ).astype(np.float32)
        else:
            action_query = np.zeros_like(actions, dtype=np.float32)

        item = {
            "images": images_np,
            "state": state.astype(np.float32),
            "state_mask": np.ones_like(state, dtype=bool),
            "actions": actions.astype(np.float32),
            "action_mask": np.ones_like(actions, dtype=np.float32),
            "action_query": action_query,
        }
        if self.qwen_processor is None:
            item["instruction_tokens"] = _tokenize_instruction(
                instruction,
                vocab_size=self.backbone_cfg.text_vocab_size,
                length=self.data_cfg.instruction_length,
            )
        else:
            item["instruction"] = instruction
        return item

    def _build_qwen_batch(self, batch_items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        messages_list = []
        for item in batch_items:
            qwen_images = [PILImage.fromarray(image.astype(np.uint8)) for image in item["images"]]
            content = [{"type": "image", "image": image} for image in qwen_images]
            content.append({"type": "text", "text": item["instruction"]})
            messages_list.append([{"role": "user", "content": content}])

        encoded = self.qwen_processor.build_batch_inputs(messages_list)
        input_ids = encoded["input_ids"].cpu().numpy().astype(np.int32)
        attention_mask = encoded["attention_mask"].cpu().numpy().astype(np.int32)
        image_grid_thw = encoded["image_grid_thw"].cpu().numpy().astype(np.int32)
        pixel_values = encoded["pixel_values"].cpu().numpy().astype(np.float32)

        if self.qwen_processor.image_token_id is None:
            raise ValueError("Unable to determine Qwen image_token_id for pure Qwen3-VL batching.")
        image_token_mask = input_ids == int(self.qwen_processor.image_token_id)
        from vla_tpu.models.qwen3_vl_jax import get_rope_index

        position_ids = np.asarray(
            get_rope_index(
                jnp.asarray(input_ids),
                jnp.asarray(image_token_mask.astype(np.int32)),
                jnp.asarray(image_grid_thw),
                jnp.asarray(attention_mask),
                self.backbone_cfg.vision_spatial_merge_size,
            )
        ).astype(np.int32)

        batch = {
            "images": np.stack([item["images"] for item in batch_items], axis=0),
            "state": np.stack([item["state"] for item in batch_items], axis=0),
            "actions": np.stack([item["actions"] for item in batch_items], axis=0),
            "action_query": np.stack([item["action_query"] for item in batch_items], axis=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": image_token_mask.astype(np.int32),
            "image_token_mask": image_token_mask,
            "image_token_id": np.asarray(self.qwen_processor.image_token_id, dtype=np.int32),
            "position_ids": position_ids,
        }
        return batch

    def sample_batch(self, batch_size: int, train: bool) -> dict[str, np.ndarray]:
        batch_items: list[dict[str, np.ndarray]] = []
        max_attempts = max(batch_size * 16, 64)
        attempts = 0
        last_error: Exception | None = None
        while len(batch_items) < batch_size and attempts < max_attempts:
            index = int(self.rng.integers(0, len(self.sample_refs)))
            attempts += 1
            try:
                batch_items.append(self.__getitem__(index, train=train))
            except Exception as exc:
                last_error = exc
                continue
        if len(batch_items) < batch_size:
            raise RuntimeError(
                f"Failed to assemble LeRobot batch after {attempts} attempts; "
                f"collected {len(batch_items)}/{batch_size} items"
            ) from last_error
        if self.qwen_processor is not None:
            return self._build_qwen_batch(batch_items)
        keys = batch_items[0].keys()
        return {key: np.stack([item[key] for item in batch_items], axis=0) for key in keys}


def make_lerobot_batch(config: ExperimentConfig, batch_size: int, train: bool) -> dict[str, np.ndarray]:
    dataset = LeRobotDataset(config)
    return dataset.sample_batch(batch_size=batch_size, train=train)


class LeRobotObservationProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_cfg = config.data
        self.backbone_cfg = config.backbone
        self.dataset_root = Path(self.data_cfg.dataset_root).expanduser()
        self.modality = json.loads((self.dataset_root / "meta/modality.json").read_text())
        self.stats = json.loads((self.dataset_root / "meta/stats.json").read_text())
        self._action_layout = self._build_action_layout()
        self.qwen_processor = None
        if self.backbone_cfg.impl == "jax_qwen3_vl_pure":
            self.qwen_processor = QwenProcessorAdapter(
                QwenProcessorConfig(
                    model_name=self.backbone_cfg.model_name,
                    input_prefix="",
                    do_resize=False,
                )
            )

    def _build_action_layout(self) -> list[tuple[str, int, dict[str, list[float]] | None]]:
        action_stats = self.stats["action"]
        layout: list[tuple[str, int, dict[str, list[float]] | None]] = []
        for key in self.data_cfg.action_keys:
            _, short_key = key.split(".", 1)
            meta = self.modality["action"][short_key]
            start = int(meta["start"])
            end = int(meta["end"])
            stats = None
            if key in self.data_cfg.action_normalization_modes:
                stats = {
                    stat_name: stat_values[start:end]
                    for stat_name, stat_values in action_stats.items()
                    if isinstance(stat_values, list)
                }
            layout.append((key, end - start, stats))
        return layout

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"Expected a single frame per camera, got shape {image.shape}")
            image = image[0]
        crop_bounds = _crop_bounds(
            height=image.shape[0],
            width=image.shape[1],
            scale=self.data_cfg.video_crop_scale,
            train=False,
            rng=np.random.default_rng(self.data_cfg.seed),
        )
        return LeRobotDataset._transform_frame(self, image, crop_bounds=crop_bounds, train=False)

    def _process_state(self, observation: dict[str, Any]) -> np.ndarray:
        parts = []
        state_stats = self.stats["observation.state"]
        for key in self.data_cfg.state_keys:
            _, short_key = key.split(".", 1)
            meta = self.modality["state"][short_key]
            source_value = np.asarray(observation[key], dtype=np.float32).reshape(1, -1)
            if self.data_cfg.state_rotation_targets.get(key) == "rotation_6d":
                source_value = _axis_angle_to_rotation_6d(source_value)
                source_value = _normalize(source_value, "min_max", DEFAULT_ROTATION_BOUNDS["rotation_6d"])
            elif key in self.data_cfg.state_normalization_modes:
                start = int(meta["start"])
                end = int(meta["end"])
                sliced_stats = {
                    stat_name: stat_values[start:end]
                    for stat_name, stat_values in state_stats.items()
                    if isinstance(stat_values, list)
                }
                source_value = _normalize(
                    source_value,
                    self.data_cfg.state_normalization_modes[key],
                    sliced_stats,
                )
            parts.append(source_value[0])
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def process(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        images = []
        for camera_key in self.data_cfg.camera_keys[: self.data_cfg.num_cameras]:
            obs_key = f"video.{camera_key}"
            if obs_key not in observation:
                raise KeyError(f"Missing camera observation: {obs_key}")
            images.append(self._prepare_image(np.asarray(observation[obs_key])))
        images_np = np.stack(images, axis=0)

        task_prompt = observation.get(self.data_cfg.language_key, [""])
        if isinstance(task_prompt, list):
            task_prompt = task_prompt[0] if task_prompt else ""
        state = self._process_state(observation)
        action_query = np.zeros(
            (self.config.action_head.action_horizon, self.config.action_head.action_dim),
            dtype=np.float32,
        )
        if self.qwen_processor is not None:
            qwen_images = [PILImage.fromarray(image.astype(np.uint8)) for image in images_np]
            content = [{"type": "image", "image": image} for image in qwen_images]
            content.append({"type": "text", "text": str(task_prompt)})
            encoded = self.qwen_processor.build_batch_inputs([[{"role": "user", "content": content}]])
            input_ids = encoded["input_ids"].cpu().numpy().astype(np.int32)
            attention_mask = encoded["attention_mask"].cpu().numpy().astype(np.int32)
            image_grid_thw = encoded["image_grid_thw"].cpu().numpy().astype(np.int32)
            pixel_values = encoded["pixel_values"].cpu().numpy().astype(np.float32)
            if self.qwen_processor.image_token_id is None:
                raise ValueError("Unable to determine Qwen image_token_id for pure Qwen3-VL observation processing.")
            image_token_mask = input_ids == int(self.qwen_processor.image_token_id)
            from vla_tpu.models.qwen3_vl_jax import get_rope_index

            position_ids = np.asarray(
                get_rope_index(
                    jnp.asarray(input_ids),
                    jnp.asarray(image_token_mask.astype(np.int32)),
                    jnp.asarray(image_grid_thw),
                    jnp.asarray(attention_mask),
                    self.backbone_cfg.vision_spatial_merge_size,
                )
            ).astype(np.int32)
            return {
                "images": images_np[None, ...],
                "state": state[None, ...],
                "state_mask": np.ones_like(state[None, ...], dtype=bool),
                "action_query": action_query[None, ...],
                "action_mask": np.ones_like(action_query[None, ...], dtype=np.float32),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": image_token_mask.astype(np.int32),
                "image_token_mask": image_token_mask,
                "image_token_id": np.asarray(self.qwen_processor.image_token_id, dtype=np.int32),
                "position_ids": position_ids,
            }

        instruction_tokens = _tokenize_instruction(
            str(task_prompt),
            vocab_size=self.backbone_cfg.text_vocab_size,
            length=self.data_cfg.instruction_length,
        )
        return {
            "images": images_np[None, ...],
            "state": state[None, ...],
            "state_mask": np.ones_like(state[None, ...], dtype=bool),
            "instruction_tokens": instruction_tokens[None, ...],
            "action_query": action_query[None, ...],
            "action_mask": np.ones_like(action_query[None, ...], dtype=np.float32),
        }

    def unprocess_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2:
            raise ValueError(f"Expected action shape (H, D), got {action.shape}")

        cursor = 0
        outputs: dict[str, np.ndarray] = {}
        for key, dim, stats in self._action_layout:
            chunk = action[:, cursor : cursor + dim]
            cursor += dim
            mode = self.data_cfg.action_normalization_modes.get(key)
            if mode is not None and stats is not None:
                chunk = _denormalize(chunk, mode, stats)
            outputs[key] = chunk.astype(np.float32)

        if cursor != action.shape[-1]:
            raise ValueError(f"Action layout consumed {cursor} dims but action has {action.shape[-1]}")
        return outputs
