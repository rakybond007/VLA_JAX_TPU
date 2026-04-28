from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_tpu.configs.base import ExperimentConfig

try:
    import cv2
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency path
    cv2 = None
    pd = None


@dataclass(frozen=True)
class EpisodeRef:
    episode_index: int
    task: str
    length: int


def _require_robocasa_deps() -> None:
    if cv2 is None or pd is None:
        raise ImportError(
            "RoboCasa LeRobot loading requires optional deps. "
            "Install with `pip install -e .[robocasa-data]`."
        )


def _tokenize_instruction(text: str, vocab_size: int, length: int) -> np.ndarray:
    tokens = np.zeros((length,), dtype=np.int32)
    for idx, piece in enumerate(text.lower().split()[:length]):
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        tokens[idx] = int.from_bytes(digest[:4], "little") % vocab_size
    return tokens


class RoboCasaLeRobotDataset:
    def __init__(self, config: ExperimentConfig):
        _require_robocasa_deps()

        self.config = config
        self.data_cfg = config.data
        self.action_cfg = config.action_head
        self.backbone_cfg = config.backbone
        self.dataset_root = Path(self.data_cfg.dataset_root).expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.info = json.loads((self.dataset_root / "meta/info.json").read_text())
        self.episodes = self._load_episodes()[: self.data_cfg.max_episodes]
        self.sample_refs = self._build_sample_refs(self.episodes)
        self._episode_cache: dict[int, Any] = {}

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

    def _build_sample_refs(self, episodes: list[EpisodeRef]) -> list[tuple[int, int]]:
        refs: list[tuple[int, int]] = []
        horizon = self.action_cfg.action_horizon
        stride = max(1, self.data_cfg.episode_stride)

        for episode in episodes:
            max_start = max(0, episode.length - horizon)
            for start in range(0, max_start + 1, stride):
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
        return self.dataset_root / self.info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=camera_key,
        )

    def _load_episode_df(self, episode_index: int):
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            return cached

        df = pd.read_parquet(self._episode_path(episode_index))
        self._episode_cache = {episode_index: df}
        return df

    def _read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (self.data_cfg.image_width, self.data_cfg.image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return frame.astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        episode_index, start_index = self.sample_refs[index]
        episode = next(ep for ep in self.episodes if ep.episode_index == episode_index)
        df = self._load_episode_df(episode_index)

        state = np.asarray(df.iloc[start_index]["observation.state"], dtype=np.float32)
        actions = np.stack(
            [
                np.asarray(df.iloc[start_index + step]["action"], dtype=np.float32)
                for step in range(self.action_cfg.action_horizon)
            ],
            axis=0,
        )

        images = []
        for camera_key in self.data_cfg.camera_keys[: self.data_cfg.num_cameras]:
            images.append(self._read_frame(self._video_path(episode_index, camera_key), start_index))
        images_np = np.stack(images, axis=0)

        instruction_tokens = _tokenize_instruction(
            episode.task,
            vocab_size=self.backbone_cfg.text_vocab_size,
            length=self.data_cfg.instruction_length,
        )

        action_query = np.zeros_like(actions, dtype=np.float32)
        return {
            "images": images_np,
            "state": state,
            "instruction_tokens": instruction_tokens,
            "actions": actions,
            "action_query": action_query,
        }


def make_robocasa_batch(config: ExperimentConfig, batch_size: int) -> dict[str, np.ndarray]:
    dataset = RoboCasaLeRobotDataset(config)
    batch_items = [dataset[idx % len(dataset)] for idx in range(batch_size)]
    keys = batch_items[0].keys()
    return {
        key: np.stack([item[key] for item in batch_items], axis=0)
        for key in keys
    }
