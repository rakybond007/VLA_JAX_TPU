from __future__ import annotations

import copy
import queue
import threading

import jax.numpy as jnp

from vla_tpu.configs.base import ExperimentConfig
from vla_tpu.data.dummy import make_dummy_batch
from vla_tpu.data.lerobot_dataset import LeRobotDataset
from vla_tpu.data.robocasa_lerobot import make_robocasa_batch


class BatchProvider:
    def next_batch(self, batch_size: int):
        raise NotImplementedError


class DummyBatchProvider(BatchProvider):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def next_batch(self, batch_size: int):
        return make_dummy_batch(self.config, batch_size=batch_size)


class RoboCasaBatchProvider(BatchProvider):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def next_batch(self, batch_size: int):
        batch = make_robocasa_batch(self.config, batch_size=batch_size)
        return {key: jnp.asarray(value) for key, value in batch.items()}


class LeRobotBatchProvider(BatchProvider):
    def __init__(self, config: ExperimentConfig, train: bool):
        self.dataset = LeRobotDataset(config)
        self.train = train

    def next_batch(self, batch_size: int):
        batch = self.dataset.sample_batch(batch_size=batch_size, train=self.train)
        return {key: jnp.asarray(value) for key, value in batch.items()}


class PrefetchLeRobotBatchProvider(BatchProvider):
    def __init__(self, config: ExperimentConfig, train: bool):
        self.config = config
        self.train = train
        self.batch_size = None
        self.queue = queue.Queue(maxsize=max(1, config.data.prefetch_size))
        self._started = False
        self._error = None
        self._lock = threading.Lock()

    def _worker_loop(self, worker_idx: int):
        worker_config = copy.deepcopy(self.config)
        worker_config.data.seed = int(self.config.data.seed + worker_idx + 1)
        dataset = LeRobotDataset(worker_config)
        while True:
            try:
                batch = dataset.sample_batch(batch_size=self.batch_size, train=self.train)
                self.queue.put(batch)
            except Exception as exc:  # pragma: no cover - worker failures surface in next_batch
                self._error = exc
                break

    def _ensure_started(self, batch_size: int):
        with self._lock:
            if self._started:
                if batch_size != self.batch_size:
                    raise ValueError(
                        f"PrefetchLeRobotBatchProvider already started with batch_size={self.batch_size}, "
                        f"got batch_size={batch_size}"
                    )
                return
            self.batch_size = batch_size
            num_workers = max(1, self.config.data.num_workers)
            for worker_idx in range(num_workers):
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_idx,),
                    name=f"lerobot-prefetch-{worker_idx}",
                    daemon=True,
                )
                thread.start()
            self._started = True

    def next_batch(self, batch_size: int):
        self._ensure_started(batch_size)
        if self._error is not None:
            raise RuntimeError("PrefetchLeRobotBatchProvider worker failed") from self._error
        batch = self.queue.get()
        if self._error is not None:
            raise RuntimeError("PrefetchLeRobotBatchProvider worker failed") from self._error
        return {key: jnp.asarray(value) for key, value in batch.items()}


def make_batch_provider(config: ExperimentConfig, train: bool) -> BatchProvider:
    if config.data.dataset_type == "dummy":
        return DummyBatchProvider(config)
    if config.data.dataset_type == "robocasa_lerobot":
        return RoboCasaBatchProvider(config)
    if config.data.dataset_type == "lerobot":
        if train and config.data.num_workers > 0:
            return PrefetchLeRobotBatchProvider(config, train=train)
        return LeRobotBatchProvider(config, train=train)
    raise ValueError(f"Unsupported dataset_type: {config.data.dataset_type}")


def make_batch(config: ExperimentConfig, batch_size: int, train: bool = False):
    return make_batch_provider(config, train=train).next_batch(batch_size=batch_size)
