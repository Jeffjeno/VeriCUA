from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .master_client import BatchSet, TaskRecord

DEFAULT_EXAMPLES_DIR = Path("evaluation_examples/examples")
DEFAULT_MANIFEST_PATH = Path("evaluation_examples/service_tasks.json")

@dataclass(frozen=True)
class OSWorldTaskSpec:
    app: str
    task_id: str
    path: Path
    config: dict[str, Any]

    @property
    def prompt(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "app": self.app,
            "instruction": self.config.get("instruction", ""),
            "snapshot": self.config.get("snapshot"),
        }

class OSWorldTaskSampler:
    def __init__(
        self,
        *,
        examples_dir: str | Path = DEFAULT_EXAMPLES_DIR,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        cycle: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.examples_dir = Path(examples_dir)
        self.manifest_path = Path(manifest_path)
        self.cycle = cycle
        self.shuffle = shuffle
        self.seed = seed
        self.tasks = self._load_tasks()
        if not self.tasks:
            raise ValueError(f"no OSWorld tasks loaded from {self.manifest_path}")
        self.epoch_id = 0
        self.sample_offset = 0
        self.samples = list(self.tasks)
        if self.shuffle:
            self._shuffle(self.epoch_id)

        self._stream_step_id = 0

    async def sample_batch(self, step_id: int, batch_size: int) -> BatchSet:

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        expected_offset = step_id * batch_size
        self._sync_cursor(expected_offset)
        batch_epoch_id = self.epoch_id
        batch_sample_offset = self.sample_offset
        specs = self._take_batch(batch_size)
        return self._build_batchset(
            step_id=step_id,
            batch_id=f"step-{step_id}-osworld",
            batch_epoch_id=batch_epoch_id,
            batch_sample_offset=batch_sample_offset,
            specs=specs,
        )

    async def sample_next(self, count: int) -> BatchSet:

        if count <= 0:
            raise ValueError("count must be positive")
        batch_epoch_id = self.epoch_id
        batch_sample_offset = self.sample_offset
        specs = self._take_batch(count)
        step_id = self._stream_step_id
        self._stream_step_id += 1
        return self._build_batchset(
            step_id=step_id,
            batch_id=f"stream-{step_id}-osworld",
            batch_epoch_id=batch_epoch_id,
            batch_sample_offset=batch_sample_offset,
            specs=specs,
        )

    def _build_batchset(
        self,
        *,
        step_id: int,
        batch_id: str,
        batch_epoch_id: int,
        batch_sample_offset: int,
        specs: list[OSWorldTaskSpec],
    ) -> BatchSet:
        records = [
            TaskRecord(
                task_id=spec.task_id,
                step_id=step_id,
                batch_id=batch_id,
                sample_index=index,
                prompt=spec.prompt,
                task_config=dict(spec.config),
                metadata={"app": spec.app, "task_path": str(spec.path)},
            )
            for index, spec in enumerate(specs)
        ]
        return BatchSet(
            step_id=step_id,
            batch_id=batch_id,
            tasks=records,
            metadata={
                "sampler": "osworld_file",
                "examples_dir": str(self.examples_dir),
                "manifest_path": str(self.manifest_path),
                "cycle": self.cycle,
                "shuffle": self.shuffle,
                "seed": self.seed,
                "epoch_id": batch_epoch_id,
                "data_source_cursor": batch_sample_offset,
                "next_epoch_id": self.epoch_id,
                "next_data_source_cursor": self.sample_offset,
                "task_ids": [spec.task_id for spec in specs],
            },
        )

    def _sync_cursor(self, expected_offset: int) -> None:
        if expected_offset == self.epoch_id * len(self.tasks) + self.sample_offset:
            return
        if expected_offset < 0:
            raise ValueError("expected_offset must be non-negative")
        if not self.cycle and expected_offset >= len(self.tasks):
            raise IndexError(f"task index {expected_offset} exceeds loaded task count {len(self.tasks)}")
        self.epoch_id = expected_offset // len(self.tasks)
        self.sample_offset = expected_offset % len(self.tasks)
        if self.shuffle:
            self._shuffle(self.epoch_id)
        else:
            self.samples = list(self.tasks)

    def _take_batch(self, batch_size: int) -> list[OSWorldTaskSpec]:
        if self.sample_offset + batch_size <= len(self.samples):
            specs = self.samples[self.sample_offset : self.sample_offset + batch_size]
            self.sample_offset += batch_size
            return specs

        specs = self.samples[self.sample_offset :]
        remaining = batch_size - len(specs)
        while remaining > 0:
            self.epoch_id += 1
            if not self.cycle:
                raise IndexError(f"task batch exceeds loaded task count {len(self.tasks)}")
            if self.shuffle:
                self._shuffle(self.epoch_id)
            else:
                self.samples = list(self.tasks)
            take = min(remaining, len(self.samples))
            specs += self.samples[:take]
            remaining -= take
        self.sample_offset = take
        return specs

    def _shuffle(self, epoch_id: int) -> None:
        rng = random.Random(self.seed + epoch_id)
        permutation = list(range(len(self.tasks)))
        rng.shuffle(permutation)
        self.samples = [self.tasks[index] for index in permutation]

    def _load_tasks(self) -> list[OSWorldTaskSpec]:
        manifest = self._load_manifest()
        tasks: list[OSWorldTaskSpec] = []
        for app, task_ids in manifest.items():
            if not isinstance(task_ids, list):
                raise ValueError(f"manifest entry for {app!r} must be a list")
            for task_id in task_ids:
                task_path = self.examples_dir / app / f"{task_id}.json"
                if not task_path.exists():
                    raise FileNotFoundError(f"task config not found: {task_path}")
                with task_path.open("r", encoding="utf-8") as file:
                    config = json.load(file)
                if not isinstance(config, dict):
                    raise ValueError(f"task config must be a JSON object: {task_path}")
                tasks.append(OSWorldTaskSpec(app=app, task_id=str(task_id), path=task_path, config=config))
        return tasks

    def _load_manifest(self) -> dict[str, list[str]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest must be a JSON object: {self.manifest_path}")
        return manifest
