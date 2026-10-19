from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

@dataclass(frozen=True)
class WebArenaDataset:
    name: str
    config_path: Path
    manifest_path: Path
    records: tuple[dict[str, Any], ...]

def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def load_webarena_dataset(
    name: str = "webarena", *, project_root: Path | None = None
) -> WebArenaDataset:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    config_path = root / "config" / "eval_datasets" / f"{name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"WebArena dataset config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if str(config.get("benchmark") or "") != "webarena_verified":
        raise ValueError(f"{config_path} is not a WebArena-Verified dataset")
    manifest_path = _resolve(root, str((config.get("paths") or {}).get("manifest") or ""))
    exclusions = config.get("exclusions") or {}
    excluded_candidate_ids = {
        str(candidate_id)
        for candidate_id in (exclusions.get("candidate_ids") or [])
        if str(candidate_id).strip()
    }
    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Invalid manifest row {line_number}: expected object")
            if str(record.get("candidate_id") or "") in excluded_candidate_ids:
                continue
            records.append(record)
    if not records:
        raise ValueError(f"WebArena dataset manifest is empty: {manifest_path}")
    return WebArenaDataset(
        name=str(config.get("name") or name),
        config_path=config_path,
        manifest_path=manifest_path,
        records=tuple(records),
    )
