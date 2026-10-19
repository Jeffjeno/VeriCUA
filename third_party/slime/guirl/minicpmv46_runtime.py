from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
from pathlib import Path

from packaging.version import Version

REQUIRED_TRANSFORMERS_VERSION = Version("5.12.1")
DEFAULT_SOURCE_SITE = Path(
    "./runtime_packages"
)
DEFAULT_RUNTIME_DIR = Path(
    "./outputs/transformers-minicpmv46"
)

def _distribution_dir(site_dir: Path, version: str) -> Path:
    matches = sorted(site_dir.glob(f"transformers-{version}.dist-info"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one transformers-{version}.dist-info under {site_dir}, "
            f"found {matches}"
        )
    return matches[0]

def prepare(source_site: Path, runtime_dir: Path) -> None:

    source_pkg = source_site / "transformers"
    if not source_pkg.is_dir():
        raise RuntimeError(f"Transformers source package not found: {source_pkg}")

    source_version_ns: dict[str, str] = {}
    version_line = next(
        (
            line
            for line in (source_pkg / "__init__.py").read_text().splitlines()
            if line.startswith("__version__")
        ),
        None,
    )
    if version_line is None:
        raise RuntimeError(f"cannot determine Transformers version from {source_pkg}")
    exec(version_line, {}, source_version_ns)
    source_version = Version(source_version_ns["__version__"])
    if source_version != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"MiniCPM-V 4.6 runtime requires Transformers "
            f"{REQUIRED_TRANSFORMERS_VERSION}, found {source_version}"
        )
    if not (source_pkg / "models" / "minicpmv4_6").is_dir():
        raise RuntimeError(f"{source_pkg} has no MiniCPM-V 4.6 implementation")

    source_dist = _distribution_dir(source_site, str(source_version))
    marker = runtime_dir / ".source-version"
    if (
        marker.is_file()
        and marker.read_text().strip() == str(source_version)
        and (runtime_dir / "transformers" / "models" / "minicpmv4_6").is_dir()
    ):
        print(f"Transformers overlay already ready: {runtime_dir} ({source_version})")
        return

    staging = runtime_dir.with_name(f"{runtime_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copytree(source_pkg, staging / "transformers", symlinks=True)
    shutil.copytree(source_dist, staging / source_dist.name, symlinks=True)
    (staging / ".source-version").write_text(f"{source_version}\n")

    old = runtime_dir.with_name(f"{runtime_dir.name}.old-{os.getpid()}")
    if runtime_dir.exists():
        runtime_dir.rename(old)
    staging.rename(runtime_dir)
    shutil.rmtree(old, ignore_errors=True)
    print(f"Prepared Transformers {source_version} overlay: {runtime_dir}")

def prepare_sglang_model_view(
    checkpoint: Path,
    model_dir: Path,
    max_slice_nums: int | None = None,
) -> None:

    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "minicpmv4_6":
        raise ValueError(f"unexpected model_type in {config_path}")
    config.pop("hidden_size", None)

    if max_slice_nums is not None and max_slice_nums <= 0:
        raise ValueError("max_slice_nums must be positive")

    staging = model_dir.with_name(f"{model_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for source in checkpoint.iterdir():
        target = staging / source.name
        if source.name == "config.json":
            target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        elif source.name == "processor_config.json" and max_slice_nums is not None:
            processor_config = json.loads(source.read_text())
            for component_name in ("image_processor", "video_processor"):
                component = processor_config.get(component_name)
                if isinstance(component, dict):
                    component["max_slice_nums"] = max_slice_nums
            target.write_text(
                json.dumps(processor_config, indent=2, sort_keys=True) + "\n"
            )
        elif source.name == "preprocessor_config.json" and max_slice_nums is not None:
            preprocessor_config = json.loads(source.read_text())
            preprocessor_config["max_slice_nums"] = max_slice_nums
            target.write_text(
                json.dumps(preprocessor_config, indent=2, sort_keys=True) + "\n"
            )
        else:
            target.symlink_to(source)
    (staging / ".source-checkpoint").write_text(f"{checkpoint}\n")
    if max_slice_nums is not None:
        (staging / ".max-slice-nums").write_text(f"{max_slice_nums}\n")

    old = model_dir.with_name(f"{model_dir.name}.old-{os.getpid()}")
    if model_dir.exists() or model_dir.is_symlink():
        model_dir.rename(old)
    staging.rename(model_dir)
    shutil.rmtree(old, ignore_errors=True)
    print(f"Prepared SGLang model view: {model_dir} -> {checkpoint}")

def _assert_runtime_overlay(runtime_dir: Path) -> None:
    imported = Path(importlib.import_module("transformers").__file__).resolve()
    expected = (runtime_dir / "transformers").resolve()
    if expected not in imported.parents:
        raise RuntimeError(
            f"Transformers was imported from {imported}, not runtime overlay {expected}; "
            "prepend the overlay directory to PYTHONPATH before starting Python"
        )

def preflight(
    checkpoint: Path,
    runtime_dir: Path,
    require_sglang: bool,
    expected_max_slice_nums: int | None = None,
) -> None:

    import transformers
    from transformers import AutoConfig
    from transformers.models.minicpmv4_6 import (
        MiniCPMV4_6Config,
        MiniCPMV4_6ForConditionalGeneration,
        MiniCPMV4_6Processor,
    )

    version = Version(transformers.__version__)
    if version != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"MiniCPM-V 4.6 runtime requires Transformers "
            f"{REQUIRED_TRANSFORMERS_VERSION}, found {version}"
        )
    if runtime_dir.exists():
        _assert_runtime_overlay(runtime_dir)

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    from slime.utils.processing_utils import load_processor

    processor = load_processor(checkpoint, trust_remote_code=True)
    if not isinstance(config, MiniCPMV4_6Config):
        raise TypeError(f"unexpected checkpoint config: {type(config)!r}")
    if not isinstance(processor, MiniCPMV4_6Processor):
        raise TypeError(f"unexpected checkpoint processor: {type(processor)!r}")
    if config.architectures != ["MiniCPMV4_6ForConditionalGeneration"]:
        raise ValueError(f"unexpected model architecture: {config.architectures}")
    if config.image_token_id != processor.image_token_id:
        raise ValueError(
            f"image token mismatch: config={config.image_token_id}, "
            f"processor={processor.image_token_id}"
        )
    if expected_max_slice_nums is not None:
        actual_max_slice_nums = processor.image_processor.max_slice_nums
        if actual_max_slice_nums != expected_max_slice_nums:
            raise ValueError(
                "MiniCPM-V 4.6 processor max_slice_nums mismatch: "
                f"expected {expected_max_slice_nums}, got {actual_max_slice_nums}"
            )

    if require_sglang:
        sglang_module = importlib.import_module("sglang.srt.models.minicpmv")
        sglang_cls = getattr(
            sglang_module, "MiniCPMV4_6ForConditionalGeneration", None
        )
        if sglang_cls is None:
            raise RuntimeError("installed SGLang has no MiniCPM-V 4.6 model class")
        importlib.import_module("sglang.srt.managers.schedule_batch")

    print(
        "MiniCPM-V 4.6 preflight OK: "
        f"transformers={transformers.__version__}, "
        f"config={type(config).__name__}, processor={type(processor).__name__}, "
        f"hf_model={MiniCPMV4_6ForConditionalGeneration.__name__}, "
        f"image_token_id={config.image_token_id}, "
        f"max_slice_nums={processor.image_processor.max_slice_nums}, "
        f"sglang={require_sglang}"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-site", type=Path, default=DEFAULT_SOURCE_SITE
    )
    prepare_parser.add_argument(
        "--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR
    )

    sglang_view_parser = subparsers.add_parser("prepare-sglang-model-view")
    sglang_view_parser.add_argument("--checkpoint", type=Path, required=True)
    sglang_view_parser.add_argument("--model-dir", type=Path, required=True)
    sglang_view_parser.add_argument("--max-slice-nums", type=int)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--checkpoint", type=Path, required=True)
    preflight_parser.add_argument(
        "--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR
    )
    preflight_parser.add_argument(
        "--skip-sglang", action="store_true", help="only check Transformers"
    )
    preflight_parser.add_argument("--max-slice-nums", type=int)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.source_site, args.runtime_dir)
    elif args.command == "prepare-sglang-model-view":
        prepare_sglang_model_view(
            args.checkpoint,
            args.model_dir,
            args.max_slice_nums,
        )
    else:
        preflight(
            args.checkpoint,
            args.runtime_dir,
            not args.skip_sglang,
            args.max_slice_nums,
        )

if __name__ == "__main__":
    main()
