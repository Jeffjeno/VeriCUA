from __future__ import annotations

import importlib.util
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_VENDORED = Path(__file__).parent / "minicpmv.py"

def _installed_minicpmv_path() -> Path | None:
    spec = importlib.util.find_spec("sglang.srt.models.minicpmv")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve()

def ensure_minicpmv_support(force: bool = False) -> bool:

    target = _installed_minicpmv_path()
    if target is None:
        logger.warning("sglang not installed or has no minicpmv module; skip patch")
        return False

    current = target.read_text(encoding="utf-8")
    if not force and "class MiniCPMV4_0" in current:
        logger.info("sglang already supports MiniCPM-V 4.0 (%s)", target)
        return True

    vendored = _VENDORED.read_text(encoding="utf-8")
    if current == vendored:
        return True
    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(vendored, encoding="utf-8")
    logger.info("Patched %s with vendored MiniCPM-V 4.x implementation", target)
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_minicpmv_support()
