from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .batch_models import ArtifactRef, ErrorCode, ProtocolError

class ArtifactStore:

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def write_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        payload = bytes(data)
        digest = hashlib.sha256(payload).hexdigest()
        target = self._digest_path(digest)
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                ref = self._make_ref(target, digest, len(payload), media_type)
                if not self.verify(ref):
                    raise ProtocolError(
                        ErrorCode.ARTIFACT_CORRUPT,
                        "existing artifact does not match its digest",
                        {"digest": digest, "uri": ref.uri},
                    )
                return ref

            fd, temp_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, target)
                self._fsync_directory(target.parent)
            except BaseException:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
                raise
        return self._make_ref(target, digest, len(payload), media_type)

    put_bytes = write_bytes

    def write_json(self, value: Any) -> ArtifactRef:
        try:
            payload = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "artifact JSON is not serializable",
                {"error": str(exc)},
            ) from exc
        return self.write_bytes(payload, media_type="application/json")

    put_json = write_json

    def read(self, artifact: ArtifactRef | str) -> bytes:
        ref, path = self._resolve(artifact)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise ProtocolError(
                ErrorCode.NOT_FOUND,
                "artifact does not exist",
                {"uri": path.as_uri()},
            ) from exc
        actual = hashlib.sha256(payload).hexdigest()
        if ref is not None and (actual != ref.digest or len(payload) != ref.size):
            raise ProtocolError(
                ErrorCode.ARTIFACT_CORRUPT,
                "artifact digest or size mismatch",
                {
                    "expected_digest": ref.digest,
                    "actual_digest": actual,
                    "expected_size": ref.size,
                    "actual_size": len(payload),
                },
            )
        if ref is None and len(path.stem) == 64 and actual != path.stem:
            raise ProtocolError(
                ErrorCode.ARTIFACT_CORRUPT,
                "artifact content does not match its content-addressed path",
                {"expected_digest": path.stem, "actual_digest": actual},
            )
        return payload

    read_bytes = read

    def read_json(self, artifact: ArtifactRef | str) -> Any:
        try:
            return json.loads(self.read(artifact).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(
                ErrorCode.ARTIFACT_CORRUPT,
                "artifact is not valid UTF-8 JSON",
                {"error": str(exc)},
            ) from exc

    def verify(self, artifact: ArtifactRef) -> bool:
        try:
            path = self._path_from_uri(artifact.uri)
            stat = path.stat()
            if stat.st_size != artifact.size:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == artifact.digest
        except (FileNotFoundError, OSError, ProtocolError):
            return False

    def delete(self, artifact: ArtifactRef | str) -> bool:
        _, path = self._resolve(artifact)
        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            self._fsync_directory(path.parent)
            return True

    def _resolve(self, artifact: ArtifactRef | str) -> tuple[ArtifactRef | None, Path]:
        if isinstance(artifact, ArtifactRef):
            return artifact, self._path_from_uri(artifact.uri)
        if len(artifact) == 64 and all(char in "0123456789abcdef" for char in artifact.lower()):
            return None, self._digest_path(artifact.lower())
        return None, self._path_from_uri(artifact)

    def _digest_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "invalid SHA-256 digest",
                {"digest": digest},
            )
        return self.root / digest[:2] / digest[2:4] / f"{digest}.blob"

    def _path_from_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme and parsed.scheme != "file":
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "artifact URI must use the file scheme",
                {"uri": uri},
            )
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ProtocolError(
                ErrorCode.VALIDATION_ERROR,
                "artifact URI is outside the configured spool",
                {"uri": uri},
            ) from exc
        return path

    @staticmethod
    def _make_ref(path: Path, digest: str, size: int, media_type: str) -> ArtifactRef:
        return ArtifactRef(digest=digest, size=size, uri=path.as_uri(), media_type=media_type)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
