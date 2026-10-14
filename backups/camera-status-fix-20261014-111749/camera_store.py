import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class CameraStore:
    """Small JSON-backed camera configuration store for Phase 2."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, cameras: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cameras, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        with self._lock:
            return self._read()

    def get(self, camera_id: str) -> Optional[dict]:
        return next((c for c in self.list() if c.get("camera_id") == camera_id), None)

    def upsert(self, config: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cameras = self._read()
            existing = next((c for c in cameras if c.get("camera_id") == config["camera_id"]), None)
            if existing:
                created_at = existing.get("created_at", now)
                existing.update(config)
                existing["created_at"] = created_at
                existing["updated_at"] = now
                result = existing.copy()
            else:
                result = {**config, "created_at": now, "updated_at": now}
                cameras.append(result)
            self._write(cameras)
            return result

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            cameras = self._read()
            updated = [c for c in cameras if c.get("camera_id") != camera_id]
            if len(updated) == len(cameras):
                return False
            self._write(updated)
            return True
