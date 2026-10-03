from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet


class ProtectedRepresentationStore:
    """Encrypts face representations at rest; raw face samples are not stored."""

    def __init__(self, key_path: str | Path):
        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()
        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        try:
            os.chmod(self.key_path, 0o600)
        except OSError:
            pass
        return key

    def protect(self, vector: np.ndarray) -> str:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        payload = array.tobytes()
        encrypted = self.cipher.encrypt(payload)
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def reveal(self, token: str, dimension: int | None = None) -> np.ndarray:
        encrypted = base64.urlsafe_b64decode(token.encode("ascii"))
        raw = self.cipher.decrypt(encrypted)
        vector = np.frombuffer(raw, dtype=np.float32).copy()
        if dimension is not None and vector.size != dimension:
            raise ValueError("Protected representation dimension mismatch.")
        return vector
