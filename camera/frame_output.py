from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class FramePacket:
    """Unified frame object emitted by every camera source."""
    frame: np.ndarray
    camera_id: Optional[str]
    room: Optional[str]
    source_type: Optional[str]
    timestamp: str
    frame_number: int
    width: int
    height: int
    fps: float

    def as_metadata(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "room": self.room,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
