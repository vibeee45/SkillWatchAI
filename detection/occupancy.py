from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .person_detector import PersonDetection
from .tracker import PersonTracker


@dataclass
class Zone:
    zone_id: str
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0

    def contains_center(self, bbox: list[int], width: int, height: int) -> bool:
        x, y, w, h = bbox
        cx = (x + w / 2) / max(width, 1)
        cy = (y + h / 2) / max(height, 1)
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def as_dict(self) -> dict[str, Any]:
        return {"zone_id": self.zone_id, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


class OccupancyService:
    """Phase 8 occupancy fusion: detection + tracking + zones + smoothing."""

    def __init__(self, history_size: int = 5):
        self.tracker = PersonTracker()
        self.history = deque(maxlen=max(1, int(history_size)))
        self.zones: list[Zone] = [Zone("classroom", 0.0, 0.0, 1.0, 1.0)]
        self.last_result: dict[str, Any] = {
            "total_occupancy": 0,
            "recognized_students": 0,
            "unknown_persons": 0,
            "raw_occupancy": 0,
            "smoothed_occupancy": 0,
            "zones": {},
            "detections": [],
        }

    def set_zones(self, zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed = []
        for item in zones:
            parsed.append(Zone(
                zone_id=str(item.get("zone_id", f"zone-{len(parsed)+1}")),
                x1=float(item.get("x1", 0.0)), y1=float(item.get("y1", 0.0)),
                x2=float(item.get("x2", 1.0)), y2=float(item.get("y2", 1.0)),
            ))
        self.zones = parsed or [Zone("classroom")]
        return [z.as_dict() for z in self.zones]

    def process(
        self,
        detections: list[PersonDetection],
        recognition_results: list[dict[str, Any]],
        face_detections: list[Any],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        self.tracker.update(detections)
        recognized_track_ids: set[int] = set()
        for result, face in zip(recognition_results, face_detections):
            if result.get("status") == "confirmed" and result.get("student_id") and face.track_id is not None:
                center = (face.x + face.width / 2, face.y + face.height / 2)
                best_track = None
                best_iou = 0.0
                for person in detections:
                    if person.track_id is None:
                        continue
                    px, py, pw, ph = person.bbox
                    if px <= center[0] <= px + pw and py <= center[1] <= py + ph:
                        best_track = person.track_id
                        break
                if best_track is not None:
                    recognized_track_ids.add(best_track)

        zone_counts: dict[str, int] = {z.zone_id: 0 for z in self.zones}
        for det in detections:
            for zone in self.zones:
                if zone.contains_center(det.bbox, frame_width, frame_height):
                    det.zone = zone.zone_id
                    zone_counts[zone.zone_id] += 1
                    break

        raw = len(detections)
        self.history.append(raw)
        smoothed = int(round(float(np.median(list(self.history))))) if self.history else 0
        recognized = min(len(recognized_track_ids), raw)
        unknown = max(0, smoothed - recognized)

        result = {
            "model": "OpenCV HOG + IoU tracker",
            "total_occupancy": smoothed,
            "raw_occupancy": raw,
            "smoothed_occupancy": smoothed,
            "recognized_students": recognized,
            "unknown_persons": unknown,
            "active_tracks": len(self.tracker.active()),
            "zones": zone_counts,
            "zone_config": [z.as_dict() for z in self.zones],
            "detections": [d.as_dict() for d in detections],
            "history": list(self.history),
        }
        self.last_result = result
        return result

    def status(self) -> dict[str, Any]:
        return {**self.last_result, "tracker": self.tracker.status(), "zone_config": [z.as_dict() for z in self.zones]}

    def reset(self) -> None:
        self.tracker.reset()
        self.history.clear()
        self.last_result = {"total_occupancy": 0, "recognized_students": 0, "unknown_persons": 0, "raw_occupancy": 0, "smoothed_occupancy": 0, "zones": {}, "detections": []}
