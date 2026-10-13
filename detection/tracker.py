from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .person_detector import PersonDetection


@dataclass
class PersonTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    missed: int = 0
    age: int = 1


class PersonTracker:
    """Lightweight IoU tracker used to keep one ID per visible person."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 8):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: list[PersonTrack] = []

    @staticmethod
    def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def update(self, detections: list[PersonDetection]) -> list[PersonDetection]:
        unmatched = set(range(len(self.tracks)))
        matched = set()
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                pairs.append((self.iou(track.bbox, tuple(det.bbox)), ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            if score < self.iou_threshold or ti not in unmatched or di in matched:
                continue
            track = self.tracks[ti]
            det = detections[di]
            track.bbox = tuple(det.bbox)
            track.confidence = det.confidence
            track.missed = 0
            track.age += 1
            det.track_id = track.track_id
            unmatched.remove(ti)
            matched.add(di)

        for ti in unmatched:
            self.tracks[ti].missed += 1

        for di, det in enumerate(detections):
            if di in matched:
                continue
            track = PersonTrack(self.next_id, tuple(det.bbox), det.confidence)
            self.next_id += 1
            self.tracks.append(track)
            det.track_id = track.track_id

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return detections

    def active(self) -> list[PersonTrack]:
        return list(self.tracks)

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1

    def status(self) -> dict[str, Any]:
        return {
            "active_tracks": len(self.tracks),
            "next_track_id": self.next_id,
            "tracks": [
                {"track_id": t.track_id, "bbox": list(t.bbox), "confidence": round(t.confidence, 4), "missed": t.missed, "age": t.age}
                for t in self.tracks
            ],
        }
