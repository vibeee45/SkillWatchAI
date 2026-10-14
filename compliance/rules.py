"""Observation-only equipment reporting for the SkillWatchAI prototype.

This module deliberately does not define expected quantities, inventory,
shortage/compliance states, or equipment-count alerts. Every count reported
here is derived from the current valid detector observation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from detection.infrastructure import EquipmentDetection


class InfrastructureObservationService:
    """Summarize detected equipment by class and configured image zone."""

    def __init__(self):
        self.zones: list[dict[str, Any]] = [
            {"zone_id": "classroom", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        ]
        self.last_result: dict[str, Any] = {
            "detection_valid": False,
            "detection_state": "NOT_RUN",
            "counts_by_class": {},
            "counts_by_zone": {"classroom": {}},
            "detections": [],
        }

    def set_zones(self, zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(zones, list):
            raise ValueError("Zones must be a list.")
        parsed = []
        seen = set()
        for i, item in enumerate(zones):
            if not isinstance(item, dict):
                raise ValueError("Each zone must be an object.")
            zone = {
                "zone_id": str(item.get("zone_id", f"zone-{i + 1}")).strip(),
                "x1": float(item.get("x1", 0.0)),
                "y1": float(item.get("y1", 0.0)),
                "x2": float(item.get("x2", 1.0)),
                "y2": float(item.get("y2", 1.0)),
            }
            if not zone["zone_id"]:
                raise ValueError("Zone ID cannot be empty.")
            if zone["zone_id"] in seen:
                raise ValueError(f"Duplicate zone ID: {zone['zone_id']}")
            if not (0 <= zone["x1"] < zone["x2"] <= 1 and 0 <= zone["y1"] < zone["y2"] <= 1):
                raise ValueError(f"Invalid normalized coordinates for zone {zone['zone_id']}.")
            seen.add(zone["zone_id"])
            parsed.append(zone)
        self.zones = parsed or [{"zone_id": "classroom", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}]
        return self.get_zones()

    def get_zones(self) -> list[dict[str, Any]]:
        return [dict(zone) for zone in self.zones]

    @staticmethod
    def _in_zone(detection: EquipmentDetection, zone: dict[str, Any], frame_width: int, frame_height: int) -> bool:
        x, y, w, h = detection.bbox
        cx, cy = (x + w / 2) / max(frame_width, 1), (y + h / 2) / max(frame_height, 1)
        return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]

    def process(self, detections: list[EquipmentDetection], frame_width: int, frame_height: int) -> dict[str, Any]:
        counts_by_class: dict[str, int] = defaultdict(int)
        counts_by_zone: dict[str, dict[str, int]] = {zone["zone_id"]: defaultdict(int) for zone in self.zones}
        for detection in detections:
            counts_by_class[detection.class_name] += 1
            detection.zone_id = None
            for zone in self.zones:
                if self._in_zone(detection, zone, frame_width, frame_height):
                    detection.zone_id = zone["zone_id"]
                    counts_by_zone[zone["zone_id"]][detection.class_name] += 1
                    break

        result = {
            "detection_valid": True,
            "detection_state": "SUCCESS",
            "counts_by_class": dict(counts_by_class),
            "counts_by_zone": {zone: dict(counts) for zone, counts in counts_by_zone.items()},
            "zones": self.get_zones(),
            "detections": [d.as_dict() for d in detections],
            "note": "Observed detections only. No expected counts, inventory comparison, or shortage inference is configured.",
        }
        self.last_result = result
        return result

    def mark_insufficient_evidence(self, reason: str, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {
            "detection_valid": False,
            "detection_state": "LOW_CONFIDENCE",
            "error": str(reason),
            "diagnostics": diagnostics or {},
            "counts_by_class": {},
            "counts_by_zone": {zone["zone_id"]: {} for zone in self.zones},
            "zones": self.get_zones(),
            "detections": [],
            "note": "The frame did not provide sufficient detection evidence. No equipment shortage is inferred.",
        }
        self.last_result = result
        return result

    def mark_inference_error(self, error: str) -> dict[str, Any]:
        result = {
            "detection_valid": False,
            "detection_state": "DETECTION_ERROR",
            "error": str(error),
            "counts_by_class": {},
            "counts_by_zone": {zone["zone_id"]: {} for zone in self.zones},
            "zones": self.get_zones(),
            "detections": [],
            "note": "Equipment counts are unavailable because inference failed. No shortage is inferred.",
        }
        self.last_result = result
        return result

    def status(self) -> dict[str, Any]:
        return {**self.last_result, "zones": self.get_zones()}

    def reset(self) -> None:
        self.last_result = {
            "detection_valid": False,
            "detection_state": "NOT_RUN",
            "counts_by_class": {},
            "counts_by_zone": {zone["zone_id"]: {} for zone in self.zones},
            "zones": self.get_zones(),
            "detections": [],
        }
