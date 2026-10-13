import numpy as np

from detection.person_detector import PersonDetection
from detection.occupancy import OccupancyService


def test_phase8_tracking_and_smoothing():
    service = OccupancyService(history_size=5)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    dets = [PersonDetection(100, 80, 120, 220, 0.8)]
    result = service.process(dets, [], [], 640, 360)
    assert result["raw_occupancy"] == 1
    assert result["total_occupancy"] == 1
    assert result["unknown_persons"] == 1
    assert result["active_tracks"] == 1


def test_phase8_zone_assignment():
    service = OccupancyService(history_size=3)
    service.set_zones([{"zone_id": "front", "x1": 0, "y1": 0, "x2": 0.5, "y2": 1}])
    dets = [PersonDetection(20, 50, 100, 200, 0.8)]
    result = service.process(dets, [], [], 640, 360)
    assert result["zones"]["front"] == 1
    assert result["detections"][0]["zone"] == "front"
