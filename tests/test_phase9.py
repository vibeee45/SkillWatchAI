from detection.infrastructure import EquipmentDetection
from compliance.rules import InfrastructureObservationService


def test_phase9_reports_observed_counts_without_expected_counts_or_alerts():
    service = InfrastructureObservationService()
    detections = [
        EquipmentDetection(56, "chair", 10, 10, 30, 30, 0.9),
        EquipmentDetection(63, "laptop", 20, 20, 40, 30, 0.8),
        EquipmentDetection(56, "chair", 500, 10, 30, 30, 0.9),
    ]
    result = service.process(detections, 640, 360)
    assert result["detection_valid"] is True
    assert result["counts_by_class"] == {"chair": 2, "laptop": 1}
    assert "expected_count" not in str(result)
    assert "missing_count" not in str(result)
    assert "compliance" not in result
    assert "alerts" not in result


def test_phase9_counts_equipment_by_zone():
    service = InfrastructureObservationService()
    service.set_zones([{"zone_id": "left", "x1": 0, "y1": 0, "x2": 0.5, "y2": 1}])
    detections = [
        EquipmentDetection(56, "chair", 10, 10, 30, 30, 0.9),
        EquipmentDetection(63, "laptop", 20, 20, 40, 30, 0.8),
        EquipmentDetection(56, "chair", 500, 10, 30, 30, 0.9),
    ]
    result = service.process(detections, 640, 360)
    assert result["counts_by_class"] == {"chair": 2, "laptop": 1}
    assert result["counts_by_zone"]["left"] == {"chair": 1, "laptop": 1}
    assert detections[0].zone_id == "left"
    assert detections[2].zone_id is None


def test_phase9_zone_validation():
    service = InfrastructureObservationService()
    try:
        service.set_zones([{"zone_id": "bad", "x1": 0.8, "y1": 0, "x2": 0.2, "y2": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid zone should be rejected")


def test_inference_error_does_not_invent_zero_observations_or_shortage():
    service = InfrastructureObservationService()
    result = service.mark_inference_error("synthetic model failure")
    assert result["detection_valid"] is False
    assert result["detection_state"] == "DETECTION_ERROR"
    assert result["counts_by_class"] == {}
    assert "expected_count" not in str(result)
    assert "missing_count" not in str(result)
    assert "alerts" not in result


def test_low_confidence_frame_is_not_reported_as_valid_zero_count():
    service = InfrastructureObservationService()
    result = service.mark_insufficient_evidence("synthetic low confidence")
    assert result["detection_valid"] is False
    assert result["detection_state"] == "LOW_CONFIDENCE"
    assert result["counts_by_class"] == {}
    assert "shortage" in result["note"].lower()
