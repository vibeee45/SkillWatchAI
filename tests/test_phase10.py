import json
from compliance.engine import ComplianceEngine, ComplianceAlertStore


def make_engine(tmp_path):
    return ComplianceEngine(tmp_path / "alerts.db", tmp_path / "config.json")


def base_context():
    return {
        "camera": {"connected": True, "health": "healthy", "camera_id": "CAM-001", "last_frame_at": None},
        "camera_frame_age_seconds": 0,
        "active_session": {"id": 1, "status": "ACTIVE", "classroom_id": "ROOM-01"},
        "attendance_summary": {"total_students": 10, "present_count": 2, "records": []},
        "occupancy": {"total_occupancy": 0, "detections": []},
        "infrastructure": {"detection_valid": True, "detection_state": "SUCCESS", "counts_by_class": {"chair": 3}, "detections": []},
        "recognition": [],
        "evaluation_valid": True,
        "room_id": "ROOM-01",
    }


def test_alert_persistence_and_metadata(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 2, "low_occupancy_minimum": 1}})
    context = base_context()
    engine.evaluate(context)
    assert engine.store.list_alerts("ACTIVE") == []
    result = engine.evaluate(context)
    alerts = result["active_alerts"]
    assert any(a["rule_id"] == "low_occupancy" for a in alerts)
    alert = next(a for a in alerts if a["rule_id"] == "low_occupancy")
    assert alert["severity"] == "low"
    assert alert["first_seen_at"] and alert["last_seen_at"]
    assert alert["evidence"]
    assert engine.store.events()


def test_infrastructure_gap_is_observation_only_and_repeated(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"infrastructure_gap_repeated_observations": 2, "persistence_observations": 1}})
    context = base_context()
    context["active_session"] = None
    context["infrastructure"]["counts_by_class"] = {"chair": 2, "laptop": 1}
    engine.evaluate(context)
    context["infrastructure"]["counts_by_class"] = {"chair": 2}
    engine.evaluate(context)
    result = engine.evaluate(context)
    gaps = [a for a in result["active_alerts"] if a["rule_id"] == "infrastructure_observation_gap"]
    assert any("laptop" in a["scope_id"] for a in gaps)
    assert all("expected_count" not in json.dumps(a) and "missing_count" not in json.dumps(a) for a in gaps)


def test_camera_health_alert_is_persisted(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 1}})
    context = base_context()
    context["camera"]["connected"] = False
    context["camera"]["health"] = "disconnected"
    result = engine.evaluate(context)
    alert = next(a for a in result["active_alerts"] if a["rule_id"] == "camera_health")
    assert alert["severity"] == "high"
    assert alert["evidence"]["connected"] is False


def test_config_is_persistent_and_validated(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"overcapacity_maximum": 25}, "enabled_rules": {"low_occupancy": False}})
    restored = make_engine(tmp_path)
    assert restored.get_config()["thresholds"]["overcapacity_maximum"] == 25
    assert restored.get_config()["enabled_rules"]["low_occupancy"] is False
    try:
        restored.update_config({"thresholds": {"expected_chairs": 10}})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown inventory-like setting must be rejected")


def test_individual_attendance_mismatch_rule(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 1}})
    context = base_context()
    context["recognition"] = [{"student_id": "S-1", "status": "confirmed", "confidence": 0.95}]
    context["attendance_summary"]["records"] = [{"student_id": "S-1", "status": "ABSENT"}]
    result = engine.evaluate(context)
    assert any(a["rule_id"] == "individual_attendance_mismatch" for a in result["active_alerts"])


def test_ended_session_attendance_discrepancy(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 1, "attendance_discrepancy_count": 2, "attendance_discrepancy_percent": 20}})
    context = base_context()
    context["active_session"]["status"] = "ENDED"
    context["attendance_summary"]["records"] = [{"student_id": "S-1", "status": "PRESENT"}]
    result = engine.evaluate(context)
    assert any(a["rule_id"] == "attendance_discrepancy" for a in result["active_alerts"])


def test_overcapacity_rule_uses_configurable_threshold(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 1, "overcapacity_maximum": 3}})
    context = base_context()
    context["occupancy"]["total_occupancy"] = 5
    result = engine.evaluate(context)
    alert = next(a for a in result["active_alerts"] if a["rule_id"] == "overcapacity")
    assert alert["evidence"]["occupancy"] == 5
    assert alert["severity"] == "high"


def test_low_confidence_rule_requests_manual_verification(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"manual_verification_repeated_observations": 1}})
    context = base_context()
    context["recognition"] = [{"student_id": None, "status": "unknown", "confidence": 0.22}]
    result = engine.evaluate(context)
    alert = next(a for a in result["active_alerts"] if a["rule_id"] == "low_confidence_manual_verification")
    assert alert["evidence"]["items"][0]["confidence"] == 0.22


def test_alert_resolves_when_condition_clears_on_valid_evaluation(tmp_path):
    engine = make_engine(tmp_path)
    engine.update_config({"thresholds": {"persistence_observations": 1, "overcapacity_maximum": 3}})
    context = base_context()
    context["occupancy"]["total_occupancy"] = 5
    engine.evaluate(context)
    assert any(a["rule_id"] == "overcapacity" for a in engine.store.list_alerts("ACTIVE"))
    context["occupancy"]["total_occupancy"] = 2
    engine.evaluate(context)
    assert not any(a["rule_id"] == "overcapacity" for a in engine.store.list_alerts("ACTIVE"))
    assert any(a["rule_id"] == "overcapacity" for a in engine.store.list_alerts("RESOLVED"))
