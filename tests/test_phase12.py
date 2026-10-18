import pytest
from fastapi.testclient import TestClient
from database.connection import Base, engine, SessionLocal, init_db
from database.models import Centre, Room, CameraModel, StudentModel, ComplianceAlertModel
from backend.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_database_models_init():
    """Verify Phase 12 SQLAlchemy models initialize correctly and tables exist."""
    init_db()
    with SessionLocal() as db:
        # Check we can insert and query a centre
        centre = Centre(code="TC-TEST-01", name="Test Training Centre", district="Central", state="Delhi")
        db.add(centre)
        db.commit()
        db.refresh(centre)
        assert centre.id is not None
        assert centre.code == "TC-TEST-01"

        # Check room relation
        room = Room(centre_id=centre.id, room_id="TEST-ROOM-101", name="Computer Lab", capacity=25)
        db.add(room)
        db.commit()
        db.refresh(room)
        assert room.id is not None
        assert room.centre.name == "Test Training Centre"

        # Cleanup
        db.delete(room)
        db.delete(centre)
        db.commit()


def test_dashboard_summary_endpoint(client):
    """Verify /api/dashboard/summary provides unified executive metrics."""
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()

    assert "timestamp" in data
    assert "camera" in data
    assert "occupancy" in data
    assert "attendance" in data
    assert "infrastructure" in data
    assert "compliance" in data
    assert "offline" in data

    # Verify key structure
    assert "total" in data["occupancy"]
    assert "recognized" in data["occupancy"]
    assert "unknown" in data["occupancy"]
    assert "has_active_session" in data["attendance"]
    assert "detected_counts" in data["infrastructure"]
    assert "active_alerts_count" in data["compliance"]
    assert "pending_events" in data["offline"]
