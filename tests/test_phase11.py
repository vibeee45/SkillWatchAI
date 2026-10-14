import json

from offline.event_queue import SQLiteEventQueue


def test_enqueue_and_retrieve_pending(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")

    event = queue.enqueue(
        "attendance",
        {"student_id": "TEST-001", "camera_id": "CAM-001"},
    )

    pending = queue.get_pending()

    assert len(pending) == 1
    assert pending[0]["event_id"] == event["event_id"]
    assert pending[0]["payload"]["student_id"] == "TEST-001"
    assert pending[0]["status"] == "pending"


def test_duplicate_event_id_is_not_duplicated(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")

    first = queue.enqueue(
        "attendance",
        {"value": 1},
        event_id="event-001",
    )

    second = queue.enqueue(
        "attendance",
        {"value": 2},
        event_id="event-001",
    )

    assert first["event_id"] == second["event_id"]
    assert queue.status()["total"] == 1

    # The original event is preserved.
    assert queue.get("event-001")["payload"] == {"value": 1}


def test_events_survive_queue_recreation(tmp_path):
    db_path = tmp_path / "events.db"

    queue = SQLiteEventQueue(db_path)
    queue.enqueue("camera_health", {"camera_id": "CAM-001"})

    reopened = SQLiteEventQueue(db_path)

    assert reopened.status()["pending"] == 1


def test_mark_synced_removes_event_from_pending(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")
    event = queue.enqueue("attendance", {"value": 1})

    assert queue.mark_synced(event["event_id"]) is True
    assert queue.get_pending() == []
    assert queue.status()["synced"] == 1


def test_failed_attempt_is_scheduled_for_retry(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")
    event = queue.enqueue("attendance", {"value": 1})

    assert queue.mark_failed(
        event["event_id"],
        "Network unavailable",
        max_attempts=3,
        retry_delay_seconds=60,
    )

    stored = queue.get(event["event_id"])

    assert stored["attempts"] == 1
    assert stored["status"] == "pending"
    assert stored["last_error"] == "Network unavailable"
    assert queue.get_pending() == []


def test_repeated_failures_eventually_stop_automatic_retries(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")
    event = queue.enqueue("attendance", {"value": 1})

    for _ in range(3):
        assert queue.mark_failed(
            event["event_id"],
            "Network unavailable",
            max_attempts=3,
            retry_delay_seconds=0,
        )

    stored = queue.get(event["event_id"])

    assert stored["attempts"] == 3
    assert stored["status"] == "failed"
    assert queue.status()["failed"] == 1
