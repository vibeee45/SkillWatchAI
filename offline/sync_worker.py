"""Background delivery of locally queued events to an explicitly configured receiver.

No events leave the machine unless SKILLWATCH_SYNC_URL is configured.
The receiver should accept POST JSON with event_id, event_type, created_at and payload,
and should treat event_id as an idempotency key.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from offline.event_queue import SQLiteEventQueue


class EventSyncWorker:
    def __init__(self, queue: SQLiteEventQueue, endpoint: str | None = None,
                 token: str | None = None, interval_seconds: float = 5.0,
                 request_timeout_seconds: float = 8.0, max_attempts: int = 8):
        self.queue = queue
        self.endpoint = (endpoint if endpoint is not None else os.getenv("SKILLWATCH_SYNC_URL", "")).strip()
        self.token = token if token is not None else os.getenv("SKILLWATCH_SYNC_TOKEN", "")
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "configured": bool(self.endpoint), "running": False,
            "last_attempt_at": None, "last_success_at": None,
            "last_error": None, "last_batch_synced": 0,
        }

    def start(self) -> None:
        if not self.endpoint or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="skillwatch-event-sync", daemon=True)
        self._thread.start()
        with self._lock:
            self._state["running"] = True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._lock:
            self._state["running"] = False

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _deliver(self, event: dict[str, Any]) -> None:
        body = json.dumps({
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "created_at": event["created_at"],
            "payload": event["payload"],
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Idempotency-Key": event["event_id"]}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Receiver returned HTTP {response.status}")

    def sync_once(self, limit: int = 50) -> dict[str, Any]:
        if not self.endpoint:
            result = {"configured": False, "synced": 0, "failed": 0,
                      "message": "No remote receiver configured; events remain safely queued locally."}
            with self._lock:
                self._state.update(last_attempt_at=self._now(), last_batch_synced=0,
                                   last_error=None, configured=False)
            return result
        synced = failed = 0
        errors: list[str] = []
        with self._lock:
            self._state["last_attempt_at"] = self._now()
            self._state["last_error"] = None
        for event in self.queue.get_pending(limit=limit):
            if not self.queue.mark_syncing(event["event_id"]):
                continue
            try:
                self._deliver(event)
                self.queue.mark_synced(event["event_id"])
                synced += 1
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                self.queue.mark_failed(event["event_id"], str(exc),
                                       max_attempts=self.max_attempts,
                                       retry_delay_seconds=min(300, 2 ** min(event["attempts"], 8)))
                failed += 1
                errors.append(str(exc)[:250])
            except Exception as exc:  # Keep the worker alive on unexpected receiver errors.
                logging.getLogger("skillwatch.offline").exception("Unexpected event sync failure")
                self.queue.mark_failed(event["event_id"], str(exc), max_attempts=self.max_attempts,
                                       retry_delay_seconds=min(300, 2 ** min(event["attempts"], 8)))
                failed += 1
                errors.append(str(exc)[:250])
        with self._lock:
            self._state["last_batch_synced"] = synced
            if synced:
                self._state["last_success_at"] = self._now()
            self._state["last_error"] = "; ".join(errors[:3]) if errors else None
        return {"configured": True, "synced": synced, "failed": failed,
                "queue": self.queue.status(), "errors": errors[:3]}

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        state["configured"] = bool(self.endpoint)
        state["running"] = bool(self._thread and self._thread.is_alive())
        state["queue"] = self.queue.status()
        state["mode"] = "remote_sync_enabled" if self.endpoint else "local_queue_only"
        return state

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sync_once()
            except Exception:
                logging.getLogger("skillwatch.offline").exception("Event sync loop failed")
