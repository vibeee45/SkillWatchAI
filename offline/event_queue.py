from __future__ import annotations

import json
import sqlite3
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteEventQueue:
    """Persistent local queue for events awaiting synchronization."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queued_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'syncing', 'synced', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_error TEXT,
                    synced_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queued_events_status
                ON queued_events(status, next_attempt_at, id)
                """
            )
            # A process may exit after reserving an event but before recording
            # delivery. Return such rows to pending; the receiver must deduplicate
            # by event_id because delivery is at-least-once across crash boundaries.
            conn.execute(
                "UPDATE queued_events SET status='pending' WHERE status='syncing'"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def enqueue(
        self,
        event_type: str,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an event locally. Repeated event IDs are not duplicated."""
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type cannot be empty")

        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")

        event_id = event_id or str(uuid.uuid4())
        created_at = self._now()
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO queued_events
                    (event_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_id, event_type, payload_json, created_at),
            )
            row = conn.execute(
                "SELECT * FROM queued_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        return self._deserialize(row)

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def get_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return pending events in creation order without changing their state."""
        limit = max(1, min(int(limit), 1000))

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM queued_events
                WHERE status = 'pending'
                  AND (
                    next_attempt_at IS NULL
                    OR next_attempt_at <= ?
                  )
                ORDER BY id ASC
                LIMIT ?
                """,
                (self._now(), limit),
            ).fetchall()

        return [self._deserialize(row) for row in rows]

    def mark_syncing(self, event_id: str) -> bool:
        """Atomically reserve a due pending event for a sync worker."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """UPDATE queued_events SET status = 'syncing'
                   WHERE event_id = ? AND status = 'pending'
                   AND (next_attempt_at IS NULL OR next_attempt_at <= ?)""",
                (event_id, self._now()),
            )
            return cursor.rowcount > 0

    def mark_synced(self, event_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE queued_events
                SET status = 'synced',
                    synced_at = ?,
                    last_error = NULL,
                    next_attempt_at = NULL
                WHERE event_id = ?
                  AND status != 'synced'
                """,
                (self._now(), event_id),
            )
            return cursor.rowcount > 0

    def mark_failed(
        self,
        event_id: str,
        error: str,
        *,
        max_attempts: int = 5,
        retry_delay_seconds: int = 5,
    ) -> bool:
        """Record a failed attempt and schedule a retry, or mark terminal failure."""
        max_attempts = max(1, int(max_attempts))
        retry_delay_seconds = max(0, int(retry_delay_seconds))
        error = str(error)[:1000]

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT attempts
                FROM queued_events
                WHERE event_id = ?
                  AND status IN ('pending', 'syncing', 'failed')
                """,
                (event_id,),
            ).fetchone()

            if row is None:
                return False

            attempts = int(row["attempts"]) + 1
            terminal = attempts >= max_attempts
            status = "failed" if terminal else "pending"

            next_attempt_at = None
            if not terminal:
                next_attempt_at = (
                    datetime.now(timezone.utc)
                    .timestamp() + retry_delay_seconds
                )
                next_attempt_at = datetime.fromtimestamp(
                    next_attempt_at,
                    timezone.utc,
                ).isoformat()

            conn.execute(
                """
                UPDATE queued_events
                SET status = ?,
                    attempts = ?,
                    last_error = ?,
                    next_attempt_at = ?
                WHERE event_id = ?
                """,
                (status, attempts, error, next_attempt_at, event_id),
            )

        return True

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM queued_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        return self._deserialize(row) if row else None

    def status(self) -> dict[str, int]:
        """Return counts without exposing event payloads."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS total
                FROM queued_events
                GROUP BY status
                """
            ).fetchall()

        counts = {
            "pending": 0,
            "syncing": 0,
            "synced": 0,
            "failed": 0,
        }
        for row in rows:
            counts[row["status"]] = int(row["total"])

        counts["total"] = sum(counts.values())
        return counts
