from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StudentStore:
    """Local SQLite store for enrollment metadata and protected representations."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    batch TEXT NOT NULL,
                    representation TEXT NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    quality_score REAL NOT NULL DEFAULT 0,
                    consent_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_students_name ON students(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_students_batch ON students(batch)")

    def upsert(
        self,
        student_id: str,
        name: str,
        batch: str,
        representation: str,
        sample_count: int,
        quality_score: float,
        consent_confirmed: bool,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, created_at FROM students WHERE student_id = ?",
                (student_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE students
                    SET name=?, batch=?, representation=?, sample_count=?, quality_score=?,
                        consent_confirmed=?, updated_at=?
                    WHERE student_id=?
                    """,
                    (name, batch, representation, sample_count, float(quality_score), int(consent_confirmed), now, student_id),
                )
                created_at = existing["created_at"]
            else:
                conn.execute(
                    """
                    INSERT INTO students
                    (student_id, name, batch, representation, sample_count, quality_score,
                     consent_confirmed, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (student_id, name, batch, representation, sample_count, float(quality_score), int(consent_confirmed), now, now),
                )
                created_at = now
        return self.get(student_id) | {"created_at": created_at}

    def get(self, student_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
        return self._public(row) if row else None

    def get_with_representation(self, student_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["consent_confirmed"] = bool(item["consent_confirmed"])
        return item

    def list(self, query: str = "") -> list[dict[str, Any]]:
        q = query.strip()
        with self._connect() as conn:
            if q:
                rows = conn.execute(
                    """
                    SELECT * FROM students
                    WHERE student_id LIKE ? OR name LIKE ? OR batch LIKE ?
                    ORDER BY name COLLATE NOCASE, student_id
                    """,
                    (f"%{q}%", f"%{q}%", f"%{q}%"),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM students ORDER BY name COLLATE NOCASE, student_id").fetchall()
        return [self._public(row) for row in rows]

    def delete(self, student_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            return cur.rowcount > 0

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "student_id": row["student_id"],
            "name": row["name"],
            "batch": row["batch"],
            "sample_count": int(row["sample_count"]),
            "quality_score": round(float(row["quality_score"]), 3),
            "consent_confirmed": bool(row["consent_confirmed"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "representation_protected": True,
        }
