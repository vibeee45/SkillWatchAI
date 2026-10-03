from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AttendanceStore:
    """SQLite-backed attendance, classroom, batch, session and audit store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self):
        with self._connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS classrooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                room TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                classroom_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                session_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'SCHEDULED',
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON attendance_sessions(status, camera_id);
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PRESENT',
                check_in_time TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                camera_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'AUTO',
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, student_id),
                FOREIGN KEY(session_id) REFERENCES attendance_sessions(id)
            );
            CREATE TABLE IF NOT EXISTS attendance_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance_id INTEGER,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT NOT NULL,
                reason TEXT,
                actor TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """)

    def upsert_classroom(self, classroom_id: str, name: str, room: str):
        now = self._now()
        with self._connect() as c:
            c.execute("""INSERT INTO classrooms(classroom_id,name,room,created_at)
                         VALUES(?,?,?,?) ON CONFLICT(classroom_id) DO UPDATE SET name=excluded.name,room=excluded.room""",
                      (classroom_id, name, room, now))
        return self.get_classroom(classroom_id)

    def get_classroom(self, classroom_id: str):
        with self._connect() as c:
            r = c.execute("SELECT * FROM classrooms WHERE classroom_id=?", (classroom_id,)).fetchone()
        return dict(r) if r else None

    def list_classrooms(self):
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM classrooms ORDER BY name").fetchall()]

    def upsert_batch(self, batch_id: str, name: str):
        now = self._now()
        with self._connect() as c:
            c.execute("""INSERT INTO batches(batch_id,name,created_at) VALUES(?,?,?)
                         ON CONFLICT(batch_id) DO UPDATE SET name=excluded.name""", (batch_id, name, now))
        with self._connect() as c:
            r = c.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        return dict(r)

    def list_batches(self):
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM batches ORDER BY name").fetchall()]

    def create_session(self, classroom_id: str, batch_id: str, session_date: str, start_time: str, end_time: str, camera_id: str):
        now = self._now()
        with self._connect() as c:
            cur = c.execute("""INSERT INTO attendance_sessions
                (classroom_id,batch_id,session_date,start_time,end_time,camera_id,status,created_at)
                VALUES(?,?,?,?,?,?, 'SCHEDULED',?)""",
                (classroom_id, batch_id, session_date, start_time, end_time, camera_id, now))
            sid = cur.lastrowid
        return self.get_session(sid)

    def get_session(self, session_id: int):
        with self._connect() as c:
            r = c.execute("SELECT * FROM attendance_sessions WHERE id=?", (session_id,)).fetchone()
        return dict(r) if r else None

    def list_sessions(self, limit: int = 50):
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM attendance_sessions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def start_session(self, session_id: int):
        now = self._now()
        with self._connect() as c:
            c.execute("UPDATE attendance_sessions SET status='ACTIVE',started_at=? WHERE id=?", (now, session_id))
        return self.get_session(session_id)

    def end_session(self, session_id: int):
        now = self._now()
        with self._connect() as c:
            c.execute("UPDATE attendance_sessions SET status='ENDED',ended_at=? WHERE id=?", (now, session_id))
        return self.get_session(session_id)

    def active_session_for_camera(self, camera_id: str):
        with self._connect() as c:
            r = c.execute("""SELECT * FROM attendance_sessions
                WHERE status='ACTIVE' AND camera_id=? ORDER BY id DESC LIMIT 1""", (camera_id,)).fetchone()
        return dict(r) if r else None

    def mark_present(self, session_id: int, student_id: str, confidence: float, camera_id: str):
        now = self._now()
        with self._connect() as c:
            existing = c.execute("SELECT * FROM attendance WHERE session_id=? AND student_id=?", (session_id, student_id)).fetchone()
            if existing:
                return dict(existing), False
            cur = c.execute("""INSERT INTO attendance
                (session_id,student_id,status,check_in_time,confidence,camera_id,source,updated_at)
                VALUES(?,?, 'PRESENT',?,?,?,?,?)""",
                (session_id, student_id, now, float(confidence), camera_id, 'AUTO', now))
            attendance_id = cur.lastrowid
            c.execute("""INSERT INTO attendance_audit
                (attendance_id,session_id,student_id,old_status,new_status,reason,actor,timestamp)
                VALUES(?,?,?,NULL,'PRESENT','Automatic confirmed recognition','SYSTEM',?)""",
                (attendance_id, session_id, student_id, now))
            row = c.execute("SELECT * FROM attendance WHERE id=?", (attendance_id,)).fetchone()
        return dict(row), True

    def correct(self, session_id: int, student_id: str, status: str, reason: str, actor: str):
        status = status.upper()
        if status not in {'PRESENT', 'ABSENT'}:
            raise ValueError('status must be PRESENT or ABSENT')
        now = self._now()
        with self._connect() as c:
            old = c.execute("SELECT * FROM attendance WHERE session_id=? AND student_id=?", (session_id, student_id)).fetchone()
            old_status = old['status'] if old else 'ABSENT'
            if old:
                c.execute("UPDATE attendance SET status=?,source='MANUAL',updated_at=? WHERE id=?", (status, now, old['id']))
                aid = old['id']
            else:
                c.execute("""INSERT INTO attendance(session_id,student_id,status,check_in_time,confidence,camera_id,source,updated_at)
                    VALUES(?,?,?, ?,0,'MANUAL','MANUAL',?)""", (session_id, student_id, status, now, now))
                aid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            c.execute("""INSERT INTO attendance_audit
                (attendance_id,session_id,student_id,old_status,new_status,reason,actor,timestamp)
                VALUES(?,?,?,?,?,?,?,?)""", (aid, session_id, student_id, old_status, status, reason, actor, now))
            row = c.execute("SELECT * FROM attendance WHERE id=?", (aid,)).fetchone()
        return dict(row)

    def records(self, session_id: int):
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM attendance WHERE session_id=? ORDER BY check_in_time", (session_id,)).fetchall()]

    def audit(self, session_id: int):
        with self._connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM attendance_audit WHERE session_id=? ORDER BY timestamp DESC", (session_id,)).fetchall()]

    def summary(self, session_id: int, students: list[dict[str, Any]]):
        session = self.get_session(session_id)
        if not session:
            return None
        batch_id = session['batch_id']
        eligible = [s for s in students if s.get('batch') == batch_id or s.get('batch') == session['batch_id']]
        records = self.records(session_id)
        by_id = {r['student_id']: r for r in records}
        present = sum(1 for r in records if r['status'] == 'PRESENT')
        total = len(eligible)
        absent = max(total - present, 0)
        pct = round((present / total) * 100, 2) if total else 0.0
        return {'session': session, 'total_students': total, 'present_count': present, 'absent_count': absent,
                'attendance_percentage': pct, 'records': records,
                'students': [{'student_id': s['student_id'], 'name': s['name'], 'status': by_id.get(s['student_id'], {}).get('status', 'ABSENT')}
                             for s in eligible]}
