"""Phase 10 compliance and alert engine.

Equipment rules are observation-only: infrastructure gaps are temporal changes
between valid observations, never comparisons against expected inventory counts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled_rules": {
        "attendance_discrepancy": True,
        "individual_attendance_mismatch": True,
        "low_occupancy": True,
        "infrastructure_observation_gap": True,
        "overcapacity": True,
        "camera_health": True,
        "low_confidence_manual_verification": True,
    },
    "thresholds": {
        "attendance_discrepancy_count": 2,
        "attendance_discrepancy_percent": 20.0,
        "low_occupancy_minimum": 1,
        "overcapacity_maximum": 40,
        "camera_stale_seconds": 10,
        "low_confidence_threshold": 0.40,
        "infrastructure_gap_repeated_observations": 5,
        "persistence_observations": 3,
        "manual_verification_repeated_observations": 3,
    },
    "severity": {
        "attendance_discrepancy": "medium",
        "individual_attendance_mismatch": "medium",
        "low_occupancy": "low",
        "infrastructure_observation_gap": "low",
        "overcapacity": "high",
        "camera_health": "high",
        "low_confidence_manual_verification": "low",
    },
}

RULE_LABELS = {
    "attendance_discrepancy": "Attendance discrepancy",
    "individual_attendance_mismatch": "Individual attendance mismatch",
    "low_occupancy": "Low occupancy",
    "infrastructure_observation_gap": "Infrastructure observation gap",
    "overcapacity": "Overcapacity",
    "camera_health": "Camera health",
    "low_confidence_manual_verification": "Manual verification required",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ComplianceAlertStore:
    """SQLite-backed active alerts and append-only alert event history."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS compliance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                rule_id TEXT NOT NULL,
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                scope_id TEXT NOT NULL DEFAULT 'global',
                confidence REAL NOT NULL DEFAULT 0.0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                resolved_at TEXT,
                consecutive_evidence INTEGER NOT NULL DEFAULT 1,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                resolution_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_compliance_alert_status ON compliance_alerts(status, last_seen_at);
            CREATE INDEX IF NOT EXISTS idx_compliance_alert_rule ON compliance_alerts(rule_id, status);
            CREATE TABLE IF NOT EXISTS compliance_alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                fingerprint TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_compliance_events_time ON compliance_alert_events(timestamp);
            """)

    def _event(self, conn, alert_id, fingerprint, event_type, payload):
        conn.execute(
            "INSERT INTO compliance_alert_events(alert_id,fingerprint,event_type,timestamp,event_json) VALUES(?,?,?,?,?)",
            (alert_id, fingerprint, event_type, utc_now(), json.dumps(payload, ensure_ascii=False, default=str)),
        )

    def observe(self, rule_id: str, title: str, severity: str, scope_id: str,
                confidence: float, evidence: dict[str, Any], required_repeats: int) -> dict[str, Any] | None:
        fingerprint = f"{rule_id}:{scope_id}"
        now = utc_now()
        confidence = max(0.0, min(1.0, float(confidence)))
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM compliance_alerts WHERE fingerprint=?", (fingerprint,)).fetchone()
            if row and row["status"] == "ACTIVE":
                count = int(row["consecutive_evidence"]) + 1
                conn.execute("UPDATE compliance_alerts SET severity=?,confidence=?,last_seen_at=?,consecutive_evidence=?,evidence_json=? WHERE id=?",
                             (severity, confidence, now, count, json.dumps(evidence, ensure_ascii=False, default=str), row["id"]))
                self._event(conn, row["id"], fingerprint, "EVIDENCE_REPEATED", {"count": count, "evidence": evidence})
                updated = conn.execute("SELECT * FROM compliance_alerts WHERE id=?", (row["id"],)).fetchone()
                return self._serialize(updated)
            previous_count = int(row["consecutive_evidence"]) + 1 if row else 1
            if previous_count < max(1, int(required_repeats)):
                # Store pending evidence as a non-alert counter in memory is handled by engine.
                return None
            if row:
                conn.execute("UPDATE compliance_alerts SET title=?,severity=?,status='ACTIVE',scope_id=?,confidence=?,first_seen_at=?,last_seen_at=?,resolved_at=NULL,consecutive_evidence=?,evidence_json=?,resolution_reason=NULL WHERE id=?",
                             (title, severity, scope_id, confidence, now, now, previous_count, json.dumps(evidence, ensure_ascii=False, default=str), row["id"]))
                alert_id = row["id"]
                event_type = "REOPENED"
            else:
                cur = conn.execute("INSERT INTO compliance_alerts(fingerprint,rule_id,title,severity,status,scope_id,confidence,first_seen_at,last_seen_at,consecutive_evidence,evidence_json) VALUES(?,?,?,?, 'ACTIVE',?,?,?,?,?,?)",
                                   (fingerprint, rule_id, title, severity, scope_id, confidence, now, now, previous_count, json.dumps(evidence, ensure_ascii=False, default=str)))
                alert_id = cur.lastrowid
                event_type = "CREATED"
            self._event(conn, alert_id, fingerprint, event_type, evidence)
            return self._serialize(conn.execute("SELECT * FROM compliance_alerts WHERE id=?", (alert_id,)).fetchone())

    def resolve_missing(self, active_fingerprints: set[str], reason: str = "Condition no longer observed in a valid evaluation"):
        now = utc_now()
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM compliance_alerts WHERE status='ACTIVE'").fetchall()
            for row in rows:
                if row["fingerprint"] not in active_fingerprints:
                    conn.execute("UPDATE compliance_alerts SET status='RESOLVED',resolved_at=?,resolution_reason=?,consecutive_evidence=0 WHERE id=?",
                                 (now, reason, row["id"]))
                    self._event(conn, row["id"], row["fingerprint"], "RESOLVED", {"reason": reason})

    @staticmethod
    def _serialize(row):
        if not row:
            return None
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json") or "{}")
        return item

    def list_alerts(self, status: str | None = "ACTIVE", limit: int = 100):
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            if status and status.upper() in {"ACTIVE", "RESOLVED"}:
                rows = conn.execute("SELECT * FROM compliance_alerts WHERE status=? ORDER BY last_seen_at DESC LIMIT ?", (status.upper(), limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM compliance_alerts ORDER BY last_seen_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._serialize(r) for r in rows]

    def events(self, limit: int = 100):
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM compliance_alert_events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["event"] = json.loads(item.pop("event_json") or "{}")
            result.append(item)
        return result

    def resolve(self, alert_id: int, reason: str = "Manually resolved"):
        now = utc_now()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM compliance_alerts WHERE id=?", (alert_id,)).fetchone()
            if not row:
                return None
            conn.execute("UPDATE compliance_alerts SET status='RESOLVED',resolved_at=?,resolution_reason=? WHERE id=?", (now, reason, alert_id))
            self._event(conn, alert_id, row["fingerprint"], "MANUALLY_RESOLVED", {"reason": reason})
            return self._serialize(conn.execute("SELECT * FROM compliance_alerts WHERE id=?", (alert_id,)).fetchone())


class ComplianceEngine:
    def __init__(self, db_path: str | Path, config_path: str | Path):
        self.store = ComplianceAlertStore(db_path)
        self.config_path = Path(config_path)
        self._lock = threading.RLock()
        self._streaks: dict[str, int] = {}
        self._previous_equipment: dict[str, set[str]] = {}
        self._equipment_absence_streaks: dict[str, int] = {}
        self.config = self._load_config()

    def _load_config(self):
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        if self.config_path.exists():
            try:
                saved = json.loads(self.config_path.read_text(encoding="utf-8"))
                for group in ("enabled_rules", "thresholds", "severity"):
                    if isinstance(saved.get(group), dict):
                        config[group].update(saved[group])
            except (OSError, ValueError, TypeError):
                pass
        return config

    def get_config(self):
        return json.loads(json.dumps(self.config))

    def update_config(self, payload: dict[str, Any]):
        candidate = self.get_config()
        for group in ("enabled_rules", "thresholds", "severity"):
            values = payload.get(group, {})
            if not isinstance(values, dict):
                raise ValueError(f"{group} must be an object")
            for key, value in values.items():
                if key not in candidate[group]:
                    raise ValueError(f"Unknown {group} setting: {key}")
                if group == "enabled_rules":
                    if not isinstance(value, bool): raise ValueError(f"{key} must be boolean")
                elif group == "severity":
                    if str(value).lower() not in {"info", "low", "medium", "high", "critical"}: raise ValueError(f"Invalid severity for {key}")
                    value = str(value).lower()
                else:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
                        raise ValueError(f"{key} must be a non-negative number")
                    value = float(value) if isinstance(value, float) or "." in str(value) else int(value)
                candidate[group][key] = value
        self.config = candidate
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        tmp.replace(self.config_path)
        return self.get_config()

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cfg = self.config
            enabled = cfg["enabled_rules"]
            th = cfg["thresholds"]
            severity = cfg["severity"]
            candidates: dict[str, dict[str, Any]] = {}
            active_fingerprints: set[str] = set()

            def add(rule_id: str, scope: str, evidence: dict[str, Any], confidence: float = 1.0, repeats: int | None = None):
                if not enabled.get(rule_id, False): return
                fp = f"{rule_id}:{scope}"
                active_fingerprints.add(fp)
                self._streaks[fp] = self._streaks.get(fp, 0) + 1
                required = max(1, int(repeats if repeats is not None else th["persistence_observations"]))
                if self._streaks[fp] >= required:
                    active_fingerprints.add(fp)
                    alert = self.store.observe(rule_id, RULE_LABELS[rule_id], severity[rule_id], scope, confidence, evidence, required_repeats=1)
                    if alert: candidates[fp] = alert

            def resolve_streaks_not_seen():
                for fp in list(self._streaks):
                    if fp not in active_fingerprints and fp.split(":", 1)[0] not in [x.split(":", 1)[0] for x in active_fingerprints]:
                        # Preserve an alert's evidence streak only while its condition remains true.
                        if fp not in observed_fingerprints:
                            self._streaks.pop(fp, None)

            observed_fingerprints: set[str] = set()
            # Attendance aggregate discrepancy, only while an actual session is active.
            session = context.get("active_session")
            summary = context.get("attendance_summary") or {}
            occ = context.get("occupancy") or {}
            recognized = context.get("recognition") or []
            confirmed_ids = sorted({str(r.get("student_id")) for r in recognized if r.get("status") == "confirmed" and r.get("student_id")})
            if session and bool((context.get("camera") or {}).get("connected", False)):
                present = int(summary.get("present_count", 0))
                total = int(summary.get("total_students", 0))
                records = summary.get("records", [])
                # Aggregate discrepancy is evaluated after the session ends, against the
                # enrolled roster and explicit attendance records, not current frame visibility.
                if str(session.get("status", "")).upper() == "ENDED":
                    discrepancy = abs(total - len(records))
                    pct = (discrepancy / max(total, 1)) * 100.0
                    fp = f"attendance_discrepancy:{session.get('id', 'active')}"
                    if discrepancy >= int(th["attendance_discrepancy_count"]) and pct >= float(th["attendance_discrepancy_percent"]):
                        observed_fingerprints.add(fp)
                        add("attendance_discrepancy", str(session.get("id", "active")), {"session_id": session.get("id"), "present_count": present, "explicit_record_count": len(records), "roster_count": total, "difference": discrepancy, "difference_percent": round(pct, 2)}, confidence=0.9)
                # Individual mismatch: recognized identity has a manual ABSENT record.
                status_by_student = {str(r.get("student_id")): str(r.get("status", "")).upper() for r in records}
                for sid in confirmed_ids if str(session.get("status", "")).upper() == "ACTIVE" else []:
                    if status_by_student.get(sid) == "ABSENT":
                        fp = f"individual_attendance_mismatch:{session.get('id', 'active')}:{sid}"
                        observed_fingerprints.add(fp)
                        add("individual_attendance_mismatch", f"{session.get('id', 'active')}:{sid}", {"session_id": session.get("id"), "student_id": sid, "attendance_status": "ABSENT", "recognition_status": "confirmed"}, confidence=1.0)

            # Occupancy rules are based on configured operational thresholds, not equipment counts.
            occ_valid = bool(context.get("evaluation_valid", False)) and bool(occ.get("detections_valid", True)) and isinstance(occ.get("total_occupancy"), (int, float))
            if occ_valid:
                occupancy = int(occ.get("total_occupancy", 0))
                scope = str(session.get("classroom_id", "classroom") if session else context.get("room_id", "classroom"))
                fp = f"low_occupancy:{scope}"
                if session and str(session.get("status", "")).upper() == "ACTIVE" and occupancy < int(th["low_occupancy_minimum"]):
                    observed_fingerprints.add(fp)
                    add("low_occupancy", scope, {"occupancy": occupancy, "minimum_threshold": int(th["low_occupancy_minimum"]), "session_id": session.get("id")}, confidence=0.8)
                fp = f"overcapacity:{scope}"
                if occupancy > int(th["overcapacity_maximum"]):
                    observed_fingerprints.add(fp)
                    add("overcapacity", scope, {"occupancy": occupancy, "maximum_threshold": int(th["overcapacity_maximum"]), "session_id": session.get("id")}, confidence=0.9)

            # Camera health and frame freshness.
            camera = context.get("camera") or {}
            camera_bad = not camera.get("connected", False) or str(camera.get("health", "unknown")).lower() in {"offline", "error", "degraded", "frozen", "poor_quality", "stale"}
            age = context.get("camera_frame_age_seconds")
            if age is not None and float(age) > float(th["camera_stale_seconds"]): camera_bad = True
            fp = f"camera_health:{camera.get('camera_id') or 'default'}"
            if camera_bad:
                observed_fingerprints.add(fp)
                add("camera_health", str(camera.get("camera_id") or "default"), {"connected": camera.get("connected", False), "health": camera.get("health", "unknown"), "last_frame_at": camera.get("last_frame_at"), "frame_age_seconds": age, "last_error": camera.get("last_error")}, confidence=1.0)

            # Low-confidence/manual verification: recognition confidence or valid equipment detections below threshold.
            low_conf = []
            low_th = float(th["low_confidence_threshold"])
            for result in recognized:
                score = result.get("confidence", result.get("similarity"))
                if score is not None and float(score) < low_th:
                    low_conf.append({"kind": "face_recognition", "student_id": result.get("student_id"), "status": result.get("status"), "confidence": float(score)})
            infra = context.get("infrastructure") or {}
            if infra.get("detection_valid"):
                for detection in infra.get("detections", []):
                    score = detection.get("confidence")
                    if score is not None and float(score) < low_th:
                        low_conf.append({"kind": "equipment_detection", "class_name": detection.get("class_name"), "confidence": float(score), "bbox": detection.get("bbox")})
            if low_conf:
                fp = f"low_confidence_manual_verification:{camera.get('camera_id') or 'default'}"
                observed_fingerprints.add(fp)
                add("low_confidence_manual_verification", str(camera.get("camera_id") or "default"), {"threshold": low_th, "items": low_conf[:30]}, confidence=min(float(x["confidence"]) for x in low_conf), repeats=int(th["manual_verification_repeated_observations"]))

            # Observation-only infrastructure gap: a previously observed class is no longer
            # visible across repeated valid frames. Never compares to a required quantity.
            if infra.get("detection_valid") and infra.get("detection_state", "SUCCESS") == "SUCCESS":
                counts = infra.get("counts_by_class", {})
                visible = {str(k) for k, v in counts.items() if int(v or 0) > 0}
                scope = str(camera.get("camera_id") or context.get("room_id", "classroom"))
                known = self._previous_equipment.get(scope, set()) | visible
                for class_name in sorted(known):
                    streak_key = f"{scope}:{class_name}"
                    if class_name in visible:
                        self._equipment_absence_streaks[streak_key] = 0
                    elif class_name in self._previous_equipment.get(scope, set()):
                        self._equipment_absence_streaks[streak_key] = self._equipment_absence_streaks.get(streak_key, 0) + 1
                        if self._equipment_absence_streaks[streak_key] >= int(th["infrastructure_gap_repeated_observations"]):
                            fp = f"infrastructure_observation_gap:{scope}:{class_name}"
                            observed_fingerprints.add(fp)
                            add("infrastructure_observation_gap", f"{scope}:{class_name}", {"camera_id": scope, "class_name": class_name, "gap_type": "previously_observed_class_not_visible", "absence_observations": self._equipment_absence_streaks[streak_key], "previously_observed_classes": sorted(known), "currently_visible_classes": sorted(visible), "note": "Observation change only; not an inventory shortage."}, confidence=0.6, repeats=1)
                self._previous_equipment[scope] = known

            # Reset consecutive evidence streaks for conditions absent from this valid evaluation.
            # Avoid resolving alerts during missing/invalid sensor observations.
            for fp in list(self._streaks):
                if fp not in observed_fingerprints:
                    self._streaks.pop(fp, None)
            if context.get("evaluation_valid", True):
                self.store.resolve_missing(active_fingerprints, "Condition cleared during a valid evaluation")
            return {"evaluated_at": utc_now(), "active_alerts": self.store.list_alerts("ACTIVE"), "new_or_updated_alerts": list(candidates.values()), "observed_conditions": sorted(observed_fingerprints), "configuration": self.get_config()}
