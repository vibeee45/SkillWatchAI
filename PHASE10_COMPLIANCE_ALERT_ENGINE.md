# Phase 10 — Compliance & Alert Engine

Phase 10 adds configurable, persistent, evidence-backed alerts. It does not reintroduce expected equipment counts or inventory-shortage comparisons.

## Rules

1. `attendance_discrepancy`: after an attendance session ends, flags a difference between enrolled roster size and explicit attendance records when configured count and percentage thresholds are met.
2. `individual_attendance_mismatch`: flags a currently confirmed identity that has an `ABSENT` attendance record during an active session.
3. `low_occupancy`: during an active attendance session, flags smoothed occupancy below a configurable minimum.
4. `infrastructure_observation_gap`: flags an equipment class that was previously visible but has not been detected across a configurable number of consecutive valid observations. This is an observation-change/manual-check signal only; it never says that an expected quantity is missing.
5. `overcapacity`: during an active session, flags smoothed occupancy above a configurable maximum.
6. `camera_health`: flags a disconnected, degraded, offline, or stale camera feed.
7. `low_confidence_manual_verification`: flags low-confidence face/equipment detections for human review.

## Persistence and evidence

- Conditions require repeated observations by default before alert creation.
- Alerts are deduplicated by rule and scope, updated while active, and resolved when the condition clears during a valid sensor evaluation.
- SQLite stores alert ID, rule, title, severity, status, scope, confidence, first/last timestamps, repeat count, evidence metadata, and resolution information.
- An append-only event table records creation, repeated evidence, reopening, and resolution.
- Rule settings persist to `data/compliance_config.json`; alerts persist to `data/compliance_alerts.db`.

## API

- `GET /api/compliance/status`
- `GET /api/compliance/config`
- `PUT /api/compliance/config`
- `POST /api/compliance/evaluate`
- `GET /api/compliance/alerts?status=ACTIVE&limit=100`
- `GET /api/compliance/events?limit=100`
- `POST /api/compliance/alerts/{alert_id}/resolve`

Config JSON shape:

```json
{
  "enabled_rules": {"low_occupancy": true},
  "thresholds": {"low_occupancy_minimum": 1, "persistence_observations": 3},
  "severity": {"low_occupancy": "low"}
}
```

Only known settings can be updated. Thresholds must be non-negative numbers and severity must be `info`, `low`, `medium`, `high`, or `critical`.

## Default thresholds

- Attendance discrepancy: minimum difference 2 students and 20% of roster.
- Low occupancy: fewer than 1 person while an attendance session is active.
- Overcapacity: more than 40 detected people while a session is active.
- Stale camera frame: more than 10 seconds old.
- Low confidence: score below 0.40.
- Infrastructure observation gap: class absent across 5 valid observations.
- General persistence: 3 observations.
- Manual verification persistence: 3 observations.

These are prototype defaults and should be calibrated for the demonstration site. No equipment expected-count setting exists.
