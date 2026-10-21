# SkillWatch AI — SIH Live Demonstration Script & Protocol
**Problem Statement 26245 — Smart India Hackathon**  
*AI-Based Real-Time Monitoring of Training Centres for Attendance and Infrastructure Compliance*

This guide outlines the 7-step live demonstration sequence aligned directly with **Slide 12 of the official SIH Submission Content Pack**.

---

## Prerequisites & Launch

1. **Start the backend server:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Open the Dashboard in your browser:**
   ```
   http://localhost:8000/
   ```
   The application features official SkillWatch AI branding with a persistent top navigation bar allowing seamless switching between dedicated views without dropping the camera feed:
   - **Live Monitoring (`#/live`)**: Executive KPI overview and real-time MJPEG camera preview.
   - **Attendance (`#/attendance`)**: Session lifecycle controls and immutable audit log.
   - **Students (`#/students`)**: Consented 3-sample face enrollment and student roster gallery.
   - **Occupancy (`#/occupancy`)**: Classroom spatial headcount and recognized vs unknown split.
   - **Infrastructure (`#/infrastructure`)**: Computer vision asset counters and zone filters.
   - **Compliance (`#/compliance`)**: Multi-rule discrepancy alerts and officer resolution portal.
   - **System (`#/system`)**: Watchdog hardware metrics and SQLite WAL offline sync queue.

---

## 7-Step Demonstration Sequence

### Step 1: Register Centre, Batch & Connect Video Feed
1. Ensure you are on the **Live Monitoring** view (`#/live`):
   - In the **Camera Configuration** panel:
     - Select **MP4 / Video File** (or Device/RTSP if presenting with live camera).
     - Enter `Camera ID`: `CAM-001`.
     - Enter `Room`: `Training Room 01`.
     - Select the bundled demo clip `camera/test_videos/classroom.mp4`.
     - Click **Connect Camera**.
2. **Result on Screen:**
   - The MJPEG preview displays live video with real-time detection bounding boxes.
   - Top KPI Cards reflect `CONNECTED` camera status with live FPS and frame age.
   - Header status pill turns green (`● Connected (Training Room 01)`).

---

### Step 2: Normal Classroom Compliance Baseline
1. Click the **Attendance** tab in the top navigation bar (`#/attendance`):
   - Classroom ID: `ROOM-01`
   - Batch ID: `Batch-A-2026`
   - Set Date: Today's date
   - Time: `09:00` to `11:00`
   - Click **Create Session**, then click **Start Session**.
2. **Result on Screen:**
   - Session status changes to `ACTIVE`.
   - SFace automatically identifies confirmed students and registers attendance without human intervention.
   - Navigate to **Occupancy** (`#/occupancy`) to view the headcount matching classroom presence.
   - In **Compliance** (`#/compliance`), alerts list shows: `No compliance alerts. Classroom operations are within sanctioned thresholds.`

---

### Step 3: Trigger Attendance Discrepancy Alert
1. **Scenario:** The official biometric portal reports 20 students present, but computer vision observes only 2 people in the room.
2. In the **Attendance** tab (`#/attendance`), use **Manual Attendance Correction** or submit an attendance record exceeding observed occupancy.
3. Switch to the **Compliance** tab (`#/compliance`) and click **Evaluate Rules Now** (or wait for the automatic persistence interval of 30 seconds).
4. **Result on Screen:**
   - The red notification pill on the navigation tab increments dynamically.
   - A `MEDIUM` severity alert appears: **Attendance discrepancy (`attendance_discrepancy`)**.
   - Details display: *Scope: `global` · Confidence: `90%` · Repeated evidence: `3`*.
   - Evaluators observe that the system does not declare fraud; it flags a discrepancy for inspector verification.

---

### Step 4: Infrastructure & Equipment Gap Observation
1. Click the **Infrastructure** tab (`#/infrastructure`).
2. Observe current computer vision counts (chairs, tables, laptops, displays).
3. Click **Test Current Frame** to inspect bounding boxes and confidence scores.
4. Adjust the zone filter or review temporal counts.
5. **Result on Screen:**
   - Infrastructure observation engine registers a temporal change.
   - An alert `infrastructure_observation_gap` is recorded without relying on rigid arbitrary static quotas.

---

### Step 5: Camera Failure / Quality Degradation vs Compliance Violation
1. Navigate to **Live Monitoring** (`#/live`) and click **Disconnect Camera** (or simulate video drop).
2. Switch to **System** tab (`#/system`):
3. **Result on Screen:**
   - Camera Health diagnostic table immediately flags `OFFLINE` or `DEGRADED`.
   - A `HIGH` severity alert is raised: **Camera health (`camera_health`)** with alert counter badge visible on the **Compliance** nav tab.
   - **Key Evaluator Pitch:** The system distinguishes between a technical feed error and an attendance fraud event. A broken camera creates a maintenance verification alert, NOT an automatic penal violation.

---

### Step 6: Offline Event Queue & Delayed Synchronization
1. Reconnect the camera from the **Live Monitoring** (`#/live`) tab.
2. Switch to the **System** tab (`#/system`):
   - Inspect the **Offline Event Queue Telemetry** table:
     - Displays `Pending: X · Syncing: 0 · Synced: Y · Failed: Z`.
     - While disconnected or when no central internet is present, local AI inference continues uninterrupted with zero data loss.
   - Click **Sync Queued Events Now** (or let background worker flush backlog).
   - Events are delivered with idempotency keys and exponential backoff retry.

---

### Step 7: Officer Review, Resolution & Audit History
1. Switch to the **Compliance** tab (`#/compliance`):
   - Click **Resolve Alert** next to any active alert.
   - Enter reason: `Physical verification conducted by District Officer. Equipment verified.`
   - Click OK.
2. **Result on Screen:**
   - Alert status transitions to `RESOLVED` with the officer's name, resolution reason, and timestamp.
   - The action is appended immutably to the **Recent Compliance Events & Audit Trail** table.
   - The navigation badge count automatically decrements.
   - No data is deleted; full evidentiary integrity is maintained for administrative audit.

---

## 30-Second Closing Pitch for Judges

> *"SkillWatch AI does not replace inspectors — it helps them know where to look first, using privacy-preserving evidence generated directly at the training centre."*
