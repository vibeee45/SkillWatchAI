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
   The unified KPI header bar will show the camera state, occupancy, attendance session, detected assets, and active compliance alerts.

---

## 7-Step Demonstration Sequence

### Step 1: Register Centre, Batch & Connect Video Feed
1. In the **Camera Configuration** card:
   - Select **MP4 / Video File** (or Device/RTSP if presenting with live camera).
   - Enter `Camera ID`: `CAM-001`.
   - Enter `Room`: `Training Room 01`.
   - Select the bundled demo clip `camera/test_videos/classroom.mp4`.
   - Click **Connect Camera**.
2. **Result on Screen:**
   - The MJPEG preview displays live video with real-time detection bounding boxes.
   - KPI Bar reflects `CONNECTED` camera status with live FPS and frame age.
   - Infrastructure card displays detected equipment (chairs, tables, tech).

---

### Step 2: Normal Classroom Compliance Baseline
1. In the **Automatic Attendance** card:
   - Classroom ID: `ROOM-01`
   - Batch ID: `Batch-A-2026`
   - Set Date: Today's date
   - Time: `09:00` to `11:00`
   - Click **Create Session**, then click **Start Session**.
2. **Result on Screen:**
   - Session status changes to `ACTIVE`.
   - SFace automatically identifies confirmed students and registers attendance without human intervention.
   - The KPI occupancy matches classroom presence.
   - Compliance alerts box shows: `✔ No compliance alerts. Classroom operations are within sanctioned thresholds.`

---

### Step 3: Trigger Attendance Discrepancy Alert
1. **Scenario:** The official biometric portal reports 20 students present, but computer vision observes only 2 people in the room.
2. In the Attendance card, use **Manual Attendance Correction** or submit an attendance record exceeding observed occupancy.
3. In the **Compliance Alerts** card, click **Evaluate Rules Now** (or wait for the automatic persistence interval of 30 seconds).
4. **Result on Screen:**
   - A `MEDIUM` severity alert appears: **Attendance discrepancy (`attendance_discrepancy`)**.
   - Details display: *Scope: `global` · Confidence: `90%` · Repeated evidence: `3`*.
   - Evaluators observe that the system does not declare fraud; it flags a discrepancy for inspector verification.

---

### Step 4: Infrastructure & Equipment Gap Observation
1. In the **Infrastructure & Equipment Monitoring** card, observe current counts (e.g., chairs, tables, laptops).
2. Click **Test Current Frame** to inspect bounding boxes and confidence scores.
3. Change the zone or filter to observe equipment departure.
4. **Result on Screen:**
   - Infrastructure observation engine registers a temporal change.
   - An alert `infrastructure_observation_gap` is recorded without relying on rigid arbitrary static quotas.

---

### Step 5: Camera Failure / Quality Degradation vs Compliance Violation
1. In the Camera card, click **Disconnect Camera** (or simulate video drop).
2. **Result on Screen:**
   - Camera Health immediately updates to `OFFLINE` or `DEGRADED`.
   - A `HIGH` severity alert is raised: **Camera health (`camera_health`)**.
   - **Key Evaluator Pitch:** The system distinguishes between a technical feed error and an attendance fraud event. A broken camera creates a maintenance verification alert, NOT an automatic penal violation.

---

### Step 6: Offline Event Queue & Delayed Synchronization
1. Reconnect the camera.
2. Notice the **Offline event queue** card in the preview panel:
   - Displays `Pending: X · Syncing: 0 · Synced: Y`.
   - While disconnected or when no central internet is present, local AI inference continues uninterrupted.
   - Click **Sync queued events** (or let background worker flush backlog).
   - Events are delivered with idempotency keys and exponential backoff retry.

---

### Step 7: Officer Review, Resolution & Audit History
1. In the **Compliance Alerts & Officer Review** card:
   - Click **Resolve Alert** next to any active alert.
   - Enter reason: `Physical verification conducted by District Officer. Equipment verified.`
   - Click OK.
2. **Result on Screen:**
   - Alert status transitions to `RESOLVED` with the officer's name, resolution reason, and timestamp.
   - The action is appended immutably to the **Recent Compliance Events & Audit Trail** table.
   - No data is deleted; full evidentiary integrity is maintained for administrative audit.

---

## 30-Second Closing Pitch for Judges

> *"SkillWatch AI does not replace inspectors — it helps them know where to look first, using privacy-preserving evidence generated directly at the training centre."*
