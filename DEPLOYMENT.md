# SkillWatch AI — Cloud Deployment Architecture & Live Guide

This guide details the complete production deployment setup for **SkillWatch AI**:
- **Frontend:** Vercel (Global Edge CDN / SPA Static Hosting)
- **Backend:** Render (FastAPI Web Service on CPU / Linux)
- **Database:** Render PostgreSQL (Free Tier Managed SQL Database with local SQLite fallback)

---

## 1. Live Deployment Links

| Component | Platform | Live URL / Claim Link | Status |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Vercel** | [https://rapid-willow-u744tjv.tmp.now](https://rapid-willow-u744tjv.tmp.now) <br> *(Claim & link to your Vercel account: [Claim Deployment](https://vercel.com/claim-deployment?code=4298c60a-88b7-4262-87a7-e041339f91a4))* | Active |
| **Backend** | **Render** | `https://skillwatch-backend.onrender.com` <br> *(Deploy via Render Blueprint: [Render Dashboard](https://dashboard.render.com/blueprints))* | Ready to launch |
| **Database** | **PostgreSQL** | Managed Render Postgres (`skillwatch-db`) via `DATABASE_URL` | Configured |
| **Source Repo** | **GitHub** | [https://github.com/vibeee45/SkillWatchAI](https://github.com/vibeee45/SkillWatchAI) | Synchronized |

---

## 2. Cloud Architecture Diagram

```
+─────────────────────────────────────────────────────────────────────────────+
|                             USER BROWSER / OFFICER                          |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
                    HTTPS Requests     │
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                         VERCEL (Global Edge CDN)                            |
|                                                                             |
|  - Serves multipage SPA frontend (`index.html`, `logo.png`, SVGs)           |
|  - Automatic Reverse-Proxy Rewrites: `/api/:path*` ────────┐                |
+────────────────────────────────────────────────────────────┼────────────────+
                                                             │
                                                             │ HTTPS API / CORS
                                                             ▼
+─────────────────────────────────────────────────────────────────────────────+
|                         RENDER (FastAPI Web Service)                        |
|                                                                             |
|  - Python 3.12 / Debian Linux runtime (Free Tier)                           |
|  - Pre-packaged ONNX AI Models (YuNet 0.22MB, SFace 36.9MB, YOLOv8n 12.3MB) |
|  - Uvicorn ASGI Server on `0.0.0.0:$PORT`                                   |
|  - Endpoints: Camera, Attendance, Students, Occupancy, Infra, Compliance    |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
                                       │ SQL Connection (psycopg)
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                    RENDER POSTGRESQL (Free Managed DB)                      |
|                                                                             |
|  - Tables: Training Centres, Batches, Consented Students, Attendance Logs  |
|  - Zero-maintenance automatic connection via `DATABASE_URL`                 |
|  - Fallback: Local SQLite WAL (`data/skillwatch.db`)                        |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 3. 1-Click Deployment Instructions

### Deploying Backend & Database on Render (2 Minutes)

1. Go to **[https://dashboard.render.com/blueprints](https://dashboard.render.com/blueprints)**.
2. Click **"New Blueprint Instance"**.
3. Select your repository: **`vibeee45/SkillWatchAI`**.
4. Render will automatically detect [`render.yaml`](file:///c:/Users/91958/Desktop/SkillWatchAI/render.yaml):
   - It automatically provisions:
     - **`skillwatch-db`**: Free PostgreSQL database.
     - **`skillwatch-backend`**: Free Web Service running Python 3.12 with headless OpenCV and pre-bundled ONNX models.
5. Click **"Apply"**.
6. Once deployed, Render will provide your public backend URL:  
   👉 `https://skillwatch-backend.onrender.com` (or your chosen service name).

---

### Deploying Frontend on Vercel (1 Minute)

#### Option A: 1-Click Permanent Link to Your Vercel Account
Click your unique claim link:  
👉 **[Claim Vercel Deployment](https://vercel.com/claim-deployment?code=4298c60a-88b7-4262-87a7-e041339f91a4)**  
This links the live deployment directly to your personal or team Vercel account with a permanent domain (e.g. `https://skillwatch-ai.vercel.app`).

#### Option B: Deploy from GitHub Repo
1. Go to **[https://vercel.com/new](https://vercel.com/new)**.
2. Import **`vibeee45/SkillWatchAI`**.
3. In **Root Directory**, enter: `frontend` (or leave as root; both have valid `vercel.json` files).
4. Click **"Deploy"**.

---

## 4. Connecting Frontend to Backend

SkillWatch AI includes a **Dynamic Cloud Backend Resolver**:

1. Open your deployed Vercel frontend.
2. Navigate to the **System** tab (`#/system`).
3. Under **Cloud Deployment & Backend API Endpoint**:
   - Paste your Render backend URL (e.g., `https://skillwatch-backend.onrender.com`).
   - Click **Save Backend URL**.
4. The dashboard will immediately communicate with your live Render backend, and the setting persists in `localStorage` across refreshes.

---

## 5. Security & Production Hardening Features Added

- **CORS Middleware:** Configured in `backend/main.py` allowing cross-origin requests from Vercel (`*.vercel.app`) and custom domains.
- **Render PostgreSQL Compatibility:** Auto-detects and converts Render's `postgres://` connection string format to SQLAlchemy-compliant `postgresql://`.
- **Headless OpenCV:** `render_build.sh` uses `opencv-contrib-python-headless` to eliminate any missing `libGL.so.1` or X11 dependencies on Linux cloud servers.
- **Zero-Download Models:** All 3 ONNX AI models (`yunet`, `sface`, `yolov8n`) are tracked directly in git, eliminating network failure risks during build time.
