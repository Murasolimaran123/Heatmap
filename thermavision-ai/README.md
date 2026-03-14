# ThermaVision AI

> Real-time AI-powered thermal heatmap visualization — no infrared hardware required.

A production-ready SaaS web app that converts your webcam or uploaded video into a **thermal heatmap** using OpenCV, YOLOv8, and MediaPipe — all running locally in your browser + a lightweight Python backend.

---

## 📁 Project Structure

```
thermavision-ai/
  backend/
    main.py               ← FastAPI app (REST + WebSocket)
    heatmap_engine.py     ← OpenCV thermal processing pipeline
    detection_engine.py   ← YOLOv8 + MediaPipe detection
    requirements.txt
  frontend/
    src/
      app/                ← Next.js App Router pages
        dashboard/        ← Overview dashboard
        live/             ← Live camera + heatmap
        upload/           ← Video file processing
        analytics/        ← Recharts analytics
        settings/         ← Config panel
      components/         ← Sidebar, TopNav, CameraView, ModeToggle
      hooks/              ← useCamera, useWebSocket
  install.bat
  start_backend.bat
  start_frontend.bat
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- A webcam

### Step 1 — Install once
```bat
install.bat
```

### Step 2 — Start the backend (Terminal 1)
```bat
start_backend.bat
```
Verify: `http://localhost:8000/health` → `{"status":"ok"}`

### Step 3 — Start the frontend (Terminal 2)
```bat
start_frontend.bat
```
Open: `http://localhost:3000`

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Live Thermal View** | Webcam → OpenCV COLORMAP_JET heatmap in real time |
| **Motion Detection** | Frame differencing highlights movement hot zones |
| **Object Detection** | YOLOv8 nano draws bounding boxes on detected people/objects |
| **Crowd Density** | People count + heat clustering |
| **Video Upload** | Process any video file through the thermal pipeline |
| **Recording** | Record + download thermal sessions as `.webm` |
| **Analytics** | Live charts: activity intensity, people count over time |
| **5 View Modes** | Normal / Thermal / Motion / Detection / Crowd |
| **3 Colormaps** | JET / INFERNO / TURBO / HOT |

---

## 🖥 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/analytics` | Session metrics |
| `POST` | `/reset` | Reset engine state |
| `POST` | `/upload` | Process uploaded video (NDJSON stream) |
| `WS` | `/ws/camera` | Real-time camera frame WebSocket |

---

## 🚀 Cloud Deployment

### Backend (FastAPI)
- Deploy to **Railway**, **Render**, or **Google Cloud Run**
- Set `PORT` env var and use `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Next.js)
- Deploy to **Vercel** (zero config)
- Set `NEXT_PUBLIC_WS_URL` and `NEXT_PUBLIC_API_URL` env vars pointing to your deployed backend

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Backend | Python FastAPI, Uvicorn |
| AI/Computer Vision | OpenCV, YOLOv8 (ultralytics), MediaPipe |
| Real-time Comms | WebSocket (native browser + FastAPI) |
| Camera | WebRTC `getUserMedia` |
