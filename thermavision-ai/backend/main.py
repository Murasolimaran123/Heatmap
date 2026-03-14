"""
ThermaVision AI — FastAPI Backend
Exposes REST + WebSocket endpoints for real-time thermal heatmap processing.
"""
import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from heatmap_engine import HeatmapEngine
from detection_engine import DetectionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global AI Engine Instances ---
heatmap_engine = HeatmapEngine()
detection_engine = DetectionEngine()

# --- Analytics Store (in-memory, per session) ---
analytics_store = {
    "frames_processed": 0,
    "average_intensity": 0.0,
    "people_count": 0,
    "intensity_history": [],   # last 60 values
    "people_history": [],      # last 60 values
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ThermaVision AI Backend started.")
    yield
    logger.info("ThermaVision AI Backend shutting down.")


app = FastAPI(
    title="ThermaVision AI",
    description="Real-time AI-powered thermal heatmap generation engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------
# REST ENDPOINTS
# -----------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - analytics_store["start_time"])}


@app.get("/analytics")
async def get_analytics():
    return {
        "frames_processed": analytics_store["frames_processed"],
        "average_intensity": round(analytics_store["average_intensity"], 2),
        "people_count": analytics_store["people_count"],
        "intensity_history": analytics_store["intensity_history"][-60:],
        "people_history": analytics_store["people_history"][-60:],
    }


@app.post("/reset")
async def reset_engine():
    heatmap_engine.reset()
    analytics_store["frames_processed"] = 0
    analytics_store["intensity_history"] = []
    analytics_store["people_history"] = []
    return {"status": "reset"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), mode: str = "thermal", colormap: str = "jet"):
    """
    Accept an uploaded video file, process each frame, and stream back
    base64-encoded processed frames as NDJSON (newline-delimited JSON).
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are accepted.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)

    # Write to a temp buffer OpenCV can read
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    async def frame_generator():
        cap = cv2.VideoCapture(tmp_path)
        engine = HeatmapEngine()  # Fresh engine per upload
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                processed = _process_frame(frame, mode, colormap, engine)
                _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf).decode("utf-8")
                payload = json.dumps({
                    "frame": b64,
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                    "progress": round(frame_idx / max(total_frames, 1) * 100, 1),
                })
                yield payload + "\n"
                await asyncio.sleep(0)  # yield to event loop
        finally:
            cap.release()
            os.unlink(tmp_path)

    return StreamingResponse(frame_generator(), media_type="application/x-ndjson")


# -----------------------------------------------
# WEBSOCKET ENDPOINT
# -----------------------------------------------

@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """
    Real-time camera WebSocket.
    Client sends: JSON { "frame": "<base64 JPEG>", "mode": "thermal|motion|detection|crowd|normal", "colormap": "jet|inferno|turbo" }
    Server returns: JSON { "frame": "<base64 JPEG>", "intensity": float, "people_count": int, "detections": [...] }
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")
    heatmap_engine.reset()

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            b64_frame = payload.get("frame", "")
            mode = payload.get("mode", "thermal")
            colormap = payload.get("colormap", "jet")

            if not b64_frame:
                continue

            try:
                frame = heatmap_engine.decode_frame(b64_frame)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Decode error: {e}"}))
                continue

            # Process frame
            detections_data = []
            people_count = 0
            processed = _process_frame(frame, mode, colormap, heatmap_engine)

            # Run detection if mode requires it
            if mode in ("detection", "crowd"):
                detection_result = detection_engine.process_frame_detection(frame, processed)
                detections_data = detection_result["detections"]
                people_count = detection_result["people_count"]
                processed = detection_result["annotated_frame"]
                # Boost heatmap where detections are
                if detection_result["heat_regions"]:
                    heatmap_engine.add_extra_heat(frame, detection_result["heat_regions"])

            # Update analytics
            intensity = heatmap_engine.get_average_intensity(frame)
            _update_analytics(intensity, people_count)

            # Encode result
            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result_b64 = base64.b64encode(buf).decode("utf-8")

            response = {
                "frame": result_b64,
                "intensity": round(intensity, 2),
                "people_count": people_count,
                "frames_processed": analytics_store["frames_processed"],
                "detections": [
                    {"label": d["label"], "confidence": d["confidence"]}
                    for d in detections_data[:10]
                ],
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# -----------------------------------------------
# HELPERS
# -----------------------------------------------

def _process_frame(
    frame: np.ndarray,
    mode: str,
    colormap: str,
    engine: HeatmapEngine,
) -> np.ndarray:
    """Dispatch to the correct processing pipeline based on mode."""
    if mode == "normal":
        return engine.process_normal(frame)
    elif mode == "motion":
        return engine.process_motion(frame, colormap)
    elif mode == "thermal":
        return engine.process_thermal(frame, colormap)
    elif mode in ("detection", "crowd"):
        # Thermal base with detection overlay added later
        return engine.process_thermal(frame, colormap)
    else:
        return engine.process_thermal(frame, colormap)


def _update_analytics(intensity: float, people_count: int):
    analytics_store["frames_processed"] += 1
    n = analytics_store["frames_processed"]
    prev_avg = analytics_store["average_intensity"]
    analytics_store["average_intensity"] = (prev_avg * (n - 1) + intensity) / n
    analytics_store["people_count"] = people_count
    analytics_store["intensity_history"].append(round(intensity, 1))
    analytics_store["people_history"].append(people_count)
    if len(analytics_store["intensity_history"]) > 300:
        analytics_store["intensity_history"].pop(0)
        analytics_store["people_history"].pop(0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
