"""
Detection Engine — YOLO object detection + MediaPipe pose landmarks.
Provides bounding boxes and heat region contributions per frame.
"""
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
import base64
import logging

logger = logging.getLogger(__name__)

# Gracefully handle missing dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available. Object detection disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not available. Pose detection disabled.")


class DetectionEngine:
    """
    Wraps YOLOv8 nano and MediaPipe Pose to detect people and objects.
    Returns bounding boxes and heatmap heat regions per frame.
    """

    DETECTION_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
        63: "laptop", 67: "cell phone",
    }

    PERSON_CLASS_ID = 0

    def __init__(self):
        self.yolo_model = None
        self.pose_detector = None
        self.people_count_history: List[int] = []
        self._init_yolo()
        self._init_mediapipe()

    def _init_yolo(self):
        if not YOLO_AVAILABLE:
            return
        try:
            # Use nano model for speed — downloads on first run (~6MB)
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n loaded successfully.")
        except Exception as e:
            logger.error(f"YOLO init failed: {e}")
            self.yolo_model = None

    def _init_mediapipe(self):
        if not MEDIAPIPE_AVAILABLE:
            return
        try:
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0,  # Lightest model for real-time
            )
            logger.info("MediaPipe Pose loaded successfully.")
        except Exception as e:
            logger.error(f"MediaPipe Pose init failed: {e}")
            self.pose_detector = None

    def _run_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO inference. Returns list of detection dicts."""
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.4, iou=0.45)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.DETECTION_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = self.DETECTION_CLASSES[cls_id]
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                        "is_person": cls_id == self.PERSON_CLASS_ID,
                    })
            return detections
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    def _run_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run MediaPipe Pose. Returns list of landmark heat regions."""
        if self.pose_detector is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose_detector.process(rgb)
            regions = []
            if result.pose_landmarks:
                h, w = frame.shape[:2]
                for lm in result.pose_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    regions.append({"cx": cx, "cy": cy, "radius": 18})
            return regions
        except Exception as e:
            logger.error(f"MediaPipe pose error: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto the frame."""
        result = frame.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            color = (0, 255, 100) if det["is_person"] else (255, 160, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            label_text = f"{det['label']} {det['confidence']:.0%}"
            cv2.putText(
                result, label_text,
                (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        return result

    def generate_detection_heat_regions(self, detections: List[Dict]) -> List[tuple]:
        """
        Convert YOLO bounding boxes to (x, y, w, h) heat region tuples
        that the HeatmapEngine can use to add concentrated heat.
        People get 3x heat intensity; other objects get 1x.
        """
        regions = []
        for det in detections:
            x, y, w, h = det["bbox"]
            regions.append((x, y, w, h))
        return regions

    def count_people(self, detections: List[Dict]) -> int:
        count = sum(1 for d in detections if d["is_person"])
        self.people_count_history.append(count)
        if len(self.people_count_history) > 300:
            self.people_count_history.pop(0)
        return count

    def process_frame_detection(
        self, frame: np.ndarray, overlay_heatmap: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run full detection pipeline on a frame.
        Returns detections, pose regions, people count, and annotated frame.
        """
        detections = self._run_yolo(frame)
        pose_regions = self._run_pose(frame)
        people_count = self.count_people(detections)

        annotated = self.draw_detections(
            overlay_heatmap if overlay_heatmap is not None else frame,
            detections,
        )

        # Draw pose blobs
        for region in pose_regions:
            cv2.circle(annotated, (region["cx"], region["cy"]), region["radius"], (255, 200, 0), -1)

        heat_regions = self.generate_detection_heat_regions(detections)

        return {
            "detections": detections,
            "pose_regions": pose_regions,
            "people_count": people_count,
            "heat_regions": heat_regions,
            "annotated_frame": annotated,
        }
