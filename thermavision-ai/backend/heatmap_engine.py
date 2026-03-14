import cv2
import numpy as np
from typing import Optional
import base64

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
}


class HeatmapEngine:
    """
    Core heatmap processing pipeline.
    Processes individual frames and converts them to thermal heatmap visualizations.
    Maintains state for frame differencing (motion detection).
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.accumulated_heat: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.decay_factor: float = 0.85  # Heat fades over time

    def decode_frame(self, b64_data: str) -> np.ndarray:
        """Decode a base64-encoded JPEG frame into a numpy BGR array."""
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame

    def encode_frame(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode a numpy BGR frame to a base64 JPEG string."""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _apply_gaussian_blur(self, gray: np.ndarray, ksize: int = 21) -> np.ndarray:
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)

    def _normalize(self, mat: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(mat, dtype=np.float32)
        cv2.normalize(mat.astype(np.float32), normalized, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _compute_motion_map(self, gray: np.ndarray) -> np.ndarray:
        """Compute absolute frame difference for motion detection."""
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray)

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray.copy()

        # Blur the diff to spread the motion region
        diff_blurred = cv2.GaussianBlur(diff, (31, 31), 0)
        return diff_blurred

    def _update_accumulated_heat(self, heat_map: np.ndarray) -> np.ndarray:
        """Accumulate heat over frames with temporal decay. Creates a persistence effect."""
        if self.accumulated_heat is None or self.accumulated_heat.shape != heat_map.shape:
            self.accumulated_heat = np.zeros_like(heat_map, dtype=np.float32)

        # Decay old heat and add new
        self.accumulated_heat = self.accumulated_heat * self.decay_factor + heat_map.astype(np.float32) * (1 - self.decay_factor)
        clipped = np.clip(self.accumulated_heat, 0, 255).astype(np.uint8)
        return clipped

    def _apply_colormap(self, gray_map: np.ndarray, colormap_name: str = "jet") -> np.ndarray:
        colormap_id = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
        return cv2.applyColorMap(gray_map, colormap_id)

    def _overlay_heatmap(self, original: np.ndarray, heatmap_colored: np.ndarray, alpha: float = 0.65) -> np.ndarray:
        """Blend the heatmap over the original frame."""
        original_resized = cv2.resize(original, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        blended = cv2.addWeighted(heatmap_colored, alpha, original_resized, 1 - alpha, 0)
        return blended

    def add_extra_heat(self, frame: np.ndarray, regions: list, intensity: float = 1.5):
        """
        Add heat blobs at specific regions (used by detection engine to increase
        heat around detected objects/people).
        regions: list of (x, y, w, h) bounding boxes
        """
        if self.accumulated_heat is None:
            h, w = frame.shape[:2]
            self.accumulated_heat = np.zeros((h, w), dtype=np.float32)

        for (x, y, w, h) in regions:
            cx, cy = x + w // 2, y + h // 2
            heat_blob = np.zeros_like(self.accumulated_heat)
            cv2.circle(heat_blob, (cx, cy), max(w, h) // 2, 200, -1)
            blob_blurred = cv2.GaussianBlur(heat_blob, (51, 51), 0)
            self.accumulated_heat = np.clip(
                self.accumulated_heat + blob_blurred * intensity, 0, 255
            )

    def process_thermal(self, frame: np.ndarray, colormap: str = "jet") -> np.ndarray:
        """Full thermal heatmap mode: brightness-based colorization."""
        gray = self._to_grayscale(frame)
        blurred = self._apply_gaussian_blur(gray, ksize=21)
        normalized = self._normalize(blurred)
        accumulated = self._update_accumulated_heat(normalized)
        colored = self._apply_colormap(accumulated, colormap)
        result = self._overlay_heatmap(frame, colored, alpha=0.75)
        return result

    def process_motion(self, frame: np.ndarray, colormap: str = "jet") -> np.ndarray:
        """Motion heatmap mode: frame-differencing-based colorization."""
        gray = self._to_grayscale(frame)
        blurred = self._apply_gaussian_blur(gray, ksize=15)
        motion_map = self._compute_motion_map(blurred)
        normalized = self._normalize(motion_map)
        accumulated = self._update_accumulated_heat(normalized)
        colored = self._apply_colormap(accumulated, colormap)
        result = self._overlay_heatmap(frame, colored, alpha=0.70)
        return result

    def process_normal(self, frame: np.ndarray) -> np.ndarray:
        """Passthrough — return original frame."""
        return frame.copy()

    def get_average_intensity(self, frame: np.ndarray) -> float:
        """Returns average pixel intensity across the frame (0-100)."""
        gray = self._to_grayscale(frame)
        return float(np.mean(gray)) / 255.0 * 100

    def reset(self):
        """Reset accumulated state (use when source changes)."""
        self.prev_gray = None
        self.accumulated_heat = None
        self.frame_count = 0
