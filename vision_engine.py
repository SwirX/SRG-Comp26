"""
color_detector.py — Optimized computer vision pipeline for SRG-Comp26.

Detects colored objects (cubes/cylinders), QR codes, and ArUco markers
in JPEG frames from ESP32-CAM. Pre-allocates buffers, reuses grayscale
conversions, and supports optional downscale for detection.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

log = logging.getLogger("Vision")

# ── Distance estimation ──────────────────────────────────────────────────────
# Tuned for a ~6cm object (cube/cylinder) in a typical hackathon setup.
# K = perceived_pixels_area_at_known_distance * known_distance²
# Adjust K_DISTANCE to match your actual objects + camera focal length.
# A value of ~40000 works for a 6cm object at ~40cm with ESP32-CAM resolution.
K_DISTANCE: float = 40_000.0   # px²·cm²  — tune this per setup

def estimate_distance_cm(area_px: float) -> Optional[float]:
    """Returns estimated distance in cm from bbox area (px²), or None if unreliable."""
    if area_px <= 0:
        return None
    d = K_DISTANCE / area_px ** 0.5
    return round(d, 1)


# ── Shape classifier ─────────────────────────────────────────────────────────
def classify_shape(cnt: np.ndarray) -> str:
    """
    Returns 'cylinder' or 'cube' based on contour shape.
    Circularity > 0.78 → cylinder, else → cube.
    """
    area = cv2.contourArea(cnt)
    if area == 0:
        return "cube"
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "cube"
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return "cylinder" if circularity > 0.78 else "cube"


# ── Color maps ───────────────────────────────────────────────────────────────
# color name → protocol ObjColor enum value
COLOR_ID_MAP = {"red": 0, "green": 1, "yellow": 2}

COLOR_RANGES = {
    "red": [
        (np.array([0,   100, 80], dtype=np.uint8), np.array([8,   255, 255], dtype=np.uint8)),
        (np.array([170, 100, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
    ],
    "green": [
        (np.array([45, 65, 30], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
    ],
    "yellow": [
        (np.array([22, 55, 50], dtype=np.uint8), np.array([34, 255, 255], dtype=np.uint8)),
    ],
}

DRAW_COLOR = {
    "red":    (0, 0, 220),
    "green":  (0, 210, 0),
    "yellow": (0, 220, 220),
    "code":   (255, 200, 0),
    "aruco":  (0, 255, 255),
    "reflection": (255, 100, 255),
}


# ── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ColorBlob:
    color: str
    color_id: int           # maps to protocol.ObjColor
    shape: str              # 'cube' or 'cylinder'
    center: Tuple[int, int]
    norm_x: float
    norm_y: float
    area: float
    area_pct: float
    bbox: Tuple[int, int, int, int]
    distance_cm: Optional[float]


@dataclass(slots=True)
class CodeResult:
    kind: str
    data: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


@dataclass(slots=True)
class ArUcoResult:
    id: int
    center: Tuple[int, int]
    corners: np.ndarray
    bbox: Tuple[int, int, int, int]


# Max pixel distance between a blob center and ArUco center to form a pair.
ARUCO_PAIR_MAX_DIST: float = 200.0


@dataclass
class DetectionResult:
    frame: np.ndarray
    blobs: List[ColorBlob] = field(default_factory=list)
    codes: List[CodeResult] = field(default_factory=list)
    aruco: List[ArUcoResult] = field(default_factory=list)
    annotated: Optional[np.ndarray] = None
    process_ms: float = 0.0
    # Pairs formed this frame: (aruco_id, color) — may contain duplicates across frames.
    aruco_color_pairs: List[Tuple[int, str]] = field(default_factory=list)


# ── Detector ─────────────────────────────────────────────────────────────────
class VisionDetector:
    """
    Optimized vision pipeline with color detection, shape classification,
    distance estimation, ArUco, and QR.

    Parameters
    ----------
    max_objects : int
        Hard cap on the total number of ColorBlob detections returned per frame.
        Most-central blob wins when trimming. 0 = unlimited.
    """

    def __init__(
        self,
        min_area: int = 800,
        blur_ksize: int = 5,
        morph_ksize: int = 7,
        draw_overlay: bool = True,
        detect_scale: float = 1.0,
        max_objects: int = 10,          # ← limit detections per frame
        on_detection: Optional[Callable[["DetectionResult"], None]] = None,
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
    ):
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.draw_overlay = draw_overlay
        self.detect_scale = detect_scale
        self.max_objects = max_objects
        self.on_detection = on_detection

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

        # pre-allocated buffers (lazily sized on first frame)
        self._mask: Optional[np.ndarray] = None
        self._gray: Optional[np.ndarray] = None
        self._hsv: Optional[np.ndarray] = None
        self._blurred: Optional[np.ndarray] = None
        self._buf_h = 0
        self._buf_w = 0

    def _ensure_buffers(self, h: int, w: int):
        if h == self._buf_h and w == self._buf_w:
            return
        self._buf_h, self._buf_w = h, w
        self._mask    = np.zeros((h, w), dtype=np.uint8)
        self._gray    = np.zeros((h, w), dtype=np.uint8)
        self._hsv     = np.zeros((h, w, 3), dtype=np.uint8)
        self._blurred = np.zeros((h, w, 3), dtype=np.uint8)

    def process(self, frame: np.ndarray) -> DetectionResult:
        t0 = cv2.getTickCount()

        work = frame
        scale = self.detect_scale
        if scale < 1.0:
            sh, sw = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
            work = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        h, w = work.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        total_area = frame_h * frame_w
        self._ensure_buffers(h, w)

        cv2.GaussianBlur(work, (self.blur_ksize, self.blur_ksize), 0, dst=self._blurred)
        cv2.cvtColor(self._blurred, cv2.COLOR_BGR2HSV, dst=self._hsv)
        cv2.cvtColor(work, cv2.COLOR_BGR2GRAY, dst=self._gray)

        blobs  = self._detect_colors(self._hsv, frame_h, frame_w, scale, total_area)
        arucos = self._detect_aruco(self._gray, scale)
        codes  = self._detect_codes(self._gray, scale) if PYZBAR_AVAILABLE else []

        # enforce max_objects cap (keep largest area blobs first)
        if self.max_objects > 0 and len(blobs) > self.max_objects:
            blobs.sort(key=lambda b: b.area, reverse=True)
            blobs = blobs[:self.max_objects]

        pairs = self._pair_blobs_to_aruco(blobs, arucos)

        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        result = DetectionResult(
            frame=frame, blobs=blobs, codes=codes,
            aruco=arucos, process_ms=elapsed,
            aruco_color_pairs=pairs,
        )

        if self.draw_overlay:
            result.annotated = self._draw(frame.copy(), blobs, codes, arucos, pairs)

        if self.on_detection:
            self.on_detection(result)

        return result

    def _detect_colors(self, hsv: np.ndarray, fh: int, fw: int,
                       scale: float, total_area: int) -> List[ColorBlob]:
        blobs = []
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0
        scaled_min = self.min_area * (scale * scale) if scale < 1.0 else self.min_area

        for name, ranges in COLOR_RANGES.items():
            # build mask — handle multi-range colors (e.g. red wraps hue)
            self._mask[:] = 0
            if len(ranges) == 1:
                cv2.inRange(hsv, ranges[0][0], ranges[0][1], dst=self._mask)
            else:
                tmp = np.zeros_like(self._mask)
                for lo, hi in ranges:
                    cv2.inRange(hsv, lo, hi, dst=tmp)
                    cv2.bitwise_or(self._mask, tmp, dst=self._mask)

            cv2.morphologyEx(self._mask, cv2.MORPH_OPEN,  self._morph_kernel, dst=self._mask)
            cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, self._morph_kernel, dst=self._mask)

            contours, _ = cv2.findContours(
                self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < scaled_min:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if scale < 1.0:
                    cx  = int(cx * inv_scale)
                    cy  = int(cy * inv_scale)
                    area *= (inv_scale * inv_scale)
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    x  = int(x  * inv_scale)
                    y  = int(y  * inv_scale)
                    bw = int(bw * inv_scale)
                    bh = int(bh * inv_scale)
                else:
                    x, y, bw, bh = cv2.boundingRect(cnt)

                shape = classify_shape(cnt)
                dist  = estimate_distance_cm(area)

                blobs.append(ColorBlob(
                    color=name,
                    color_id=COLOR_ID_MAP.get(name, 4),
                    shape=shape,
                    center=(cx, cy),
                    norm_x=cx / fw,
                    norm_y=cy / fh,
                    area=area,
                    area_pct=(area / total_area) * 100.0,
                    bbox=(x, y, bw, bh),
                    distance_cm=dist,
                ))
        return blobs

    def _detect_aruco(self, gray: np.ndarray, scale: float) -> List[ArUcoResult]:
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None:
            return []

        inv = 1.0 / scale if scale < 1.0 else 1.0
        results = []
        for i in range(len(ids)):
            c = corners[i].reshape((4, 2))
            if scale < 1.0:
                c = c * inv
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            results.append(ArUcoResult(
                int(ids[i][0]), (cx, cy), c,
                cv2.boundingRect(corners[i].astype(np.int32) if scale >= 1.0
                                 else (corners[i] * inv).astype(np.int32)),
            ))
        return results

    def _detect_codes(self, gray: np.ndarray, scale: float) -> List[CodeResult]:
        inv = 1.0 / scale if scale < 1.0 else 1.0
        codes = []
        for obj in pyzbar.decode(gray):
            x, y, w, h = obj.rect
            if scale < 1.0:
                x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
            codes.append(CodeResult(
                kind=obj.type,
                data=obj.data.decode("utf-8", errors="replace"),
                center=(x + w // 2, y + h // 2),
                bbox=(x, y, w, h),
            ))
        return codes

    def _pair_blobs_to_aruco(
        self,
        blobs: List[ColorBlob],
        arucos: List[ArUcoResult],
    ) -> List[Tuple[int, str]]:
        """Return (aruco_id, color) for each blob whose nearest ArUco is within ARUCO_PAIR_MAX_DIST."""
        if not blobs or not arucos:
            return []
        pairs: List[Tuple[int, str]] = []
        for blob in blobs:
            bx, by = blob.center
            best_dist = float("inf")
            best_id   = -1
            for a in arucos:
                ax, ay = a.center
                d = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_id   = a.id
            if best_dist <= ARUCO_PAIR_MAX_DIST:
                pairs.append((best_id, blob.color))
        return pairs

    def _draw(self, frame: np.ndarray, blobs, codes, arucos,
              pairs: Optional[List[Tuple[int, str]]] = None) -> np.ndarray:
        # build a quick lookup: aruco_id → color for overlay
        pair_map: Dict[int, str] = {aid: col for aid, col in (pairs or [])}

        for b in blobs:
            c = DRAW_COLOR.get(b.color, (200, 200, 200))
            x, y, w, h = b.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)

            dist_str = f" ~{b.distance_cm}cm" if b.distance_cm is not None else ""
            label = f"{b.color} {b.shape}{dist_str} {b.area_pct:.1f}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

        for cd in codes:
            c = DRAW_COLOR["code"]
            x, y, w, h = cd.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
            cv2.putText(frame, f"CODE:{cd.data[:12]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        for a in arucos:
            c = DRAW_COLOR["aruco"]
            x, y, w, h = a.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
            paired_col = pair_map.get(a.id)
            label = f"ID:{a.id}" + (f"={paired_col}" if paired_col else "")
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        return frame

    def process_reflections(self, diff: np.ndarray, original_frame: np.ndarray) -> DetectionResult:
        """
        Process the subtracted image (flash_on - flash_off) to find strong reflections.
        Classifies objects as cylinders (vertical stripes) or cubes (flat/wider reflections).
        """
        t0 = cv2.getTickCount()
        
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 80, 255, cv2.THRESH_BINARY)
        
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, dst=mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        fh, fw = original_frame.shape[:2]
        total_area = fh * fw

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area * 0.2: # Reflections can be smaller than the whole object
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / float(max(w, 1))
            
            # Cylinders create vertical bright stripes (tall and narrow)
            # Cubes have flatter or wider reflections
            shape = "cylinder (refl)" if aspect_ratio > 1.8 else "cube (refl)"
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dist = estimate_distance_cm(area * 3) # Roughly adjusting area since reflection is smaller
            
            blobs.append(ColorBlob(
                color="reflection",
                color_id=99,
                shape=shape,
                center=(cx, cy),
                norm_x=cx / fw,
                norm_y=cy / fh,
                area=area,
                area_pct=(area / total_area) * 100.0,
                bbox=(x, y, w, h),
                distance_cm=dist,
            ))
            
        elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        result = DetectionResult(
            frame=original_frame, 
            blobs=blobs, 
            process_ms=elapsed,
        )

        if self.draw_overlay:
            result.annotated = self._draw(original_frame.copy(), blobs, [], [])

        if self.on_detection:
            self.on_detection(result)

        return result


if __name__ == "__main__":
    detector = VisionDetector(max_objects=5)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = detector.process(frame)
        cv2.imshow("Test", res.annotated if res.annotated is not None else frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()