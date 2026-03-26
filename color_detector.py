"""
detector.py — Color blob + QR/Barcode detector.
Designed to plug into UDPVideoServer.on_frame callback.

Dependencies:
    pip install opencv-python numpy pyzbar

pyzbar also needs the system library:
    Ubuntu/Debian : sudo apt install libzbar0
    macOS        : brew install zbar
    Windows      : the pip wheel bundles it — no extra step needed
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logging.warning("pyzbar not installed — QR/barcode detection disabled. "
                    "Run: pip install pyzbar")

log = logging.getLogger("Detector")

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColorBlob:
    color:      str                    # "red" | "green" | "blue"
    center:     Tuple[int, int]        # (cx, cy) in pixels
    area:       float                  # contour area in px²
    bbox:       Tuple[int,int,int,int] # (x, y, w, h)
    confidence: float = 1.0           # future use


@dataclass
class CodeResult:
    kind:   str   # "QR" | "CODE128" | "EAN13" | etc.
    data:   str   # decoded string
    center: Tuple[int, int]
    bbox:   Tuple[int,int,int,int]   # bounding rect (x, y, w, h)


@dataclass
class DetectionResult:
    frame:       np.ndarray
    blobs:       List[ColorBlob]  = field(default_factory=list)
    codes:       List[CodeResult] = field(default_factory=list)
    annotated:   Optional[np.ndarray] = None   # frame with overlays drawn


# ─────────────────────────────────────────────────────────────────────────────
# HSV colour ranges  (hue is 0-179 in OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_RANGES = {
    "red": [
        # Red wraps around 0° so we need two ranges
        (np.array([0,   100, 80]),  np.array([8,  255, 255])),
        (np.array([170, 100, 80]),  np.array([179, 255, 255])),
    ],
    "green": [
        (np.array([40,  60, 60]),   np.array([85,  255, 255])),
    ],
    "blue": [
        (np.array([95,  80, 50]),   np.array([135, 255, 255])),
    ],
}

# Draw colours (BGR)
DRAW_COLOR = {
    "red":   (0,   0,   220),
    "green": (0,   210, 0  ),
    "blue":  (220, 80,  0  ),
    "code":  (255, 200, 0  ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class VisionDetector:
    """
    Frame-level detector. Call process(frame) to get a DetectionResult.

    Tunable parameters
    ------------------
    min_area        : ignore blobs smaller than this (px²)
    blur_ksize      : Gaussian blur kernel before HSV conversion (odd int)
    morph_ksize     : morphology kernel for noise removal
    draw_overlay    : if True, annotated frame is attached to result
    on_detection    : optional callback(DetectionResult) fired after each frame
    """

    def __init__(
        self,
        min_area:     int   = 800,
        blur_ksize:   int   = 5,
        morph_ksize:  int   = 7,
        draw_overlay: bool  = True,
        on_detection: Optional[Callable[[DetectionResult], None]] = None,
    ):
        self.min_area     = min_area
        self.blur_ksize   = blur_ksize
        self.morph_ksize  = morph_ksize
        self.draw_overlay = draw_overlay
        self.on_detection = on_detection

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )

        # Cache seen QR data to avoid spamming the callback
        self._seen_codes: dict = {}   # data → last_seen_time

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> DetectionResult:
        """Run full detection pipeline and return DetectionResult."""
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        blobs = self._detect_colors(hsv)
        codes = self._detect_codes(frame) if PYZBAR_AVAILABLE else []

        result = DetectionResult(frame=frame, blobs=blobs, codes=codes)

        if self.draw_overlay:
            result.annotated = self._draw(frame.copy(), blobs, codes)

        if self.on_detection:
            self.on_detection(result)

        return result

    # Convenience: use as a callback directly with UDPVideoServer.on_frame
    def __call__(self, frame: np.ndarray) -> DetectionResult:
        return self.process(frame)

    # ── Colour detection ──────────────────────────────────────────────────────

    def _detect_colors(self, hsv: np.ndarray) -> List[ColorBlob]:
        blobs = []
        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)

            # Clean up noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue
                M   = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx  = int(M["m10"] / M["m00"])
                cy  = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                blobs.append(ColorBlob(
                    color=color_name,
                    center=(cx, cy),
                    area=area,
                    bbox=(x, y, w, h),
                ))

        return blobs

    # ── QR / barcode detection ────────────────────────────────────────────────

    def _detect_codes(self, frame: np.ndarray) -> List[CodeResult]:
        codes = []
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar.decode(gray)

        for obj in decoded:
            data   = obj.data.decode("utf-8", errors="replace")
            kind   = obj.type                      # "QRCODE", "CODE128", etc.
            pts    = np.array(obj.polygon, dtype=np.int32)
            x, y, w, h = obj.rect
            cx     = x + w // 2
            cy     = y + h // 2
            codes.append(CodeResult(
                kind=kind, data=data,
                center=(cx, cy),
                bbox=(x, y, w, h),
            ))

        return codes

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def _draw(
        self,
        frame: np.ndarray,
        blobs: List[ColorBlob],
        codes: List[CodeResult],
    ) -> np.ndarray:

        for blob in blobs:
            color = DRAW_COLOR.get(blob.color, (255, 255, 255))
            x, y, w, h = blob.bbox
            cx, cy     = blob.center

            # Bounding rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Centre dot
            cv2.circle(frame, (cx, cy), 5, color, -1)
            # Label
            label = f"{blob.color}  {blob.area:.0f}px²"
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for code in codes:
            color = DRAW_COLOR["code"]
            x, y, w, h = code.bbox
            cx, cy     = code.center
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            label = f"[{code.kind}] {code.data[:30]}"
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test (uses webcam)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    detector = VisionDetector(draw_overlay=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No webcam found.")
        sys.exit(1)

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process(frame)

        for blob in result.blobs:
            print(f"  COLOR  {blob.color:6s}  area={blob.area:.0f}  center={blob.center}")
        for code in result.codes:
            print(f"  CODE   [{code.kind}]  data={code.data!r}")

        if result.annotated is not None:
            cv2.imshow("Detector", result.annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()