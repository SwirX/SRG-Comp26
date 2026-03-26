"""
udp_server.py — PC-side UDP server (v3).

New in this version
───────────────────
  • Auto-detects the local LAN IP and prints it so you can paste it into
    the ESP32 sketch — no more manual config.
  • Detection filter: DETECT_COLORS / DETECT_CODES (or ServerConfig fields)
    let you list exactly which colours and code types are reported/drawn.
    Everything else is silently ignored.
  • Handshake: on first contact from the ESP32 the server sends a CONFIG
    packet (resolution, framerate, JPEG quality).  The ESP32 must ACK;
    the server retransmits up to MAX_HS_RETRIES times.

Run:
    python udp_server.py

Dependencies:
    pip install opencv-python numpy pyzbar
"""

import socket
import struct
import threading
import time
import numpy as np
import cv2
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple
from collections import deque

# ── Optional detector import ──────────────────────────────────────────────────
try:
    from color_detector import VisionDetector, DetectionResult
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    logging.warning(
        "color_detector.py not found — RAW display mode. "
        "Rename new.py → color_detector.py and place it alongside this file."
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s"
)
log = logging.getLogger("UDPServer")


# ─────────────────────────────────────────────────────────────────────────────
# ① DETECTION FILTER  ← edit these two lines
#    Colour names : "red"  "green"  "blue"  (keys in COLOR_RANGES)
#    Code types   : "QRCODE"  "CODE128"  "EAN13"  etc.  (pyzbar strings)
#    Set to None  → allow everything in that category.
# ─────────────────────────────────────────────────────────────────────────────
DETECT_COLORS: Optional[List[str]] = ["red", "green", "blue"]   # or None
DETECT_CODES:  Optional[List[str]] = ["QRCODE"]                 # or None


# ─────────────────────────────────────────────────────────────────────────────
# ② CAMERA CONFIG  (sent to ESP32 during handshake)
#    Resolution IDs mirror esp_camera framesize_t enum:
#      5 = QVGA 320×240  |  6 = CIF 400×296   |  8 = VGA  640×480
#      9 = SVGA 800×600  | 10 = XGA 1024×768  | 13 = UXGA 1600×1200
# ─────────────────────────────────────────────────────────────────────────────
CAM_RESOLUTION: int = 8    # VGA 640×480
CAM_FRAMERATE:  int = 15   # fps  (1 – 30)
CAM_QUALITY:    int = 12   # JPEG quality  (0 = best, 63 = worst)


# ─────────────────────────────────────────────────────────────────────────────
# Protocol constants  (keep in sync with the ESP32 sketch)
# ─────────────────────────────────────────────────────────────────────────────
MAGIC_VIDEO     = b'\xAA\xBB'
MAGIC_CMD       = b'\xCC\xDD'
MAGIC_SENSOR    = b'\xEE\xFF'
MAGIC_HANDSHAKE = b'\x11\x22'   # PC → ESP32  (config request)
MAGIC_ACK       = b'\x33\x44'   # ESP32 → PC  (config confirmed)

MAX_PACKET     = 65507
FRAME_TIMEOUT  = 3.0    # seconds before stale partial frames are dropped
MAX_HS_RETRIES = 5
HS_RETRY_DELAY = 1.0    # seconds between retransmissions

# Command IDs (unchanged from previous version)
CMD_MOVE      = 0x01
CMD_STOP      = 0x02
CMD_SET_SPEED = 0x03
CMD_FOLLOW    = 0x04
CMD_TURN      = 0x05
CMD_ESTOP     = 0xFF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """
    Returns the machine's primary LAN IP — the address it uses to reach
    the outside world.  Does NOT actually send any traffic.
    Falls back to 127.0.0.1 if no network adapter is up.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotTelemetry:
    timestamp:  float = 0.0
    battery_mv: int   = 0
    imu_roll:   float = 0.0
    imu_pitch:  float = 0.0
    imu_yaw:    float = 0.0
    extra:      bytes = b''


@dataclass
class ServerConfig:
    host:            str   = "0.0.0.0"
    video_port:      int   = 5005       # ESP32 → PC
    cmd_port:        int   = 5006       # PC → ESP32
    frame_buffer:    int   = 4
    stats_interval:  float = 5.0
    show_window:     bool  = True
    run_detector:    bool  = True
    # Detection filter (None = use module-level DETECT_* constants)
    detect_colors:   Optional[List[str]] = None
    detect_codes:    Optional[List[str]] = None
    # Camera parameters sent during handshake
    cam_resolution:  int = CAM_RESOLUTION
    cam_framerate:   int = CAM_FRAMERATE
    cam_quality:     int = CAM_QUALITY


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

class UDPVideoServer:
    """
    Receives chunked JPEG frames from ESP32, (optionally) runs the filtered
    VisionDetector, shows an annotated window, and sends commands to the robot.

    Handshake flow
    ──────────────
      1. Server binds and waits.
      2. ESP32 sends its first UDP packet (any video chunk).
      3. Server records the ESP32 address and spawns _handshake_loop.
      4. _handshake_loop sends MAGIC_HANDSHAKE | res | fps | quality
         to the ESP32's cmd_port.  Retransmits every HS_RETRY_DELAY seconds.
      5. ESP32 applies the config and replies with MAGIC_ACK | res | fps | quality.
      6. Server marks handshake complete and stops retransmitting.
    """

    def __init__(self, config: ServerConfig = ServerConfig()):
        self.cfg      = config
        self._running = False

        # Resolve detection filter sets once at startup
        raw_colors = (config.detect_colors
                      if config.detect_colors is not None else DETECT_COLORS)
        raw_codes  = (config.detect_codes
                      if config.detect_codes  is not None else DETECT_CODES)

        self._allowed_colors: Optional[Set[str]] = (
            {c.lower() for c in raw_colors} if raw_colors is not None else None
        )
        self._allowed_codes: Optional[Set[str]] = (
            {c.upper() for c in raw_codes}  if raw_codes  is not None else None
        )
        log.info(
            "Detection filter — "
            f"colours: {'ALL' if self._allowed_colors is None else self._allowed_colors}  |  "
            f"codes:   {'ALL' if self._allowed_codes  is None else self._allowed_codes}"
        )

        # Auto-detect LAN IP
        self._local_ip = get_local_ip()
        log.info(f"Local LAN IP: {self._local_ip}")

        # ── Sockets ──────────────────────────────────────────────────────────
        self._rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._rx_sock.settimeout(1.0)

        self._tx_sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._robot_addr: Optional[Tuple[str, int]] = None

        # ── Handshake state ───────────────────────────────────────────────────
        self._hs_done    = False
        self._hs_lock    = threading.Lock()

        # ── Frame state ───────────────────────────────────────────────────────
        self._partial_frames: dict               = {}
        self._frame_buffer: deque                = deque(maxlen=config.frame_buffer)
        self._frame_lock                         = threading.Lock()
        self._latest_raw:    Optional[np.ndarray] = None
        self._display_frame: Optional[np.ndarray] = None

        # ── Telemetry ─────────────────────────────────────────────────────────
        self._telemetry  = RobotTelemetry()
        self._telem_lock = threading.Lock()

        # ── Detector ──────────────────────────────────────────────────────────
        self.detector = None
        if config.run_detector and DETECTOR_AVAILABLE:
            self.detector = VisionDetector(
                min_area=800,
                draw_overlay=True,
                on_detection=self._on_detection,
            )
            log.info("VisionDetector active.")
        else:
            log.info("Raw-display mode (no detector).")

        # ── External callbacks ────────────────────────────────────────────────
        self.on_frame:     Optional[Callable[[np.ndarray], None]]      = None
        self.on_telemetry: Optional[Callable[[RobotTelemetry], None]]  = None
        self.on_detection: Optional[Callable]                           = None

        # ── Stats ─────────────────────────────────────────────────────────────
        self._frames_rx       = 0
        self._bytes_rx        = 0
        self._display_fps     = 0.0
        self._last_stats_time = time.time()
        self._last_frame_time = 0.0

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def local_ip(self) -> str:
        """The LAN IP the ESP32 should use as SERVER_IP."""
        return self._local_ip

    @property
    def handshake_done(self) -> bool:
        with self._hs_lock:
            return self._hs_done

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._rx_sock.bind((self.cfg.host, self.cfg.video_port))
        log.info(
            f"Server ready — {self.cfg.host}:{self.cfg.video_port}\n"
            f"  >>> Set  const char* SERVER_IP = \"{self._local_ip}\";  "
            f"in your ESP32 sketch <<<"
        )
        self._running = True
        threading.Thread(target=self._rx_loop,      daemon=True, name="RX").start()
        threading.Thread(target=self._cleanup_loop, daemon=True, name="Cleanup").start()

        if self.cfg.show_window:
            self._window_loop()          # blocks — OpenCV must run on main thread
        else:
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        self.stop()

    def stop(self):
        self._running = False
        self._rx_sock.close()
        log.info("Server stopped.")

    # ── Frame / telemetry getters ─────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_raw.copy() if self._latest_raw is not None else None

    def get_telemetry(self) -> RobotTelemetry:
        with self._telem_lock:
            return self._telemetry

    # ── Command API ───────────────────────────────────────────────────────────

    def send_command(self, cmd_id: int, payload: bytes = b'') -> bool:
        if self._robot_addr is None:
            log.warning("No robot connected — command dropped.")
            return False
        pkt = MAGIC_CMD + struct.pack("!BH", cmd_id, len(payload)) + payload
        try:
            self._tx_sock.sendto(pkt, (self._robot_addr[0], self.cfg.cmd_port))
            return True
        except OSError as e:
            log.error(f"send_command failed: {e}")
            return False

    def cmd_stop(self):                   self.send_command(CMD_STOP)
    def cmd_estop(self):                  self.send_command(CMD_ESTOP)
    def cmd_move(self, vx, vy, vz=0.0):  self.send_command(CMD_MOVE,      struct.pack("!fff", vx, vy, vz))
    def cmd_turn(self, yaw_rate):         self.send_command(CMD_TURN,      struct.pack("!f",   yaw_rate))
    def cmd_follow(self, tx, ty, depth):  self.send_command(CMD_FOLLOW,    struct.pack("!fff", tx, ty, depth))
    def cmd_set_speed(self, speed):       self.send_command(CMD_SET_SPEED, struct.pack("!f",   speed))

    # ── Handshake ─────────────────────────────────────────────────────────────

    def _send_handshake(self):
        """
        Handshake packet: MAGIC_HANDSHAKE(2) | resolution_id(1) | fps(1) | quality(1)
        Sent to the ESP32's cmd_port.
        """
        pkt = MAGIC_HANDSHAKE + struct.pack(
            "BBB",
            self.cfg.cam_resolution,
            self.cfg.cam_framerate,
            self.cfg.cam_quality,
        )
        self._tx_sock.sendto(pkt, (self._robot_addr[0], self.cfg.cmd_port))
        log.info(
            f"[HS] Sent config → "
            f"resolution={self.cfg.cam_resolution}  "
            f"fps={self.cfg.cam_framerate}  "
            f"quality={self.cfg.cam_quality}"
        )

    def _handshake_loop(self):
        """Retransmit config until ACK or retry limit."""
        for attempt in range(1, MAX_HS_RETRIES + 1):
            with self._hs_lock:
                if self._hs_done:
                    return
            log.info(f"[HS] Attempt {attempt}/{MAX_HS_RETRIES} ...")
            self._send_handshake()
            time.sleep(HS_RETRY_DELAY)

        with self._hs_lock:
            if not self._hs_done:
                log.warning(
                    "[HS] No ACK received — ESP32 may not have the handshake handler. "
                    "Continuing in default camera mode."
                )

    def _handle_ack(self, data: bytes):
        if len(data) < 3:
            log.warning("[HS] ACK packet too short, ignoring.")
            return
        res, fps, qual = struct.unpack("BBB", data[:3])
        with self._hs_lock:
            self._hs_done = True
        log.info(
            f"[HS] ACK ✓ — ESP32 confirmed resolution={res} fps={fps} quality={qual}"
        )

    # ── RX loop ───────────────────────────────────────────────────────────────

    def _rx_loop(self):
        while self._running:
            try:
                data, addr = self._rx_sock.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 2:
                continue

            if self._robot_addr is None:
                self._robot_addr = addr
                log.info(f"ESP32 connected from {addr[0]}:{addr[1]}")
                threading.Thread(
                    target=self._handshake_loop, daemon=True, name="Handshake"
                ).start()

            magic = data[:2]
            if magic == MAGIC_VIDEO:
                self._handle_video(data[2:])
            elif magic == MAGIC_SENSOR:
                self._handle_sensor(data[2:])
            elif magic == MAGIC_ACK:
                self._handle_ack(data[2:])

            self._bytes_rx += len(data)
            self._maybe_stats()

    # ── Video ─────────────────────────────────────────────────────────────────

    def _handle_video(self, data: bytes):
        """
        Packet layout (after magic):
          seq_id(uint16) | chunk_id(uint8) | total_chunks(uint8) | JPEG bytes
        """
        if len(data) < 4:
            return
        seq_id, chunk_id, total_chunks = struct.unpack("!HBB", data[:4])
        payload = data[4:]

        pf = self._partial_frames.setdefault(seq_id, {
            "total": total_chunks, "chunks": {}, "ts": time.time()
        })
        pf["chunks"][chunk_id] = payload

        if len(pf["chunks"]) == pf["total"]:
            jpeg = b''.join(pf["chunks"][i] for i in range(pf["total"]))
            del self._partial_frames[seq_id]
            self._decode_and_process(jpeg)

    def _decode_and_process(self, jpeg: bytes):
        arr   = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("cv2.imdecode returned None — corrupt JPEG?")
            return

        now = time.time()
        if self._last_frame_time:
            elapsed = now - self._last_frame_time
            self._display_fps = (
                0.9 * self._display_fps + 0.1 * (1.0 / max(elapsed, 1e-6))
            )
        self._last_frame_time = now

        with self._frame_lock:
            self._latest_raw    = frame
            self._display_frame = frame     # raw — overwritten if detector runs
            self._frame_buffer.append(frame)
        self._frames_rx += 1

        if self.detector is not None:
            self.detector.process(frame)    # fires _on_detection

        if self.on_frame:
            self.on_frame(frame)

    # ── Sensor ────────────────────────────────────────────────────────────────

    def _handle_sensor(self, data: bytes):
        if len(data) < 18:
            return
        ts, batt, roll, pitch, yaw = struct.unpack("!fHfff", data[:18])
        with self._telem_lock:
            self._telemetry = RobotTelemetry(
                timestamp=ts, battery_mv=batt,
                imu_roll=roll, imu_pitch=pitch, imu_yaw=yaw,
                extra=data[18:],
            )
        if self.on_telemetry:
            self.on_telemetry(self._telemetry)

    # ── Detection callback with filter ────────────────────────────────────────

    def _on_detection(self, result: "DetectionResult"):
        # Apply colour filter
        if self._allowed_colors is not None:
            result.blobs = [b for b in result.blobs
                            if b.color.lower() in self._allowed_colors]
        # Apply code-type filter
        if self._allowed_codes is not None:
            result.codes = [c for c in result.codes
                            if c.kind.upper() in self._allowed_codes]

        for blob in result.blobs:
            log.info(
                f"COLOR  {blob.color:6s}  area={blob.area:.0f}  center={blob.center}"
            )
        for code in result.codes:
            log.info(
                f"CODE   [{code.kind}]  data={code.data!r}  center={code.center}"
            )

        if result.annotated is not None:
            with self._frame_lock:
                self._display_frame = result.annotated

        if self.on_detection:
            self.on_detection(result)

    # ── Display window ────────────────────────────────────────────────────────

    def _window_loop(self):
        WIN = "ESP32-CAM  |  Q / Esc to quit"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 800, 600)

        # Build a placeholder shown while waiting for the ESP32
        ph = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(ph, "Waiting for ESP32...",
                    (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 80), 2)
        cv2.putText(ph, f'Set SERVER_IP = "{self._local_ip}"',
                    (55,  255), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 200, 255), 1)
        cv2.putText(ph, f"in your ESP32 sketch",
                    (175, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 200, 255), 1)
        cv2.putText(ph, f"Listening  port {self.cfg.video_port}",
                    (175, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 140), 1)

        log.info("Display window open — press Q or Esc to quit.")

        while self._running:
            with self._frame_lock:
                frame = (self._display_frame.copy()
                         if self._display_frame is not None else None)

            display = ph.copy() if frame is None else frame
            h, w    = display.shape[:2]

            # Semi-transparent top bar
            bar = display.copy()
            cv2.rectangle(bar, (0, 0), (w, 30), (15, 15, 15), -1)
            display = cv2.addWeighted(bar, 0.65, display, 0.35, 0)

            # HUD labels
            hs_txt  = "HS:✓" if self.handshake_done else "HS:…"
            col_txt = ("col:ALL" if self._allowed_colors is None
                       else "col:" + ",".join(sorted(self._allowed_colors)))
            cod_txt = ("code:ALL" if self._allowed_codes is None
                       else "code:" + ",".join(sorted(self._allowed_codes)))
            rob_txt = (f"ESP32:{self._robot_addr[0]}"
                       if self._robot_addr else "Waiting…")

            cv2.putText(display, rob_txt, (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 230, 80), 1)
            cv2.putText(display, hs_txt,  (int(w * 0.32), 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1)
            cv2.putText(display, col_txt, (int(w * 0.43), 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 160, 80), 1)
            cv2.putText(display, cod_txt, (int(w * 0.65), 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 160, 80), 1)
            cv2.putText(display, f"FPS:{self._display_fps:.1f}", (w - 105, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 80), 1)

            cv2.imshow(WIN, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                log.info("Quit key pressed.")
                break
            time.sleep(0.008)

        cv2.destroyAllWindows()
        self._running = False

    # ── Maintenance ───────────────────────────────────────────────────────────

    def _cleanup_loop(self):
        while self._running:
            time.sleep(1.0)
            now   = time.time()
            stale = [k for k, v in self._partial_frames.items()
                     if now - v["ts"] > FRAME_TIMEOUT]
            for k in stale:
                del self._partial_frames[k]

    def _maybe_stats(self):
        now = time.time()
        if now - self._last_stats_time >= self.cfg.stats_interval:
            elapsed = now - self._last_stats_time
            log.info(
                f"Stats │ {self._frames_rx / elapsed:.1f} fps │ "
                f"{self._bytes_rx * 8 / 1000 / elapsed:.0f} kbps │ "
                f"robot={self._robot_addr} │ hs={self.handshake_done}"
            )
            self._frames_rx   = 0
            self._bytes_rx    = 0
            self._last_stats_time = now


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — the three "edit here" sections are all you normally need to
# touch when tuning the system.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── ① What to detect (filter) ────────────────────────────────────────────
    #    Colours : any subset of ["red", "green", "blue"], or None = all
    #    Codes   : e.g. ["QRCODE"], ["QRCODE", "CODE128"], or None = all
    detect_colors = ["red", "green", "blue"]
    detect_codes  = ["QRCODE"]

    # ── ② Camera settings sent to ESP32 at handshake ─────────────────────────
    #    cam_resolution : 5=QVGA  8=VGA(default)  9=SVGA  13=UXGA
    #    cam_framerate  : 1 – 30 fps
    #    cam_quality    : 0 (best JPEG) – 63 (worst)  →  10-15 is balanced
    cam_resolution = 8
    cam_framerate  = 60
    cam_quality    = 18

    # ── ③ Server settings ────────────────────────────────────────────────────
    cfg = ServerConfig(
        host           = "0.0.0.0",
        video_port     = 5005,
        cmd_port       = 5006,
        show_window    = True,
        run_detector   = True,
        detect_colors  = detect_colors,
        detect_codes   = detect_codes,
        cam_resolution = cam_resolution,
        cam_framerate  = cam_framerate,
        cam_quality    = cam_quality,
    )

    server = UDPVideoServer(cfg)

    # ── Optional: react to detections after filtering ────────────────────────
    def my_detection_hook(result):
        if result.blobs:
            biggest = max(result.blobs, key=lambda b: b.area)
            h, w    = result.frame.shape[:2]
            nx = (biggest.center[0] / w) - 0.5
            ny = (biggest.center[1] / h) - 0.5
            # server.cmd_follow(nx, ny, 0.5)   # ← uncomment to drive

    server.on_detection = my_detection_hook

    log.info("Server starting — waiting for ESP32-CAM...")
    server.start()