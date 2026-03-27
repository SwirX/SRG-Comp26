import os
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

try:
    from vision_engine import VisionDetector, DetectionResult
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    logging.warning(
        "vision_engine.py not found — RAW display mode. "
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s"
)
log = logging.getLogger("LaptopControl")

DETECT_COLORS: Optional[List[str]] = ["red", "green", "yellow"]
DETECT_CODES:  Optional[List[str]] = ["QRCODE"]

CAM_RESOLUTION: int = 8
CAM_FRAMERATE:  int = 15
CAM_QUALITY:    int = 20

MAGIC_VIDEO     = b'\xAA\xBB'
MAGIC_CMD       = b'\xCC\xDD'
MAGIC_SENSOR    = b'\xEE\xFF'
MAGIC_HANDSHAKE = b'\x11\x22'
MAGIC_ACK       = b'\x33\x44'

MAX_PACKET     = 65507
FRAME_TIMEOUT  = 3.0    
MAX_HS_RETRIES = 5
HS_RETRY_DELAY = 1.0    

CMD_MOVE      = 0x01
CMD_STOP      = 0x02
CMD_SET_SPEED = 0x03
CMD_FOLLOW    = 0x04
CMD_TURN      = 0x05
CMD_SCAN_REFLECTIONS = 0x06
CMD_SET_FLASH_N = 0x10
CMD_ESTOP     = 0xFF

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"

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
    host:            str   = "10.221.49.10"
    video_port:      int   = 5005       # ESP32-CAM → PC
    master_port:     int   = 5006       # PC ↔ ESP32 Master
    cam_port:        int   = 5007       # PC → ESP32-CAM
    frame_buffer:    int   = 4
    stats_interval:  float = 5.0
    show_window:     bool  = True
    run_detector:    bool  = True
    detect_colors:   Optional[List[str]] = None
    detect_codes:    Optional[List[str]] = None
    debug_log:       bool  = False
    cam_resolution:  int = CAM_RESOLUTION
    cam_framerate:   int = CAM_FRAMERATE
    cam_quality:     int = CAM_QUALITY

class LaptopControlServer:
    def __init__(self, config: ServerConfig = ServerConfig()):
        self.cfg      = config
        self._running = False

        raw_colors = config.detect_colors if config.detect_colors is not None else DETECT_COLORS
        raw_codes  = config.detect_codes  if config.detect_codes  is not None else DETECT_CODES

        self._allowed_colors = {c.lower() for c in raw_colors} if raw_colors is not None else None
        self._allowed_codes  = {c.upper() for c in raw_codes}  if raw_codes  is not None else None

        self._local_ip = get_local_ip()

        if self.cfg.debug_log:
            os.makedirs("debug_logs", exist_ok=True)

        self._rx_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._rx_video.settimeout(1.0)

        self._rx_telem = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_telem.settimeout(1.0)

        self._tx_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cam_addr    = None    # Tuple[str, int]
        self._master_addr = None    # Tuple[str, int]

        self._hs_done    = False
        self._hs_lock    = threading.Lock()

        self._partial_frames: dict               = {}
        self._frame_buffer: deque                = deque(maxlen=config.frame_buffer)
        self._frame_lock                         = threading.Lock()
        self._latest_raw:    Optional[np.ndarray] = None
        self._display_frame: Optional[np.ndarray] = None
        self._last_flash_off: Optional[np.ndarray] = None

        self._telemetry  = RobotTelemetry()
        self._telem_lock = threading.Lock()

        self.detector = None
        if config.run_detector and DETECTOR_AVAILABLE:
            self.detector = VisionDetector(min_area=800, draw_overlay=True, on_detection=self._on_detection)

        self.on_frame:     Optional[Callable] = None
        self.on_telemetry: Optional[Callable] = None
        self.on_detection: Optional[Callable] = None

        self._frames_rx       = 0
        self._bytes_rx        = 0
        self._display_fps     = 0.0
        self._last_stats_time = time.time()
        self._last_frame_time = 0.0

    @property
    def handshake_done(self) -> bool:
        with self._hs_lock:
            return self._hs_done

    def start(self):
        self._rx_video.bind((self.cfg.host, self.cfg.video_port))
        self._rx_telem.bind((self.cfg.host, self.cfg.master_port))
        log.info(f"Server ready — {self.cfg.host}")
        
        self._running = True
        threading.Thread(target=self._rx_video_loop, daemon=True, name="RX-Video").start()
        threading.Thread(target=self._rx_telem_loop, daemon=True, name="RX-Telem").start()
        threading.Thread(target=self._cleanup_loop,  daemon=True, name="Cleanup").start()

        if self.cfg.show_window:
            self._window_loop()
        else:
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        self.stop()

    def stop(self):
        self._running = False
        self._rx_video.close()
        self._rx_telem.close()
        log.info("Server stopped.")

    def send_cam_command(self, cmd_id: int, payload: bytes = b'') -> bool:
        if self._cam_addr is None:
            return False
        pkt = MAGIC_CMD + struct.pack("!BH", cmd_id, len(payload)) + payload
        try:
            self._tx_sock.sendto(pkt, (self._cam_addr[0], self.cfg.cam_port))
            return True
        except OSError:
            return False

    def send_master_command(self, cmd_id: int, payload: bytes = b'') -> bool:
        if self._master_addr is None:
            return False
        pkt = MAGIC_CMD + struct.pack("!BH", cmd_id, len(payload)) + payload
        try:
            self._tx_sock.sendto(pkt, (self._master_addr[0], self.cfg.master_port))
            return True
        except OSError:
            return False

    def cmd_stop(self):                   self.send_master_command(CMD_STOP)
    def cmd_estop(self):                  self.send_master_command(CMD_ESTOP)
    def cmd_move(self, vx, vy, vz=0.0):   self.send_master_command(CMD_MOVE,      struct.pack("!fff", vx, vy, vz))
    def cmd_turn(self, yaw_rate):         self.send_master_command(CMD_TURN,      struct.pack("!f",   yaw_rate))
    def cmd_follow(self, tx, ty, depth):  self.send_master_command(CMD_FOLLOW,    struct.pack("!fff", tx, ty, depth))
    def cmd_set_speed(self, speed):       self.send_master_command(CMD_SET_SPEED, struct.pack("!f",   speed))
    def cmd_scan_reflections(self):       self.send_cam_command(CMD_SCAN_REFLECTIONS)
    def cmd_set_flash_n(self, n: int):    self.send_cam_command(CMD_SET_FLASH_N, struct.pack("!B", n))

    def _send_handshake(self):
        pkt = MAGIC_HANDSHAKE + struct.pack("BBB", self.cfg.cam_resolution, self.cfg.cam_framerate, self.cfg.cam_quality)
        self._tx_sock.sendto(pkt, (self._cam_addr[0], self.cfg.cam_port))

    def _handshake_loop(self):
        for attempt in range(1, MAX_HS_RETRIES + 1):
            with self._hs_lock:
                if self._hs_done: return
            self._send_handshake()
            time.sleep(HS_RETRY_DELAY)

    def _rx_video_loop(self):
        while self._running:
            try:
                data, addr = self._rx_video.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 2: continue
            
            if self._cam_addr is None:
                self._cam_addr = addr
                log.info(f"ESP32-CAM connected from {addr[0]}")
                threading.Thread(target=self._handshake_loop, daemon=True, name="Handshake").start()

            magic = data[:2]
            if magic == MAGIC_VIDEO:
                self._handle_video(data[2:])
            elif magic == MAGIC_ACK:
                self._handle_ack(data[2:])

            self._bytes_rx += len(data)
            self._maybe_stats()

    def _rx_telem_loop(self):
        while self._running:
            try:
                data, addr = self._rx_telem.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break
            
            if len(data) >= 2 and data[:2] == MAGIC_SENSOR:
                if self._master_addr is None:
                    self._master_addr = addr
                    log.info(f"ESP32-MASTER connected from {addr[0]}")
                self._handle_sensor(data[2:])

    def _handle_ack(self, data: bytes):
        if len(data) < 3: return
        with self._hs_lock: self._hs_done = True

    def _handle_video(self, data: bytes):
        if len(data) < 6: return
        seq_id, chunk_id, total_chunks, flash_state, _ = struct.unpack("!HBBBB", data[:6])
        payload = data[6:]

        pf = self._partial_frames.setdefault(seq_id, {
            "total": total_chunks, "chunks": {}, "ts": time.time(), "flash_state": flash_state
        })
        pf["chunks"][chunk_id] = payload

        if len(pf["chunks"]) == pf["total"]:
            jpeg = b''.join(pf["chunks"][i] for i in range(pf["total"]))
            fstate = pf["flash_state"]
            del self._partial_frames[seq_id]
            self._decode_and_process(jpeg, fstate)

    def _decode_and_process(self, jpeg: bytes, flash_state: int):
        arr   = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None: return

        now = time.time()
        if self._last_frame_time:
            self._display_fps = 0.9 * self._display_fps + 0.1 * (1.0 / max(now - self._last_frame_time, 1e-6))
        self._last_frame_time = now

        with self._frame_lock:
            self._latest_raw    = frame
            self._display_frame = frame
            self._frame_buffer.append(frame)
        self._frames_rx += 1

        if flash_state == 0:
            self._last_flash_off = frame
            if self.detector is not None: self.detector.process(frame)
        elif flash_state == 1:
            log.info("Received flash_on frame, generating reflection difference...")
            if self._last_flash_off is not None:
                diff = cv2.subtract(frame, self._last_flash_off)
                if self.detector is not None and hasattr(self.detector, 'process_reflections'):
                    res = self.detector.process_reflections(diff, frame)
                    with self._frame_lock:
                        if res and res.annotated is not None:
                            self._display_frame = res.annotated

        if self.on_frame: self.on_frame(frame)

    def _handle_sensor(self, data: bytes):
        if len(data) < 18: return
        ts, batt, roll, pitch, yaw = struct.unpack("!fHfff", data[:18])
        with self._telem_lock:
            self._telemetry = RobotTelemetry(timestamp=ts, battery_mv=batt, imu_roll=roll, imu_pitch=pitch, imu_yaw=yaw, extra=data[18:])
        if self.on_telemetry: self.on_telemetry(self._telemetry)

    def _on_detection(self, result: "DetectionResult"):
        if self._allowed_colors is not None: result.blobs = [b for b in result.blobs if b.color.lower() in self._allowed_colors]
        if self._allowed_codes is not None:  result.codes = [c for c in result.codes if c.kind.upper() in self._allowed_codes]

        if result.annotated is not None:
            with self._frame_lock:
                self._display_frame = result.annotated
        if self.on_detection: self.on_detection(result)

    def _window_loop(self):
        WIN = "Laptop | Q=Quit S=1Scan F=AutoFlash +/-=Rate"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 800, 600)
        ph = np.zeros((480, 640, 3), dtype=np.uint8)
        
        while self._running:
            with self._frame_lock:
                frame = self._display_frame.copy() if self._display_frame is not None else None

            display = ph.copy() if frame is None else frame
            h, w    = display.shape[:2]

            bar = display.copy()
            cv2.rectangle(bar, (0, 0), (w, 30), (15, 15, 15), -1)
            display = cv2.addWeighted(bar, 0.65, display, 0.35, 0)

            rob_txt = f"M:{self._master_addr[0] if self._master_addr else 'WAIT'} | C:{self._cam_addr[0] if self._cam_addr else 'WAIT'}"
            cv2.putText(display, rob_txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 80), 1)
            cv2.putText(display, f"FPS:{self._display_fps:.1f}", (w - 105, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 80), 1)

            cv2.imshow(WIN, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): break
            elif key == ord('s'):
                log.info("Scanning single reflection...")
                self.cmd_scan_reflections()
            elif key == ord('f'):
                new_n = 0 if getattr(self, '_auto_flash_n', 0) > 0 else 5
                self._auto_flash_n = new_n
                log.info(f"Auto-flash every {new_n} frames" if new_n else "Auto-flash disabled")
                self.cmd_set_flash_n(new_n)
            elif key in (ord('+'), ord('=')):
                curr_n = getattr(self, '_auto_flash_n', 0)
                if curr_n > 0:
                    self._auto_flash_n = curr_n + 1
                    log.info(f"Auto-flash every {self._auto_flash_n} frames")
                    self.cmd_set_flash_n(self._auto_flash_n)
            elif key in (ord('-'), ord('_')):
                curr_n = getattr(self, '_auto_flash_n', 0)
                if curr_n > 1:
                    self._auto_flash_n = curr_n - 1
                    log.info(f"Auto-flash every {self._auto_flash_n} frames")
                    self.cmd_set_flash_n(self._auto_flash_n)
            time.sleep(0.008)

        cv2.destroyAllWindows()
        self._running = False

    def _cleanup_loop(self):
        while self._running:
            time.sleep(1.0)
            now   = time.time()
            stale = [k for k, v in self._partial_frames.items() if now - v["ts"] > FRAME_TIMEOUT]
            for k in stale: del self._partial_frames[k]

    def _maybe_stats(self):
        now = time.time()
        if now - self._last_stats_time >= self.cfg.stats_interval:
            self._frames_rx = 0
            self._bytes_rx  = 0
            self._last_stats_time = now

if __name__ == "__main__":
    cfg = ServerConfig(host="0.0.0.0", video_port=5005, master_port=5006, cam_port=5007, show_window=True, run_detector=True)
    server = LaptopControlServer(cfg)

    def my_detection_hook(result):
        if result.blobs:
            biggest = max(result.blobs, key=lambda b: b.area)
            h, w    = result.frame.shape[:2]
            nx = (biggest.center[0] / w) - 0.5
            ny = (biggest.center[1] / h) - 0.5
            # server.cmd_follow(nx, ny, 0.5)

    server.on_detection = my_detection_hook
    log.info("Server starting — waiting for ESP32 devices...")
    server.start()
