"""
ws_bridge.py  —  WebSocket ↔ UDP bridge
Run this on your PC, then open controller.html in any browser.

    pip install websockets
    python ws_bridge.py
"""

import asyncio, json, socket, struct, logging
import websockets

# ── Config ────────────────────────────────────────────────────
WS_HOST       = "0.0.0.0"
WS_PORT       = 8765
ROBOT_IP      = ""        # leave blank — auto-discovered via QUAD: broadcast
ROBOT_PORT    = 5006
ANNOUNCE_PORT = 4999
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("bridge")

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

robot_ip = ROBOT_IP or None

# ── UDP helpers ───────────────────────────────────────────────
MAGIC_CMD = b'\xCC\xDD'

CMD_MOVE      = 0x01
CMD_STOP      = 0x02
CMD_SET_SPEED = 0x03
CMD_FOLLOW    = 0x04
CMD_TURN      = 0x05
CMD_ESTOP     = 0xFF

def send_udp(cmd_id: int, payload: bytes = b''):
    ip = robot_ip or ROBOT_IP
    if not ip:
        log.warning("Robot IP unknown — command dropped")
        return
    pkt = MAGIC_CMD + struct.pack("!BH", cmd_id, len(payload)) + payload
    udp.sendto(pkt, (ip, ROBOT_PORT))

# ── Discovery ─────────────────────────────────────────────────
async def discover():
    global robot_ip
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", ANNOUNCE_PORT))
    sock.setblocking(False)
    log.info(f"Listening for QUAD: broadcast on :{ANNOUNCE_PORT} ...")
    loop = asyncio.get_event_loop()
    while True:
        try:
            data = await loop.sock_recv(sock, 64)
            msg = data.decode(errors="ignore").strip()
            if msg.startswith("QUAD:"):
                ip = msg[5:]
                if ip != robot_ip:
                    robot_ip = ip
                    log.info(f"Robot found at {ip}")
        except Exception:
            await asyncio.sleep(0.2)

# ── WebSocket handler ─────────────────────────────────────────
async def handler(ws):
    log.info(f"Controller connected: {ws.remote_address}")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type")

            if t == "move":
                vx = float(msg.get("vx", 0))
                vy = float(msg.get("vy", 0))
                # Pack all 3 floats big-endian as the ESP32 expects
                send_udp(CMD_MOVE, struct.pack("!fff", vx, vy, 0.0))

            elif t == "turn":
                yr = float(msg.get("yaw", 0))
                send_udp(CMD_TURN, struct.pack("!f", yr))

            elif t == "stop":
                send_udp(CMD_STOP)

            elif t == "estop":
                send_udp(CMD_ESTOP)

            elif t == "speed":
                sp = max(0.0, min(1.0, float(msg.get("value", 1.0))))
                send_udp(CMD_SET_SPEED, struct.pack("!f", sp))

            elif t == "ping":
                await ws.send(json.dumps({
                    "type": "pong",
                    "robot": robot_ip or "searching..."
                }))

    except websockets.exceptions.ConnectionClosed:
        pass
    log.info(f"Controller disconnected: {ws.remote_address}")

# ── Main ──────────────────────────────────────────────────────
async def main():
    asyncio.ensure_future(discover())
    async with websockets.serve(handler, WS_HOST, WS_PORT):
        local = socket.gethostbyname(socket.gethostname())
        log.info(f"WS bridge on ws://0.0.0.0:{WS_PORT}")
        log.info(f"Open controller.html — on mobile use: http://{local}:8080/controller.html")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())