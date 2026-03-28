// ═══════════════════════════════════════════════════════════
//  QUADRUPED MASTER  –  ESP32
//  PCA9685 @ 0x40  SDA=21 SCL=22
//  FL knee=0 hip=2 | FR knee=4 hip=6
//  RL knee=8 hip=10| RR knee=12 hip=14
// ═══════════════════════════════════════════════════════════

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ── Config ────────────────────────────────────────────────────
#define WIFI_SSID "Nyx"
#define WIFI_PASS "a123456a"
#define SERVER_IP "10.231.28.10"
#define MASTER_PORT 5006
#define ANNOUNCE_PORT 4999

// ── Gait tuning ───────────────────────────────────────────────
#define STAND_KNEE 90.0f // neutral knee angle (your servo home)
#define STAND_HIP 90.0f  // neutral hip  angle (your servo home)
#define LIFT_DEG 20.0f   // how much knee lifts during swing
#define STEP_DEG 18.0f   // how much hip sweeps per step
#define GAIT_MS 800      // ms per full trot cycle

// Sign per leg — +1 or -1.
// KNEE_SIGN: +1 = adding degrees LIFTS the leg
// HIP_SIGN:  +1 = adding degrees swings leg FORWARD
//                    FL    FR    RL    RR
float KNEE_SIGN[4] = {1, 1, 1, 1};
float HIP_SIGN[4] = {1, -1, 1, -1};

// ── PCA9685 (raw, no library) ─────────────────────────────────
#define PCA_ADDR 0x40
#define PWM_FREQ 50
#define SERV_MIN 102 // ~500  µs
#define SERV_MAX 512 // ~2500 µs

void pcaWrite(uint8_t ch, uint16_t off) {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x06 + ch * 4);
  Wire.write(0);
  Wire.write(0);
  Wire.write(off & 0xFF);
  Wire.write(off >> 8);
  Wire.endTransmission();
}

void pcaInit() {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x00);
  Wire.write(0x10);
  Wire.endTransmission();
  delay(5);
  uint8_t pre = (uint8_t)(25000000.0f / (4096.0f * PWM_FREQ) - 1.5f);
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0xFE);
  Wire.write(pre);
  Wire.endTransmission();
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x00);
  Wire.write(0xA0);
  Wire.endTransmission();
  delay(5);
}

// ── Servo helpers ─────────────────────────────────────────────
uint16_t angleToPwm(float a) {
  a = a < 0 ? 0 : a > 180 ? 180 : a;
  return (uint16_t)(SERV_MIN + (a / 180.0f) * (SERV_MAX - SERV_MIN));
}

static const uint8_t CH[4][2] = {{0, 2}, {4, 6}, {8, 10}, {12, 14}};

float curKnee[4], curHip[4];

void setLegNow(uint8_t leg, float knee, float hip) {
  pcaWrite(CH[leg][0], angleToPwm(knee));
  pcaWrite(CH[leg][1], angleToPwm(hip));
  curKnee[leg] = knee;
  curHip[leg] = hip;
}

void standAll() {
  for (uint8_t i = 0; i < 4; i++)
    setLegNow(i, STAND_KNEE, STAND_HIP);
}

// ── Protocol ──────────────────────────────────────────────────
#define CMD_MOVE 0x01
#define CMD_STOP 0x02
#define CMD_SET_SPEED 0x03
#define CMD_FOLLOW 0x04
#define CMD_TURN 0x05
#define CMD_ESTOP 0xFF

float g_vx = 0, g_yaw = 0, g_speed = 1.0f;
bool g_moving = false, g_estop = false;

// ── Trot gait ─────────────────────────────────────────────────
// Diagonal pairs: Group 0 = FL(0)+RR(3), Group 1 = FR(1)+RL(2)
// Group 0 swings first half of cycle, Group 1 swings second half.
static const uint8_t DIAG[4] = {0, 1, 1, 0};

void gaitTick() {
  if (g_estop || !g_moving) {
    standAll();
    return;
  }

  float t = (float)(millis() % (uint32_t)GAIT_MS) / (float)GAIT_MS;

  for (uint8_t leg = 0; leg < 4; leg++) {
    float phase = fmodf(t + (DIAG[leg] ? 0.5f : 0.0f), 1.0f);
    float drive = g_vx * g_speed;
    float turn = g_yaw * g_speed;
    float knee, hip;

    if (phase < 0.5f) {
      // SWING — leg in air, repositioning
      float sw = phase / 0.5f; // 0→1
      knee = STAND_KNEE + KNEE_SIGN[leg] * LIFT_DEG * sinf(sw * M_PI);
      float delta = STEP_DEG * (sw * 2.0f - 1.0f); // -STEP → +STEP
      hip = STAND_HIP + HIP_SIGN[leg] * (drive + turn) * delta;
    } else {
      // STANCE — leg on ground, pushing body
      float st = (phase - 0.5f) / 0.5f; // 0→1
      knee = STAND_KNEE;
      float delta = STEP_DEG * (1.0f - st * 2.0f); // +STEP → -STEP
      hip = STAND_HIP + HIP_SIGN[leg] * (drive + turn) * delta;
    }

    setLegNow(leg, knee, hip);
  }
}

WiFiUDP udp;
static uint8_t buf[64];

float bswapf(float v) {
  uint32_t u;
  memcpy(&u, &v, 4);
  u = __builtin_bswap32(u);
  memcpy(&v, &u, 4);
  return v;
}

void handleCmd(int len) {
  if (len < 5 || buf[0] != 0xCC || buf[1] != 0xDD)
    return;
  uint8_t cmd = buf[2];
  const uint8_t *p = buf + 5;

  switch (cmd) {
  case CMD_MOVE:
    if (len < 17)
      break;
    g_vx = bswapf(*(float *)p);
    g_moving = true;
    g_estop = false;
    break;

  case CMD_TURN:
    if (len < 9)
      break;
    g_yaw = bswapf(*(float *)p);
    g_moving = true;
    g_estop = false;
    break;

  case CMD_FOLLOW: {
    if (len < 17)
      break;
    float tx = bswapf(*(float *)p);
    float depth = bswapf(*(float *)(p + 8));
    g_yaw = tx / 320.0f;
    g_vx = constrain((200.0f - depth) / 200.0f, -1.0f, 1.0f);
    g_moving = true;
    g_estop = false;
    break;
  }

  case CMD_SET_SPEED:
    if (len < 9)
      break;
    g_speed = constrain(bswapf(*(float *)p), 0.0f, 1.0f);
    break;

  case CMD_STOP:
    g_moving = false;
    g_vx = g_yaw = 0;
    standAll();
    break;

  case CMD_ESTOP:
    g_estop = true;
    g_moving = false;
    standAll();
    break;
  }
}

void receiveCommands() {
  int sz = udp.parsePacket();
  if (sz <= 0)
    return;
  handleCmd(udp.read(buf, sizeof(buf)));
}

void sendTelemetry() {
  uint8_t pkt[20];
  pkt[0] = 0xEE;
  pkt[1] = 0xFF;
  float ts = millis() / 1000.0f;
  memcpy(pkt + 2, &ts, 4);
  memset(pkt + 6, 0, 14);
  udp.beginPacket(SERVER_IP, MASTER_PORT);
  udp.write(pkt, 20);
  udp.endPacket();
}

// ── Announce ──────────────────────────────────────────────────
void announce() {
  char msg[32];
  snprintf(msg, sizeof(msg), "QUAD:%s", WiFi.localIP().toString().c_str());
  udp.beginPacket(IPAddress(255, 255, 255, 255), ANNOUNCE_PORT);
  udp.print(msg);
  udp.endPacket();
}

// ── Setup / Loop ──────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  pcaInit();

  // Set all legs to stand pose — no sudden movement
  for (uint8_t i = 0; i < 4; i++) {
    curKnee[i] = STAND_KNEE;
    curHip[i] = STAND_HIP;
  }
  standAll();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] %s\n", WiFi.localIP().toString().c_str());
  udp.begin(MASTER_PORT);
  Serial.println("[READY]");
}

void loop() {
  static uint32_t lastTelem = 0, lastAnnounce = 0;
  uint32_t now = millis();

  receiveCommands();
  gaitTick();

  if (now - lastTelem > 200) {
    lastTelem = now;
    sendTelemetry();
  }
  if (now - lastAnnounce > 3000) {
    lastAnnounce = now;
    announce();
  }
}