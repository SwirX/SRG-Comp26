// ═══════════════════════════════════════════════════════════
//  QUADRUPED MASTER  –  ESP32 (IK VERSION - FIXED)
//  PCA9685 @ 0x40  SDA=21 SCL=22
//  IK Engine: 2-Link Planar (Hip Pitch, Knee Pitch)
// ═══════════════════════════════════════════════════════════
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>

// ── Config ────────────────────────────────────────────────────
#define WIFI_SSID     "Nyx"
#define WIFI_PASS     "a123456a"
#define SERVER_IP     "10.231.28.10"
#define MASTER_PORT   5006
#define ANNOUNCE_PORT 4999

// ── Kinematics ───────────────────────────────────────────────
#define L1 100.0f   // Hip to Knee (mm)
#define L2 125.0f   // Knee to Foot (mm)
#define MAX_REACH (L1 + L2 - 10.0f)
#define NEUTRAL_HIP  90.0f
#define NEUTRAL_KNEE 90.0f
#define STAND_HEIGHT 180.0f 

// Gait Parameters
#define GAIT_MS      800
#define STEP_HEIGHT  40.0f
#define STEP_LENGTH  100.0f

// ── PCA9685 ───────────────────────────────────────────────────
#define PCA_ADDR  0x40
#define PWM_FREQ  50
#define SERV_MIN  102
#define SERV_MAX  512

void pcaWrite(uint8_t ch, uint16_t off) {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x06 + ch * 4);
  Wire.write(0); Wire.write(0);
  Wire.write(off & 0xFF); Wire.write(off >> 8);
  Wire.endTransmission();
}

void pcaInit() {
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x00); Wire.write(0x10); Wire.endTransmission(); delay(5);
  uint8_t pre = (uint8_t)(25000000.0f / (4096.0f * PWM_FREQ) - 1.5f);
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0xFE); Wire.write(pre); Wire.endTransmission();
  Wire.beginTransmission(PCA_ADDR);
  Wire.write(0x00); Wire.write(0xA0); Wire.endTransmission(); delay(5);
}

// ── Servo helpers ─────────────────────────────────────────────
uint16_t angleToPwm(float a) {
  a = a < 0 ? 0 : a > 180 ? 180 : a;
  return (uint16_t)(SERV_MIN + (a / 180.0f) * (SERV_MAX - SERV_MIN));
}

// Leg Mapping: {Knee Channel, Hip Channel}
static const uint8_t CH[4][2] = {{0,2},{4,6},{8,10},{12,14}};

// Signs to correct mechanical orientation per leg
float HIP_SIGN[4]  = { 1,  -1,   1,  -1}; 
float KNEE_SIGN[4] = { 1,   1,   1,   1};

void setLegNow(uint8_t leg, float kneeDeg, float hipDeg) {
  float k = NEUTRAL_KNEE + (kneeDeg * KNEE_SIGN[leg]);
  float h = NEUTRAL_HIP  + (hipDeg  * HIP_SIGN[leg]);
  pcaWrite(CH[leg][0], angleToPwm(k));
  pcaWrite(CH[leg][1], angleToPwm(h));
}

// ── Inverse Kinematics ────────────────────────────────────────
void solveIK(float x, float y, float &outHip, float &outKnee) {
  float dist = sqrt(x*x + y*y);
  if (dist > MAX_REACH) {
    float scale = MAX_REACH / dist;
    x *= scale; y *= scale;
    dist = MAX_REACH;
  }

  float cosKnee = (x*x + y*y - L1*L1 - L2*L2) / (2.0f * L1 * L2);
  if (cosKnee > 1.0f) cosKnee = 1.0f;
  if (cosKnee < -1.0f) cosKnee = -1.0f;
  
  float kneeRad = acos(cosKnee);
  float atan2YX = atan2(y, x);
  float atan2Term = atan2(L2 * sin(kneeRad), L1 + L2 * cos(kneeRad));
  float hipRad = atan2YX - atan2Term;

  outHip  = hipRad  * (180.0f / PI);
  outKnee = kneeRad * (180.0f / PI); 
}

// ── Protocol ──────────────────────────────────────────────────
#define CMD_MOVE      0x01
#define CMD_STOP      0x02
#define CMD_SET_SPEED 0x03
#define CMD_FOLLOW    0x04
#define CMD_TURN      0x05
#define CMD_ESTOP     0xFF

float  g_vx     = 0, g_vy = 0, g_yaw   = 0, g_speed = 1.0f;
bool   g_moving = false, g_estop = false;

// ── IK Gait Engine ────────────────────────────────────────────
void ikGaitTick() {
  if (g_estop || !g_moving) { 
    for (uint8_t i = 0; i < 4; i++) setLegNow(i, 0, 0); 
    return; 
  }

  float t = (float)(millis() % (uint32_t)GAIT_MS) / (float)GAIT_MS;
  float stepX = g_vx * STEP_LENGTH * g_speed;
  float stepY = g_vy * STEP_LENGTH * g_speed;
  float rotR  = g_yaw * STEP_LENGTH * g_speed;

  static const uint8_t DIAG[4] = {0, 1, 1, 0};

  for (uint8_t leg = 0; leg < 4; leg++) {
    float phase = fmodf(t + (DIAG[leg] ? 0.5f : 0.0f), 1.0f);
    float footX, footY;
    float lift = 0;

    if (phase < 0.5f) {
      float sw = phase / 0.5f;
      lift = sinf(sw * PI) * STEP_HEIGHT;
      footX = stepX * (sw * 2.0f - 1.0f);
      footY = STAND_HEIGHT - lift;
    } else {
      float st = (phase - 0.5f) / 0.5f;
      lift = 0;
      footX = stepX * (1.0f - st * 2.0f);
      footY = STAND_HEIGHT;
    }

    float rotOffset = 0;
    if (leg == 0 || leg == 2) rotOffset = -rotR;
    else rotOffset = rotR;
    footX += rotOffset;

    float hipAng, kneeAng;
    solveIK(footX, footY, hipAng, kneeAng);
    
    // Bias to match mechanical neutral standing pose
    float kneeBias = -45.0f; 

    setLegNow(leg, kneeAng + kneeBias, hipAng);
  }
}

// ── UDP ───────────────────────────────────────────────────────
WiFiUDP udp;
static uint8_t buf[64];

float bswapf(float v) {
  uint32_t u; memcpy(&u, &v, 4);
  u = __builtin_bswap32(u);
  memcpy(&v, &u, 4); return v;
}

void handleCmd(int len) {
  if (len < 5 || buf[0] != 0xCC || buf[1] != 0xDD) return;
  uint8_t cmd = buf[2];
  const uint8_t *p = buf + 5;
  switch (cmd) {
    case CMD_MOVE:
      if (len < 17) break;
      // ✅ FIXED: Cast to float pointer THEN dereference
      g_vx = bswapf(*(float*)p);
      g_vy = bswapf(*(float*)(p+4));
      g_moving = true; g_estop = false; break;
    case CMD_TURN:
      if (len < 9) break;
      g_yaw = bswapf(*(float*)p);
      g_moving = true; g_estop = false; break;
    case CMD_SET_SPEED:
      if (len < 9) break;
      g_speed = constrain(bswapf(*(float*)p), 0.0f, 1.0f); break;
    case CMD_STOP:
      g_moving = false; g_vx = g_vy = g_yaw = 0; break;
    case CMD_ESTOP:
      g_estop = true; g_moving = false; g_vx = g_vy = g_yaw = 0; break;
  }
}

void receiveCommands() {
  int sz = udp.parsePacket(); if (sz <= 0) return;
  handleCmd(udp.read(buf, sizeof(buf)));
}

// ── Telemetry & Announce ──────────────────────────────────────
void sendTelemetry() {
  uint8_t pkt[20];
  pkt[0] = 0xEE; pkt[1] = 0xFF;
  float ts = millis() / 1000.0f; memcpy(pkt+2,&ts, 4);
  memset(pkt+6, 0, 14);
  udp.beginPacket(SERVER_IP, MASTER_PORT);
  udp.write(pkt, 20); udp.endPacket();
}

void announce() {
  char msg[32]; snprintf(msg, sizeof(msg), "QUAD:%s", WiFi.localIP().toString().c_str());
  udp.beginPacket(IPAddress(255,255,255,255), ANNOUNCE_PORT);
  udp.print(msg); udp.endPacket();
}

// ── Setup / Loop ──────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  pcaInit();
  for (uint8_t i = 0; i < 4; i++) setLegNow(i, 0, 0);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\n[WiFi] %s\n", WiFi.localIP().toString().c_str());
  udp.begin(MASTER_PORT);
  Serial.println("[READY] IK Engine Active");
}

void loop() {
  static uint32_t lastTelem = 0, lastAnnounce = 0;
  uint32_t now = millis();
  receiveCommands();
  ikGaitTick();
  if (now - lastTelem    > 200 ) { lastTelem    = now; sendTelemetry(); }
  if (now - lastAnnounce > 3000) { lastAnnounce = now; announce(); }
}