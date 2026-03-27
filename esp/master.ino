#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>

const char *WIFI_SSID = "Helios";
const char *WIFI_PASSWORD = "a123456a";
const char *SERVER_IP = "10.221.49.10";
const uint16_t TELEMETRY_PORT = 5006;
const uint16_t MASTER_PORT = 5006;

const uint8_t MAGIC_CMD[2] = {0xCC, 0xDD};
const uint8_t MAGIC_SENSOR[2] = {0xEE, 0xFF};

// Commands
const uint8_t CMD_MOVE = 0x01;
const uint8_t CMD_STOP = 0x02;
const uint8_t CMD_SET_SPEED = 0x03;
const uint8_t CMD_FOLLOW = 0x04;
const uint8_t CMD_TURN = 0x05;
const uint8_t CMD_ESTOP = 0xFF;

// PCA9685 Config
#define SDA_PIN 21
#define SCL_PIN 22
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN 150 // Minimum pulse length count out of 4096 (0 degrees)
#define SERVOMAX 600 // Maximum pulse length count out of 4096 (180 degrees)

WiFiUDP udp;

float readBEFloat(const uint8_t *buf) {
  uint32_t raw = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
                 ((uint32_t)buf[2] << 8) | (uint32_t)buf[3];
  float f;
  memcpy(&f, &raw, sizeof(f));
  return f;
}

// OOP Architecture Modeled for 8DOF Spider
class Leg {
private:
  uint8_t hip_idx, knee_idx;
  float L1, L2; // Link lengths in mm
  bool inverted;
  float offset_hip, offset_knee;

public:
  Leg(uint8_t hip, uint8_t knee, float l1, float l2, bool inv)
      : hip_idx(hip), knee_idx(knee), L1(l1), L2(l2), inverted(inv) {
    offset_hip = 90.0;
    offset_knee = 90.0;
  }

  void setAngles(float hip_deg, float knee_deg) {
    if (inverted) {
      hip_deg = 180.0 - hip_deg;
      knee_deg = 180.0 - knee_deg;
    }

    hip_deg = constrain(hip_deg, 0.0, 180.0);
    knee_deg = constrain(knee_deg, 0.0, 180.0);

    uint16_t hip_pulse = map((long)(hip_deg * 10), 0, 1800, SERVOMIN, SERVOMAX);
    uint16_t knee_pulse =
        map((long)(knee_deg * 10), 0, 1800, SERVOMIN, SERVOMAX);

    pwm.setPWM(hip_idx, 0, hip_pulse);
    pwm.setPWM(knee_idx, 0, knee_pulse);
  }

  // Advanced 2DOF IK Algorithm:
  // x: forward/backward relative to shoulder
  // z: height down relative to shoulder
  void moveToIK(float x, float z) {
    float d2 = x * x + z * z;
    float d = sqrt(d2);

    // IK Bounds Check
    if (d >= (L1 + L2))
      d = L1 + L2 - 0.1;
    if (d <= abs(L1 - L2))
      d = abs(L1 - L2) + 0.1;

    // Law of cosines setup
    float alpha1 = atan2(x, z);
    float alpha2 = acos((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d));

    float hip_angle_rad = alpha1 + alpha2;
    float knee_angle_rad = acos((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2));

    // Convert to degrees and map to servo range (90 offset usually straight
    // down)
    float hip_angle = (hip_angle_rad * 180.0 / PI);
    float knee_angle = (knee_angle_rad * 180.0 / PI);

    setAngles(hip_angle + offset_hip - 90.0, knee_angle + offset_knee - 90.0);
  }
};

class Quadruped {
private:
  Leg *legs[4];
  float phase;
  float frequency;  // gait freq (Hz)
  float amplitudeX; // stride
  float amplitudeZ; // step height
  float zOffset;    // standing height (z down)

  float gaitPhase[4] = {0.0, PI, PI,
                        0.0}; // Phase offsets for FL, FR, BL, BR (Trot)
  float dx[4] = {0, 0, 0, 0};

  unsigned long lastTime;
  bool isMoving;

public:
  Quadruped() {
    // Leg layout:
    // 0,1: Front Left
    // 2,3: Front Right (inverted)
    // 4,5: Back Left
    // 6,7: Back Right (inverted)
    legs[0] = new Leg(0, 1, 50.0, 50.0, false);
    legs[1] = new Leg(2, 3, 50.0, 50.0, true);
    legs[2] = new Leg(4, 5, 50.0, 50.0, false);
    legs[3] = new Leg(6, 7, 50.0, 50.0, true);

    zOffset = 70.0;
    phase = 0;
    frequency = 0;
    isMoving = false;
    lastTime = millis();
  }

  void setup() {
    Wire.begin(SDA_PIN, SCL_PIN);
    pwm.begin();
    pwm.setPWMFreq(50);
    stand();
  }

  void stand() {
    isMoving = false;
    for (int i = 0; i < 4; i++) {
      legs[i]->moveToIK(0, zOffset);
    }
  }

  // Preset Functions mapped for intuitive motion triggering
  void walkForward() {
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 25.0;
    amplitudeZ = 20.0;
    dx[0] = 1.0;
    dx[1] = 1.0;
    dx[2] = 1.0;
    dx[3] = 1.0;
  }

  void walkBackward() {
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 25.0;
    amplitudeZ = 20.0;
    dx[0] = -1.0;
    dx[1] = -1.0;
    dx[2] = -1.0;
    dx[3] = -1.0;
  }

  void turnLeft() {
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 20.0;
    amplitudeZ = 20.0;
    dx[0] = -1.0;
    dx[2] = -1.0;
    dx[1] = 1.0;
    dx[3] = 1.0;
  }

  void turnRight() {
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 20.0;
    amplitudeZ = 20.0;
    dx[0] = 1.0;
    dx[2] = 1.0;
    dx[1] = -1.0;
    dx[3] = -1.0;
  }

  void strafeLeft() {
    // 2DOF pseudo-strafe using gait offsets
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 10.0;
    amplitudeZ = 25.0; // Higher steps for clearance
    dx[0] = 0.5;
    dx[2] = -0.5;
    dx[1] = -0.5;
    dx[3] = 0.5;
  }

  void strafeRight() {
    isMoving = true;
    frequency = 1.0;
    amplitudeX = 10.0;
    amplitudeZ = 25.0;
    dx[0] = -0.5;
    dx[2] = 0.5;
    dx[1] = 0.5;
    dx[3] = -0.5;
  }

  void update() {
    unsigned long now = millis();
    float dt = (now - lastTime) / 1000.0f;
    lastTime = now;

    if (isMoving) {
      phase += 2.0 * PI * frequency * dt;
      if (phase > 2.0 * PI)
        phase -= 2.0 * PI;

      for (int i = 0; i < 4; i++) {
        float p = phase + gaitPhase[i];
        // x follows sine wave, z lifts only during swing phase (when cos(p) >
        // 0)
        float x = amplitudeX * dx[i] * sin(p);
        float zLift = (cos(p) > 0) ? (amplitudeZ * cos(p)) : 0;
        float z = zOffset - zLift;

        legs[i]->moveToIK(x, z);
      }
    }
  }
};

Quadruped spider;

void sendTelemetry() {
  uint8_t buf[20];
  buf[0] = MAGIC_SENSOR[0];
  buf[1] = MAGIC_SENSOR[1];

  float ts_stamp = millis() / 1000.0f;
  uint32_t raw_ts;
  memcpy(&raw_ts, &ts_stamp, sizeof(raw_ts));

  buf[2] = (raw_ts >> 24) & 0xFF;
  buf[3] = (raw_ts >> 16) & 0xFF;
  buf[4] = (raw_ts >> 8) & 0xFF;
  buf[5] = raw_ts & 0xFF;

  buf[6] = 0;
  buf[7] = 0;
  memset(buf + 8, 0, 12);

  udp.beginPacket(SERVER_IP, TELEMETRY_PORT);
  udp.write(buf, sizeof(buf));
  udp.endPacket();
}

void receiveCommands() {
  int packetSize = udp.parsePacket();
  if (packetSize <= 0)
    return;

  uint8_t buf[256];
  int len = udp.read(buf, sizeof(buf));
  if (len < 5)
    return;

  if (buf[0] != MAGIC_CMD[0] || buf[1] != MAGIC_CMD[1])
    return;

  uint8_t cmdId = buf[2];
  uint16_t payloadLen = ((uint16_t)buf[3] << 8) | buf[4];
  const uint8_t *payload = buf + 5;

  switch (cmdId) {
  case CMD_MOVE: {
    if (payloadLen < 12)
      break;
    float vx = readBEFloat(payload);
    float vy = readBEFloat(payload + 4);
    float vz = readBEFloat(payload + 8);

    // Simplistic mapping of vx, vy to Quadruped preset IK functions
    if (vx > 0.5)
      spider.walkForward();
    else if (vx < -0.5)
      spider.walkBackward();
    else if (vy > 0.5)
      spider.strafeRight();
    else if (vy < -0.5)
      spider.strafeLeft();
    else
      spider.stand();

    Serial.printf(">> [CMD] MOVE  vx=%.3f  vy=%.3f\n", vx, vy);
    break;
  }
  case CMD_STOP:
    spider.stand();
    Serial.println(">> [CMD] STOP");
    break;
  case CMD_ESTOP:
    spider.stand();
    Serial.println(">> [CMD] EMERGENCY STOP");
    break;
  case CMD_TURN: {
    if (payloadLen < 4)
      break;
    float yawRate = readBEFloat(payload);
    if (yawRate > 0.1)
      spider.turnRight();
    else if (yawRate < -0.1)
      spider.turnLeft();
    else
      spider.stand();
    Serial.printf(">> [CMD] TURN  yaw_rate=%.3f\n", yawRate);
    break;
  }
  case CMD_FOLLOW: {
    // Advanced logic placeholder
    spider.walkForward();
    break;
  }
  default:
    break;
  }
}

void setup() {
  Serial.begin(115200);

  Serial.println("\n[INIT] Starting PCA and Servos...");
  spider.setup();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[NET] Connected!");

  udp.begin(MASTER_PORT);
}

void loop() {
  static uint32_t lastTelem = 0;
  if (millis() - lastTelem >= 1000) {
    sendTelemetry();
    lastTelem = millis();
  }

  receiveCommands();
  spider.update();
  delay(10); // Loop pacing
}
