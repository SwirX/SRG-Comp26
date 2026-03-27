#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>

// USER CONFIG bdlo hadxi
const char* WIFI_SSID     = "DESKTOP-3AUAAIU 7989";
// const char* WIFI_SSID     = "Helios";
const char* WIFI_PASSWORD = "password2008";
// const char* WIFI_PASSWORD = "a123456a";
const char* SERVER_IP     = "192.168.137.1";
const uint16_t VIDEO_PORT = 5005;
const uint16_t CMD_PORT   = 5006;

const uint32_t FRAME_INTERVAL_MS = 66;

// ── Flash LED ────────────────────────────────────────────────────────────────
#define FLASH_LED 4                  // GPIO 4 = AI-Thinker onboard white LED
const uint8_t FLASH_EVERY_N  = 5;    // do a flash-diff pair every Nth frame
bool     g_flashEnabled       = true; // master toggle (CMD_FLASH can disable)
uint16_t g_flashCounter       = 0;    // counts frames since last flash pair

// Protocol constants - ila bdlti f python bdl 7ta hna
const uint8_t MAGIC_VIDEO[2]  = {0xAA, 0xBB};
const uint8_t MAGIC_CMD[2]    = {0xCC, 0xDD};
const uint8_t MAGIC_SENSOR[2] = {0xEE, 0xFF};

// ─────────────────────────────────────────────────────────────────────────────
// Handshake Additions
// ─────────────────────────────────────────────────────────────────────────────
// Handshake magic bytes  (keep in sync with udp_server.py)
const uint8_t MAGIC_HANDSHAKE[2] = {0x11, 0x22};
const uint8_t MAGIC_ACK[2]       = {0x33, 0x44};

// Framesize lookup — maps the resolution_id byte to an esp_camera enum
// Add or remove entries to match what you put in CAM_RESOLUTION on the PC.
static const framesize_t FRAMESIZE_MAP[] = {
    FRAMESIZE_96X96,   // 0
    FRAMESIZE_QQVGA,   // 1  160×120
    FRAMESIZE_QCIF,    // 2  176×144
    FRAMESIZE_HQVGA,   // 3  240×176
    FRAMESIZE_240X240, // 4
    FRAMESIZE_QVGA,    // 5  320×240
    FRAMESIZE_CIF,     // 6  400×296
    FRAMESIZE_HVGA,    // 7  480×320
    FRAMESIZE_VGA,     // 8  640×480   ← default
    FRAMESIZE_SVGA,    // 9  800×600
    FRAMESIZE_XGA,     // 10 1024×768
    FRAMESIZE_HD,      // 11 1280×720
    FRAMESIZE_SXGA,    // 12 1280×1024
    FRAMESIZE_UXGA,    // 13 1600×1200
};
static const uint8_t FRAMESIZE_MAP_LEN =
    sizeof(FRAMESIZE_MAP) / sizeof(FRAMESIZE_MAP[0]);

// Frame interval (updated by handshake)
volatile uint32_t g_frameIntervalMs = FRAME_INTERVAL_MS;

// Command IDs
const uint8_t CMD_MOVE      = 0x01;
const uint8_t CMD_STOP      = 0x02;
const uint8_t CMD_SET_SPEED = 0x03;
const uint8_t CMD_FOLLOW    = 0x04;
const uint8_t CMD_TURN      = 0x05;
const uint8_t CMD_FLASH     = 0x10;   // toggle flash-assisted detection
const uint8_t CMD_ESTOP     = 0xFF;

// Max UDP payload that fits in a single datagram (leave headroom)
const size_t CHUNK_SIZE = 1400;

// Camera pin mapping — AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WiFiUDP udp;
uint16_t frameSeq = 0;

// Helper: read a big-endian float from a byte buffer
float readBEFloat(const uint8_t* buf) {
    uint32_t raw = ((uint32_t)buf[0] << 24) |
                   ((uint32_t)buf[1] << 16) |
                   ((uint32_t)buf[2] <<  8) |
                    (uint32_t)buf[3];
    float f;
    memcpy(&f, &raw, sizeof(f));
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera initialisation
// ─────────────────────────────────────────────────────────────────────────────
bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;    // 640×480
        config.jpeg_quality = 20;               // 0=best, 63=worst
        config.fb_count     = 2;
        config.grab_mode    = CAMERA_GRAB_LATEST;  // always newest frame
    } else {
        config.frame_size   = FRAMESIZE_QVGA;   // 320×240
        config.jpeg_quality = 20;
        config.fb_count     = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }

    // Optional quality tweaks
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);      // auto

    Serial.println("[CAM] Initialised OK");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Send one JPEG frame (chunked)
// ─────────────────────────────────────────────────────────────────────────────
// tag: 0x00 = normal frame, 0x01 = flash-ON frame, 0x02 = flash-OFF frame
void sendFrame(const uint8_t* jpeg, size_t len, uint8_t frameTag = 0x00) {
    size_t totalChunks = (len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (totalChunks > 255) {
        Serial.println("[TX] Frame too large, skipping.");
        return;
    }

    uint8_t buf[CHUNK_SIZE + 8];   // header + chunk payload

    for (uint8_t chunkId = 0; chunkId < (uint8_t)totalChunks; chunkId++) {
        size_t offset    = chunkId * CHUNK_SIZE;
        size_t chunkLen  = min(CHUNK_SIZE, len - offset);

        // Packet header: magic(2) | seq_id(2) | chunk_id(1) | total(1) | tag(1)
        buf[0] = MAGIC_VIDEO[0];
        buf[1] = MAGIC_VIDEO[1];
        buf[2] = (frameSeq >> 8) & 0xFF;
        buf[3] =  frameSeq       & 0xFF;
        buf[4] = chunkId;
        buf[5] = (uint8_t)totalChunks;
        buf[6] = frameTag;

        memcpy(buf + 7, jpeg + offset, chunkLen);

        udp.beginPacket(SERVER_IP, VIDEO_PORT);
        udp.write(buf, 7 + chunkLen);
        udp.endPacket();

        // Magic number: 200 microseconds gives the ESP32 WiFi MAC exactly enough time 
        // to physically push the UDP packet out without stalling the CPU loop. 
        // 0µs = queue overflows and frame drops. 1000µs = massive lag.
        delayMicroseconds(200);
    }
    frameSeq++;
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply Handshake Configuration
// ─────────────────────────────────────────────────────────────────────────────
void applyConfig(uint8_t resolutionId, uint8_t fps, uint8_t quality) {
    Serial.printf(
        "[CFG] Received config — resolution_id=%d  fps=%d  quality=%d\n",
        resolutionId, fps, quality
    );

    // ── Resolution ───────────────────────────────────────────────────────────
    if (resolutionId < FRAMESIZE_MAP_LEN) {
        sensor_t* s = esp_camera_sensor_get();
        if (s) {
            s->set_framesize(s, FRAMESIZE_MAP[resolutionId]);
            Serial.printf("[CFG] Frame size set to index %d\n", resolutionId);
        }
    } else {
        Serial.printf("[CFG] Unknown resolution_id %d — ignored.\n", resolutionId);
    }

    // ── JPEG quality ─────────────────────────────────────────────────────────
    {
        sensor_t* s = esp_camera_sensor_get();
        if (s) s->set_quality(s, quality);
        Serial.printf("[CFG] JPEG quality set to %d\n", quality);
    }

    // ── Frame rate → interval in ms ──────────────────────────────────────────
    if (fps > 0 && fps <= 60) {
        g_frameIntervalMs = 1000UL / fps;
        Serial.printf("[CFG] Frame interval set to %lu ms (~%d fps)\n",
                      g_frameIntervalMs, fps);
    }

    // ── Send ACK ─────────────────────────────────────────────────────────────
    uint8_t ack[5];
    ack[0] = MAGIC_ACK[0];
    ack[1] = MAGIC_ACK[1];
    ack[2] = resolutionId;
    ack[3] = fps;
    ack[4] = quality;
    udp.beginPacket(SERVER_IP, VIDEO_PORT);   // ACK goes back on the video port
    udp.write(ack, sizeof(ack));
    udp.endPacket();
    Serial.println("[CFG] ACK sent.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Receive and decode a command packet from the PC
// ─────────────────────────────────────────────────────────────────────────────
void receiveCommands() {
    int packetSize = udp.parsePacket();
    if (packetSize <= 0) return;

    uint8_t buf[256];
    int len = udp.read(buf, sizeof(buf));
    if (len < 2) return;

    // ── Handshake / config packet ─────────────────────────────────────────────
    if (buf[0] == MAGIC_HANDSHAKE[0] && buf[1] == MAGIC_HANDSHAKE[1]) {
        if (len >= 5) {
            // magic(2) | resolution_id(1) | fps(1) | quality(1)
            applyConfig(buf[2], buf[3], buf[4]);
        } else {
            Serial.println("[CFG] Handshake packet too short.");
        }
        return;   // don't fall through to command parsing
    }

    // ── Command packet ────────────────────────────────────────────────────────
    if (buf[0] != MAGIC_CMD[0] || buf[1] != MAGIC_CMD[1]) return;
    if (len < 5) return;   // magic(2) + cmd_id(1) + payload_len(2)

    uint8_t  cmdId      = buf[2];
    uint16_t payloadLen = ((uint16_t)buf[3] << 8) | buf[4];
    const uint8_t* payload = buf + 5;

    switch (cmdId) {

        case CMD_MOVE: {
            if (payloadLen < 12) break;
            float vx = readBEFloat(payload);
            float vy = readBEFloat(payload + 4);
            float vz = readBEFloat(payload + 8);
            Serial.printf("[CMD] MOVE  vx=%.3f  vy=%.3f  vz=%.3f\n", vx, vy, vz);
            // TODO: drive your motors here
            break;
        }

        case CMD_STOP:
            Serial.println("[CMD] STOP");
            // TODO: stop all motors
            break;

        case CMD_ESTOP:
            Serial.println("[CMD] EMERGENCY STOP");
            // TODO: hard stop, disable drivers
            break;

        case CMD_TURN: {
            if (payloadLen < 4) break;
            float yawRate = readBEFloat(payload);
            Serial.printf("[CMD] TURN  yaw_rate=%.3f\n", yawRate);
            // TODO: differential steering
            break;
        }

        case CMD_FOLLOW: {
            if (payloadLen < 12) break;
            float tx    = readBEFloat(payload);
            float ty    = readBEFloat(payload + 4);
            float depth = readBEFloat(payload + 8);
            Serial.printf("[CMD] FOLLOW  target=(%.3f, %.3f)  depth=%.3f\n",
                          tx, ty, depth);
            // TODO: implement visual follow logic
            break;
        }

        case CMD_SET_SPEED: {
            if (payloadLen < 4) break;
            float speed = readBEFloat(payload);
            Serial.printf("[CMD] SET_SPEED  speed=%.3f\n", speed);
            // TODO: update motor speed limit
            break;
        }

        case CMD_FLASH: {
            // payload[0] = 0 → disable, 1 → enable
            if (payloadLen >= 1) {
                g_flashEnabled = (payload[0] != 0);
                Serial.printf("[CMD] FLASH %s\n", g_flashEnabled ? "ON" : "OFF");
            }
            break;
        }

        default:
            Serial.printf("[CMD] Unknown command: 0x%02X\n", cmdId);
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n[BOOT] ESP32-CAM UDP Client starting...");

    // Flash LED init
    pinMode(FLASH_LED, OUTPUT);
    digitalWrite(FLASH_LED, LOW);

    if (!initCamera()) {
        Serial.println("[BOOT] Camera failed — halting.");
        while (true) delay(1000);
    }

    Serial.printf("[WIFI] Connecting to %s ...", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\n[WIFI] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

    // Open UDP socket for receiving commands
    udp.begin(CMD_PORT);
    Serial.printf("[UDP]  Listening for commands on port %d\n", CMD_PORT);
    Serial.printf("[UDP]  Sending video to %s:%d\n", SERVER_IP, VIDEO_PORT);
    Serial.println("[BOOT] Ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    static uint32_t lastFrameMs = 0;
    uint32_t now = millis();

    // ── Capture & transmit ────────────────────────────────────────────────────
    if (now - lastFrameMs >= g_frameIntervalMs) {
        // Reset strictly to millis() to avoid the "death spiral" where the interval drops below 0 
        // and the ESP32 tries to catch up by rapid-firing frames, crashing the WiFi.
        lastFrameMs = millis();
        
        g_flashCounter++;
        bool doFlashPair = g_flashEnabled && (g_flashCounter >= FLASH_EVERY_N);

        if (doFlashPair) {
            g_flashCounter = 0;

            // 1) Capture flash-OFF frame (ambient)
            camera_fb_t* fbOff = esp_camera_fb_get();
            if (fbOff) {
                sendFrame(fbOff->buf, fbOff->len, 0x02);  // tag = flash-OFF
                esp_camera_fb_return(fbOff);
            }

            // 2) Flash ON → capture → flash OFF
            digitalWrite(FLASH_LED, HIGH);
            delay(30);   // let auto-exposure settle briefly
            camera_fb_t* fbOn = esp_camera_fb_get();
            digitalWrite(FLASH_LED, LOW);
            if (fbOn) {
                sendFrame(fbOn->buf, fbOn->len, 0x01);    // tag = flash-ON
                esp_camera_fb_return(fbOn);
            }
        } else {
            // Normal frame
            camera_fb_t* fb = esp_camera_fb_get();
            if (fb == nullptr) {
                Serial.println("[CAM] Frame capture failed");
            } else {
                sendFrame(fb->buf, fb->len, 0x00);        // tag = normal
                esp_camera_fb_return(fb);
            }
        }
    }

    // ── Commands ──────────────────────────────────────────────────────────────
    receiveCommands();

    // Yield to RTOS without wasting time
    delay(1);
}