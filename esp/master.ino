/*
 * ESP32 Spider Robot - Inverse Kinematics & Walking Engine
 * 
 * Configuration:
 * - 4 Legs, 2 DOF per leg (Hip + Knee)
 * - Leg Mounting: 45 Degrees splay
 * - Hardware: ESP32 + PCA9685
 * - Servo Mapping: Even channels (0, 2, 4... 14)
 * 
 * Step 1 Goal: 
 *   - Initialize ALL servos to 90° (stand position)
 *   - Wait 2 seconds
 *   - Walk Forward using IK with REVERSED knee angles (0 = knee UP)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

// ============================================================================
// 1. GLOBAL CONFIGURATION (Settings)
// ============================================================================

// --- Robot Dimensions (cm) ---
const float UPPER_LEG_LEN = 10.0;   // Coxa/Femur length
const float LOWER_LEG_LEN = 12.0;   // Tibia length

// --- Servo Settings ---
#define PCA9685_ADDRESS   0x40
#define SERVO_FREQ        50        // Analog servos run at ~50 Hz
#define MIN_PULSE         100       // Minimum pulse width (calibrate for your servo)
#define MAX_PULSE         500       // Maximum pulse width (calibrate for your servo)
#define CENTER_PULSE      306       // Approx 1500us for 90 degrees

// --- Movement Settings ---
const float STAND_HEIGHT    = 15.0; // Height of foot from hip in stand position (cm)
const float STEP_LENGTH     = 6.0;  // How far forward/back the leg swings (cm)
const float STEP_HEIGHT     = 4.0;  // How high the leg lifts during swing (cm)
const int   GAIT_SPEED      = 20;   // Delay in ms between gait steps (Lower = Faster)

// --- Servo Direction Calibration ---
// If a leg moves backwards when it should move forwards, toggle the 1 to -1
// This accounts for mirror mounting on Left vs Right sides
const int DIR_FL = 1; 
const int DIR_FR = -1; 
const int DIR_RL = 1; 
const int DIR_RR = -1;

// --- Knee Angle Inversion ---
// Set to true if servo 0 = knee UP, false if servo 0 = knee DOWN
const bool INVERT_KNEE_ANGLE = true;  // <-- CHANGE THIS based on your mechanical mounting

// ============================================================================
// 2. HARDWARE ABSTRACTION
// ============================================================================

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDRESS);

// Enum for Leg Indices
enum LegID { FL, FR, RL, RR, NUM_LEGS };

// Structure to hold Leg Configuration
struct LegConfig {
  uint8_t pinKnee;
  uint8_t pinHip;
  int     direction; // 1 or -1 for servo reversal (hip direction)
};

// Pin Mapping based on prompt: 0=Knee, 2=Hip, 4=Knee, 6=Hip...
const LegConfig legsConfig[NUM_LEGS] = {
  { 0,  2,  DIR_FL }, // Front Left
  { 4,  6,  DIR_FR }, // Front Right
  { 8,  10, DIR_RL }, // Rear Left
  { 12, 14, DIR_RR }  // Rear Right
};

// ============================================================================
// 3. INVERSE KINEMATICS CLASS
// ============================================================================

class RobotLeg {
  public:
    uint8_t pinKnee;
    uint8_t pinHip;
    int     direction;
    
    // Current Target Coordinates (Local Leg Plane)
    // X = Forward/Back along leg mount axis, Y = Height (Down is positive)
    float targetX; 
    float targetY;

    // Current Smoothed Coordinates (for interpolation)
    float currentX;
    float currentY;

    RobotLeg(uint8_t kPin, uint8_t hPin, int dir) {
      pinKnee = kPin;
      pinHip = hPin;
      direction = dir;
      currentX = 0;
      currentY = STAND_HEIGHT;
      targetX = 0;
      targetY = STAND_HEIGHT;
    }

    // Set the target position for the foot
    void setTarget(float x, float y) {
      targetX = x;
      targetY = y;
    }

    // Update the leg state (Interpolate and Solve IK)
    void update() {
      // Simple Linear Interpolation (Lerp) for smooth movement
      // Move 20% of the way to the target per call
      currentX = currentX + (targetX - currentX) * 0.2;
      currentY = currentY + (targetY - currentY) * 0.2;

      // Solve IK
      float kneeAngle = 0;
      float hipAngle = 0;
      solveIK(currentX, currentY, kneeAngle, hipAngle);

      // Write to Servos
      // KNEE: Apply inversion if configured (0 = knee UP)
      if (INVERT_KNEE_ANGLE) {
        writeServo(pinKnee, 180.0 - kneeAngle);  // Invert knee angle
      } else {
        writeServo(pinKnee, kneeAngle);
      }
      // HIP: Normal direction handling
      writeServo(pinHip, hipAngle);
    }

  private:
    // Convert Angle (0-180) to PCA9685 Pulse
    void writeServo(uint8_t pin, float angle) {
      // Constrain angle to valid servo range
      if (angle < 0) angle = 0;
      if (angle > 180) angle = 180;

      // Apply Direction Multiplier (Mirror mounting correction for HIP)
      // We map 90deg as center. 
      // If direction is -1: 90 stays 90, 100 becomes 80, 80 becomes 100.
      float adjustedAngle = 90 + ((angle - 90) * direction);

      uint16_t pulse = map(adjustedAngle, 0, 180, MIN_PULSE, MAX_PULSE);
      pwm.setPWM(pin, 0, pulse);
    }

    // 2-DOF Inverse Kinematics Solver
    // x: Horizontal distance from Hip (along leg swing axis)
    // y: Vertical distance from Hip (Down is positive)
    void solveIK(float x, float y, float &kneeOut, float &hipOut) {
      // Prevent division by zero or sqrt negative
      float distSq = x*x + y*y;
      if (distSq == 0) {
        kneeOut = 90; 
        hipOut = 90; 
        return;
      }

      float dist = sqrt(distSq);

      // Check reachability
      float maxReach = UPPER_LEG_LEN + LOWER_LEG_LEN;
      if (dist > maxReach) {
        // Target too far, clamp to max reach
        float scale = maxReach / dist;
        x *= scale;
        y *= scale;
        dist = maxReach;
      }

      // Law of Cosines for Knee Angle (Joint 2)
      // cos(theta2) = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2)
      float cosKnee = (distSq - UPPER_LEG_LEN*UPPER_LEG_LEN - LOWER_LEG_LEN*LOWER_LEG_LEN) / 
                      (2 * UPPER_LEG_LEN * LOWER_LEG_LEN);
      
      // Clamp for acos safety
      if (cosKnee > 1) cosKnee = 1;
      if (cosKnee < -1) cosKnee = -1;

      float theta2 = acos(cosKnee); // Angle in Radians

      // Law of Cosines for Hip Angle (Joint 1)
      // theta1 = atan2(y, x) - acos((L1^2 + dist^2 - L2^2) / (2 * L1 * dist))
      float theta1 = atan2(y, x) - acos((UPPER_LEG_LEN*UPPER_LEG_LEN + distSq - LOWER_LEG_LEN*LOWER_LEG_LEN) / 
                                        (2 * UPPER_LEG_LEN * dist));

      // Convert Radians to Degrees
      float kneeDeg = degrees(theta2);
      float hipDeg = degrees(theta1);

      // --- Mechanical Offset Calibration ---
      // These offsets align the mathematical model with your physical servo horn mounting
      // Adjust these values if the leg posture looks incorrect at "neutral"
      const float KNEE_OFFSET = 0.0;   // Typically 0 when using INVERT_KNEE_ANGLE
      const float HIP_OFFSET  = 90.0;  // 90 = servo neutral position

      kneeOut = kneeDeg + KNEE_OFFSET; 
      hipOut  = hipDeg + HIP_OFFSET;
    }
};

// Instantiate Legs
RobotLeg legFL(legsConfig[FL].pinKnee, legsConfig[FL].pinHip, legsConfig[FL].direction);
RobotLeg legFR(legsConfig[FR].pinKnee, legsConfig[FR].pinHip, legsConfig[FR].direction);
RobotLeg legRL(legsConfig[RL].pinKnee, legsConfig[RL].pinHip, legsConfig[RL].direction);
RobotLeg legRR(legsConfig[RR].pinKnee, legsConfig[RR].pinHip, legsConfig[RR].direction);

RobotLeg* allLegs[NUM_LEGS] = { &legFL, &legFR, &legRL, &legRR };

// ============================================================================
// 4. GAIT ENGINE
// ============================================================================

// Gait States
enum GaitState { STAND, WALK_FORWARD };
GaitState currentState = STAND;

// Gait Timing
unsigned long lastStepTime = 0;
int stepIndex = 0; // 0 to 3 for 4 legs sequence

void setupGait() {
  // Initialize all legs to stand position via IK
  for (int i = 0; i < NUM_LEGS; i++) {
    allLegs[i]->setTarget(0, STAND_HEIGHT);
    allLegs[i]->update(); // Force immediate update
  }
  delay(500); // Brief wait for servos to settle
}

// Initialize ALL servos to exact 90 degrees (bypass IK)
void initializeServosTo90() {
  Serial.println("Setting all servos to 90 degrees (neutral)...");
  for (int i = 0; i < NUM_LEGS; i++) {
    // Write CENTER_PULSE directly to both knee and hip channels
    pwm.setPWM(legsConfig[i].pinKnee, 0, CENTER_PULSE);
    pwm.setPWM(legsConfig[i].pinHip, 0, CENTER_PULSE);
  }
}

void updateGait() {
  if (currentState == STAND) {
    // Keep standing via IK
    for (int i = 0; i < NUM_LEGS; i++) {
      allLegs[i]->setTarget(0, STAND_HEIGHT);
      allLegs[i]->update();
    }
  } 
  else if (currentState == WALK_FORWARD) {
    // Simple Wave Gait using sine wave approximation
    unsigned long now = millis();
    if (now - lastStepTime > GAIT_SPEED) {
      lastStepTime = now;
      stepIndex = (stepIndex + 1) % 4;
      
      // Update each leg with phase-offset sine wave
      for (int i = 0; i < NUM_LEGS; i++) {
        // Phase offset: each leg 90 degrees (PI/2) apart for wave gait
        float phase = (millis() * 0.005) + (i * PI / 2);
        
        // X: Forward/back oscillation
        float x = sin(phase) * STEP_LENGTH;
        
        // Y: Height - lift when at swing extremes (abs(cos) creates lift at peaks)
        float y = STAND_HEIGHT - abs(cos(phase)) * STEP_HEIGHT;
        
        allLegs[i]->setTarget(x, y);
      }
    }

    // Update all servos with interpolated positions
    for (int i = 0; i < NUM_LEGS; i++) {
      allLegs[i]->update();
    }
  }
}

// ============================================================================
// 5. MAIN ARDUINO FUNCTIONS
// ============================================================================

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // ESP32 SDA=GPIO21, SCL=GPIO22
  
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  
  Serial.println("Spider Robot Initializing...");
  
  // STEP 1: Set ALL servos to exact 90 degrees (neutral mechanical position)
  initializeServosTo90();
  
  // STEP 2: Wait 2 seconds for robot to stabilize in neutral pose
  Serial.println("Waiting 2 seconds in neutral position...");
  delay(2000);
  
  // STEP 3: Initialize gait system and start walking
  Serial.println("Standing Complete. Starting Walk Forward.");
  setupGait();
  currentState = WALK_FORWARD;
}

void loop() {
  updateGait();
  // Small delay to prevent watchdog trigger and allow servo smoothing
  delay(5); 
}