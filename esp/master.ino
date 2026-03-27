#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

#define SERVOMIN 150
#define SERVOMAX 600
#define SERVO_FREQ 60

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

int servoChannels[] = {0, 2, 4, 6, 8, 10, 12, 14};

void setServoAngle(int channel, int angle) {
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}

void stand() {
  for (int i = 0; i < 8; i += 2) {
    setServoAngle(servoChannels[i], 90);     // knees
    setServoAngle(servoChannels[i + 1], 90); // hips
  }
}

void stepForward() {
  // hz zmr
  setServoAngle(servoChannels[0], 60);
  setServoAngle(servoChannels[4], 60);
  delay(100);
  // di l9dam
  setServoAngle(servoChannels[1], 110);
  setServoAngle(servoChannels[5], 110);
  delay(100);
  // hbt zmr
  setServoAngle(servoChannels[0], 95);
  setServoAngle(servoChannels[4], 95);
  delay(100);
  // df3 lmrd
  setServoAngle(servoChannels[1], 60);
  setServoAngle(servoChannels[5], 60);
}

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22);

  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);

  delay(500);

  Serial.println("Setting all servos to 90°");

  stand();

  delay(2000);
  stepForward();
}

void loop() {}