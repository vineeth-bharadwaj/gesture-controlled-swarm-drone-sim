#include <Arduino_LSM6DS3.h>
#include <math.h>

#define MIN_PULSE_WIDTH 1000
#define MAX_PULSE_WIDTH 2000

// Complementary filter constant (adjust between 0 and 1)
#define ALPHA 0.05

// Smoothing factor for PWM relaxation (0.0 = no change, 1.0 = immediate snap)
#define PWM_SMOOTHING 0.1

// Timing variables
uint32_t LoopTimer;
unsigned long prevTime = 0;

// Fused angles (in degrees)
float rollFusion = 0.0;
float pitchFusion = 0.0;

// PWM output values (using float for gradual adjustments; eventually mapped to int)
float throttlePWM = 1000;  // Default to minimum (off)
float rollPWM     = 1500;  // Center position
float pitchPWM    = 1500;  // Center position
float yawPWM      = 1500;  // Center position

// Button pins and states
const int buttonPins[] = {2, 3};
int buttonStates[] = {HIGH, HIGH};  // Default state HIGH (INPUT_PULLUP)

// Flag for continuous data transmission
bool continuousMode = false;

// Flag for IMU data mode
bool imuDataMode = false;

void setup() {
  Serial.begin(115200);
  
  // Don't wait for serial in final deployment
  // while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.println("IMU initialized successfully");

  // Initialize button pins as INPUT_PULLUP
  for (int i = 0; i < 2; i++) {
    pinMode(buttonPins[i], INPUT_PULLUP);
  }

  LoopTimer = micros();
  prevTime = micros();
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    switch(cmd) {
      case 'g':
        // Reset mode flags to ensure consistent state
        imuDataMode = false;
        continuousMode = false;
        
        // Send drone control data once when requested
        sendDroneData();
        break;
      case 'i':
        // Toggle IMU data mode and reset other modes
        imuDataMode = true;
        continuousMode = false;
        sendIMUData();
        break;
      case 'c':
        // Toggle continuous mode
        continuousMode = !continuousMode;
        imuDataMode = false;
        Serial.print("Continuous mode: ");
        Serial.println(continuousMode ? "ON" : "OFF");
        break;
      case 'x':
        // Exit IMU data mode and reset to default control
        imuDataMode = false;
        continuousMode = false;
        // Reset PWM values to neutral
        throttlePWM = 1500;
        rollPWM = 1500;
        pitchPWM = 1500;
        yawPWM = 1500;
        break;
    }
  }

  // Update button states
  for (int i = 0; i < 2; i++) {
    buttonStates[i] = digitalRead(buttonPins[i]);
  }

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax, ay, az;
    float gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);  // Gyro data in degrees per second

    // Calculate time delta in seconds
    unsigned long currentTime = micros();
    float dt = (currentTime - prevTime) / 1000000.0;
    prevTime = currentTime;

    // Calculate angles from the accelerometer (in degrees)
    float accelRoll  = atan2(ay, az) * 180.0 / PI;
    float accelPitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    accelRoll = -accelRoll;  // Invert roll if needed

    // Gyroscope rates (assumed: gx -> roll, gy -> pitch)
    float gyroRollRate  = gx;
    float gyroPitchRate = gy;

    // Complementary filter fusion
    rollFusion  = (1 - ALPHA) * (rollFusion + gyroRollRate * dt) + ALPHA * accelRoll;
    pitchFusion = (1 - ALPHA) * (pitchFusion + gyroPitchRate * dt) + ALPHA * accelPitch;

    // Normal control logic (only if not in IMU data mode)
    if (!imuDataMode) {
      // Roll and Pitch control logic (unchanged from previous version)
      if (buttonStates[1] == LOW) {
        float mappedRoll  = map(rollFusion, -90, 90, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        float mappedPitch = map(pitchFusion, -90, 90, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        mappedRoll  = constrain(mappedRoll, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        mappedPitch = constrain(mappedPitch, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);

        // Apply relaxation around center (hover) values for roll
        if (mappedRoll > 1570) mappedRoll -= 70;
        else if (mappedRoll < 1430) mappedRoll += 70;
        else mappedRoll = 1500;

        if (mappedPitch > 1570) mappedPitch -= 70;
        else if (mappedPitch < 1430) mappedPitch += 70;
        else mappedPitch = 1500;

        rollPWM  = mappedRoll;
        pitchPWM = mappedPitch;
      } else {
        // Gradually move roll and pitch PWM towards 1500
        rollPWM  += PWM_SMOOTHING * (1500 - rollPWM);
        pitchPWM += PWM_SMOOTHING * (1500 - pitchPWM);
      }

      // Throttle and Yaw control logic (unchanged from previous version)
      if (buttonStates[0] == LOW) {
        float mappedThrottle = map(rollFusion, -90, 90, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        float mappedYaw      = map(pitchFusion, -90, 90, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        mappedThrottle = constrain(mappedThrottle, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        mappedYaw      = constrain(mappedYaw, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);

        // Apply relaxation around center (hover) values
        if (mappedThrottle > 1570) mappedThrottle -= 70;
        else if (mappedThrottle < 1430) mappedThrottle += 70;
        else mappedThrottle = 1500;

        if (mappedYaw > 1570) mappedYaw -= 70;
        else if (mappedYaw < 1430) mappedYaw += 70;
        else mappedYaw = 1500;

        throttlePWM = mappedThrottle;
        yawPWM      = mappedYaw;
      } else {
        // Gradually move throttle and yaw PWM towards 1500
        throttlePWM += PWM_SMOOTHING * (1500 - throttlePWM);
        yawPWM += PWM_SMOOTHING * (1500 - yawPWM);
      }
    }

    // Maintain a loop cycle of at least 10 ms
    while (micros() - LoopTimer < 10000) {
      // busy-wait until 4ms have passed
    }
    LoopTimer = micros();

    // If in continuous mode, send data every loop
    if (continuousMode) {
      sendDroneData();
    }
  }
}

void sendDroneData() {
  // Format the data as expected by the Python code
  // Convert PWM values from 1000-2000 to 1.0-2.0 format
  Serial.print("T:"); Serial.print(throttlePWM/1000.0, 2);
  Serial.print(",R:"); Serial.print(pitchPWM/1000.0, 2);
  Serial.print(",P:"); Serial.print(rollPWM/1000.0, 2);
  Serial.print(",Y:"); Serial.print(yawPWM/1000.0, 2);
  Serial.println();
}

void sendIMUData() {
  // Send raw IMU data when in IMU mode
  float ax, ay, az;
  float gx, gy, gz;
  
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  Serial.print(micros() / 1000.0);  // Timestamp in milliseconds
  Serial.print(", ");
  Serial.print(ax, 6);  // Acceleration x-axis
  Serial.print(", ");
  Serial.print(ay, 6);  // Acceleration y-axis
  Serial.print(", ");
  Serial.print(az, 6);  // Acceleration z-axis
  Serial.print(", ");
  Serial.print(gx, 6);  // Gyroscope x-axis
  Serial.print(", ");
  Serial.print(gy, 6);  // Gyroscope y-axis
  Serial.print(", ");
  Serial.println(gz, 6);  // Gyroscope z-axis
}
