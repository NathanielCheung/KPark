/*
 * Parking Lot + Street Sensor (ESP32 or compatible)
 * - Beamish Munro Hall: FSR entry/exit, LED1-3 = occupancy (0-3), send available spots to site
 * - Clergy St W: ultrasonic, LED4 = occupied/vacant, send 0 or 1 available to site
 * Serial/Bluetooth output: PARK,lotId,availableSpots so a bridge app can forward to website.
 */

#include "BluetoothSerial.h"
BluetoothSerial BTSerial;
// ----------------------------
// Pin definitions
// ----------------------------
const int FSR1_PIN = A0;   // Entry sensor
const int FSR2_PIN = A1;   // Exit sensor

const int LED1_PIN = 2;    // Parking occupancy LED 1
const int LED2_PIN = 3;    // Parking occupancy LED 2
const int LED3_PIN = 4;    // Parking occupancy LED 3
const int LED4_PIN = 5;    // Ultrasonic vacancy LED

const int ULTRASONIC_TRIG_PIN = 9;
const int ULTRASONIC_ECHO_PIN = 10;

// ----------------------------
// Constants
// ----------------------------
const int TOUCH_THRESHOLD = 300;
const unsigned long MAX_TIME_BETWEEN_SENSORS = 2000;
const unsigned long DEBOUNCE_TIME = 50;
const float MAX_DISTANCE = 100.0;
const float OBJECT_DISTANCE_THRESHOLD = 10.0;
const int MAX_SPOTS = 3;

// ----------------------------
// Variables
// ----------------------------
float duration;
float distance;

enum CarDirection { NONE, ENTERING, EXITING };
CarDirection currentDirection = NONE;

bool fsr1Active = false, fsr2Active = false;
bool fsr1Processed = false, fsr2Processed = false;
bool fsr1WasPressed = false, fsr2WasPressed = false;
unsigned long fsr1PressTime = 0, fsr2PressTime = 0;

int vehiclesEntered = 0, vehiclesExited = 0;
int currentOccupancy = 0;

// For website: send PARK,lotId,availableSpots when these change (serial bridge forwards to API)
int lastBeamishAvailable = -1;
int lastClergyAvailable = -1;
unsigned long lastClergySendTime = 0;

// ----------------------------
// Helper: Update parking LEDs
// ----------------------------
void updateParkingLEDs() {
  digitalWrite(LED1_PIN, currentOccupancy >= 1 ? HIGH : LOW);
  digitalWrite(LED2_PIN, currentOccupancy >= 2 ? HIGH : LOW);
  digitalWrite(LED3_PIN, currentOccupancy >= 3 ? HIGH : LOW);
}

// Print PARK line so serial/BT bridge can POST to website
void sendToWebsite(const char* lotId, int availableSpots) {
  // USB serial (for debugging or a wired bridge)
  Serial.print("PARK,");
  Serial.print(lotId);
  Serial.print(",");
  Serial.println(availableSpots);

  // Bluetooth Serial (for wireless bridge)
  if (BTSerial.hasClient()) {
    BTSerial.print("PARK,");
    BTSerial.print(lotId);
    BTSerial.print(",");
    BTSerial.println(availableSpots);
  }
}

// ----------------------------
// Setup
// ----------------------------
void setup() {
  Serial.begin(9600);
  BTSerial.begin("QHacks_Meter");  // Bluetooth name shown when pairing

  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(LED3_PIN, OUTPUT);
  pinMode(LED4_PIN, OUTPUT);
  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);

  updateParkingLEDs();
  digitalWrite(LED4_PIN, HIGH);   // assume vacant at start

  sendToWebsite("beamish-munro-hall", MAX_SPOTS - currentOccupancy);
  sendToWebsite("clergy-st-w", 1);
  lastBeamishAvailable = MAX_SPOTS - currentOccupancy;
  lastClergyAvailable = 1;

  Serial.println("Parking Lot Tracking System Ready");
  Serial.println("D2-D4 = Occupancy | D5 = Ultrasonic Vacancy");
}

// ----------------------------
// Main Loop
// ----------------------------
void loop() {
  unsigned long currentTime = millis();

  int fsr1Value = analogRead(FSR1_PIN);
  int fsr2Value = analogRead(FSR2_PIN);

  // ---------- FSR1 Edge Detection ----------
  if (fsr1Value > TOUCH_THRESHOLD) {
    if (!fsr1Active && !fsr1WasPressed) {
      fsr1Active = true;
      fsr1WasPressed = true;
      fsr1PressTime = currentTime;
      fsr1Processed = true;
      Serial.println("FSR1 Pressed");
    }
  } else {
    fsr1Active = false;
    if (fsr1WasPressed && currentTime - fsr1PressTime > DEBOUNCE_TIME)
      fsr1WasPressed = false;
  }

  // ---------- FSR2 Edge Detection ----------
  if (fsr2Value > TOUCH_THRESHOLD) {
    if (!fsr2Active && !fsr2WasPressed) {
      fsr2Active = true;
      fsr2WasPressed = true;
      fsr2PressTime = currentTime;
      fsr2Processed = true;
      Serial.println("FSR2 Pressed");
    }
  } else {
    fsr2Active = false;
    if (fsr2WasPressed && currentTime - fsr2PressTime > DEBOUNCE_TIME)
      fsr2WasPressed = false;
  }

  // ---------- Direction Logic (Beamish) ----------
  if (fsr1Processed && fsr2Processed) {
    if (fsr1PressTime < fsr2PressTime &&
        fsr2PressTime - fsr1PressTime <= MAX_TIME_BETWEEN_SENSORS) {
      currentDirection = ENTERING;
      vehiclesEntered++;
      if (currentOccupancy < MAX_SPOTS) currentOccupancy++;
      updateParkingLEDs();
      Serial.println("=== ENTERING ===");
    } else if (fsr2PressTime < fsr1PressTime &&
               fsr1PressTime - fsr2PressTime <= MAX_TIME_BETWEEN_SENSORS) {
      currentDirection = EXITING;
      vehiclesExited++;
      if (currentOccupancy > 0) currentOccupancy--;
      updateParkingLEDs();
      Serial.println("=== EXITING ===");
    }
    fsr1Processed = false;
    fsr2Processed = false;
    fsr1WasPressed = false;
    fsr2WasPressed = false;
  }

  // Send Beamish to website when occupancy changes (light on = car parked = fewer spots)
  int beamishAvailable = MAX_SPOTS - currentOccupancy;
  if (beamishAvailable != lastBeamishAvailable) {
    lastBeamishAvailable = beamishAvailable;
    sendToWebsite("beamish-munro-hall", beamishAvailable);
  }

  // ---------- Ultrasonic (Clergy St) ----------
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);

  duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, 30000);
  distance = (duration * 0.0343f) / 2.0f;

  // LED4: on when occupied (object close), off when vacant. Website: 0 available when occupied, 1 when vacant.
  int clergyAvailable;
  if (distance > 0 && distance <= OBJECT_DISTANCE_THRESHOLD) {
    digitalWrite(LED4_PIN, HIGH);   // Occupied -> spots decrease
    clergyAvailable = 0;
  } else {
    digitalWrite(LED4_PIN, LOW);    // Vacant -> spots go back up
    clergyAvailable = 1;
  }
  if (clergyAvailable != lastClergyAvailable) {
    lastClergyAvailable = clergyAvailable;
    sendToWebsite("clergy-st-w", clergyAvailable);
  }
  // Re-send Clergy every 3s so website stays in sync if bridge connected late
  if (currentTime - lastClergySendTime >= 3000) {
    lastClergySendTime = currentTime;
    sendToWebsite("clergy-st-w", clergyAvailable);
  }

  // ---------- Debug ----------
  Serial.print("Occ: ");
  Serial.print(currentOccupancy);
  Serial.print("/");
  Serial.print(MAX_SPOTS);
  Serial.print(" | Ultrasonic: ");
  Serial.print(distance);
  Serial.print("cm | Vacancy LED: ");
  Serial.println(digitalRead(LED4_PIN));

  delay(50);
}
