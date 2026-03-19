#include <ArduinoHttpClient.h>
#include <SPI.h>
#include <WiFiNINA.h>


// ============================================================================
// CONFIGURATION
// ============================================================================

// WiFi Credentials
const char WIFI_SSID[] = "YOUR_WIFI_SSID";
const char WIFI_PASSWORD[] = "YOUR_WIFI_PASSWORD";

// API Configuration
// NOTE: Use the local IP address of the PC running the backend, not "localhost"
const char API_SERVER[] = "192.168.1.100";
int API_PORT = 3001;
const String API_PATH = "/api/parking";

// Lot Configuration
const String LOT_ID = "lot-a"; // Identifier for this parking lot
const int DISTANCE_THRESHOLD_CM =
    50; // Distance below this means spot is occupied

// Sensor Configuration
const int NUM_SENSORS = 2; // Total number of ultrasonic sensors

// Pin Definitions for each sensor: {TRIG_PIN, ECHO_PIN}
const int SENSOR_PINS[NUM_SENSORS][2] = {
    {7, 6}, // Sensor 1
    {5, 4}  // Sensor 2
};

// ============================================================================
// GLOBALS & STATE
// ============================================================================

int lastAvailableSpots =
    -1; // Track the last sent availability count to debounce

WiFiClient wifi;
HttpClient client = HttpClient(wifi, API_SERVER, API_PORT);

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB port only)
  }

  Serial.println("Initializing KPark Sensor node...");

  // 1. Initialize sensor pins
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(SENSOR_PINS[i][0], OUTPUT); // Set TRIG as Output
    pinMode(SENSOR_PINS[i][1], INPUT);  // Set ECHO as Input
  }

  // 2. Connect to WiFi network
  connectWiFi();
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
  // 1. Ensure WiFi stays connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection lost. Reconnecting...");
    connectWiFi();
  }

  int availableSpots = 0;

  // 2. Read from each sensor and determine availability
  for (int i = 0; i < NUM_SENSORS; i++) {
    long distance = measureDistance(SENSOR_PINS[i][0], SENSOR_PINS[i][1]);

    Serial.print("Sensor ");
    Serial.print(i + 1);
    Serial.print(" -> Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    // A distance of 0 often means out-of-range (no echo received), so spot is
    // empty. Otherwise, if the distance is greater than or equal to the
    // threshold, it is empty.
    if (distance == 0 || distance >= DISTANCE_THRESHOLD_CM) {
      availableSpots++;
    }
  }

  Serial.print("--> Total Available Spots: ");
  Serial.println(availableSpots);
  Serial.println("----------------------------------------");

  // 3. Debounce: Only send POST request if the availability count has changed
  if (availableSpots != lastAvailableSpots) {
    bool success = postAvailability(availableSpots);

    // If the POST was successful (or at least attempted and didn't crash),
    // update the state so we don't spam the API with the same data.
    lastAvailableSpots = availableSpots;
  }

  // 4. Wait ~500ms before taking the next reading
  delay(500);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Measures the distance using an HC-SR04 ultrasonic sensor.
 *
 * @param trigPin The pin connected to the sensor's TRIG.
 * @param echoPin The pin connected to the sensor's ECHO.
 * @return The measured distance in centimeters. Returns 0 if out of range.
 */
long measureDistance(int trigPin, int echoPin) {
  // Ensure the TRIG pin is clear
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Send a 10-microsecond HIGH pulse to trigger the sensor
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the ECHO pin; returns the sound wave travel time in microseconds.
  // We use a 30ms timeout which corresponds to a max distance of ~5 meters.
  long duration = pulseIn(echoPin, HIGH, 30000);

  if (duration == 0) {
    return 0; // Sensor timed out (nothing in range)
  }

  // Calculate the distance:
  // Speed of sound is ~0.034 cm/microsecond.
  // We divide by 2 because the sound wave travels to the object and back.
  long distance = (duration * 0.034) / 2;

  return distance;
}

/**
 * Connects to the configured WiFi network and blocks until successful.
 */
void connectWiFi() {
  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(WIFI_SSID);

  // Loop until we connect
  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    delay(2000); // Wait 2 seconds before retrying
  }

  Serial.println("\nSuccessfully connected to the network!");
  Serial.print("Local IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("----------------------------------------");
}

/**
 * Sends a POST request to the backend REST API with the updated availability.
 *
 * @param spots The current number of available parking spots.
 * @return true if the POST request completed, false otherwise.
 */
bool postAvailability(int spots) {
  Serial.println("Availability changed! Sending POST request to backend...");

  // Construct the JSON payload string manually
  // Example: {"lotId":"lot-a","availableSpots":2}
  String postData =
      "{\"lotId\":\"" + LOT_ID + "\",\"availableSpots\":" + String(spots) + "}";
  String contentType = "application/json";

  Serial.print("Payload: ");
  Serial.println(postData);

  // Send the HTTP POST request
  client.post(API_PATH, contentType, postData);

  // Read the response from the server
  int statusCode = client.responseStatusCode();
  String response = client.responseBody();

  Serial.print("HTTP Status Code: ");
  Serial.println(statusCode);
  Serial.print("Server Response: ");
  Serial.println(response);

  // If status is positive, it completed the request.
  return statusCode > 0;
}
