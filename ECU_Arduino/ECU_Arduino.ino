// Vigil - ESP32-C3 Mini: simulated car actuator (brake + hazard lights) over MQTT
// Board:     "ESP32C3 Dev Module", Tools -> USB CDC On Boot: Enabled
// Libraries: PubSubClient (Nick O'Leary), ArduinoJson (Benoit Blanchon, v7)
//
// Topics
//   IN   vigil/car/cmd       {"cmd":"emergency","id":"e1"}          hazards ON + gentle brake to 0
//                            {"cmd":"emergency","decel_mps2":2.5}  (optional decel, 0.5..4.0)
//                            {"cmd":"cancel_emergency"}             driver OK: stop braking, hazards stay on
//                            {"cmd":"slow_down","target_kmh":0,"decel_mps2":2.0}
//                            {"cmd":"hazard","on":true}
//                            {"cmd":"resume"}   stop braking, keep current speed (refused in emergency)
//                            {"cmd":"reset"}    demo reset: cruise speed, hazards off
//                            any command may carry "id" - it is echoed back in the events
//   IN   vigil/alert         "0".."3"  (alert level from ML2 -> LED when hazards are off: 1 on, 2 blink,
//                                       3 fast blink; plus buzzer if one is wired)
//   OUT  vigil/car/state     {"speed_kmh":72.4,"target_kmh":0,"braking":true,"hazard":true,
//                             "stopped":false,"emergency":true,"alert":3}
//   OUT  vigil/car/event     {"event":"emergency_braking_started","speed_kmh":100,"target_kmh":0,
//                             "decel_mps2":2,"hazard":true,"emergency":true,"id":"e1"}
//                            events: emergency_braking_started, emergency_cancelled, braking_started,
//                                    target_reached, stopped, hazard_on, hazard_off, resumed, reset, error
//   OUT  vigil/esp32/status  "online" / "offline" (retained, last will)

// ===== MODE =====
// 1 = MOCK: no Wi-Fi/broker. Type commands in Serial Monitor (line ending: Newline):
//       emergency | cancel | brake | hazard on | hazard off | resume | reset | 0..3 | {json}
//     Everything the ESP32 would publish is printed as [PUB] topic payload.
// 0 = REAL: Wi-Fi + Mosquitto.
#define MOCK_MODE 0
#define AUTO_DEMO 1            // MOCK only: 5 s after boot, send slow_down + hazard on by itself
// ================

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ---------- NETWORK CONFIG (only used when MOCK_MODE 0) ----------
// Wi-Fi name + password live in wifi_secrets.h next to this file (NOT in git, the repo is public).
// First time: copy wifi_secrets.h.example to wifi_secrets.h and fill it in.
#include "wifi_secrets.h"                         // defines WIFI_SSID and WIFI_PASSWORD (2.4 GHz hotspot)
const char* MQTT_HOST     = "172.22.118.20";     // IP of the PC running Mosquitto
const uint16_t MQTT_PORT  = 1883;
const char* MQTT_USER     = "";
const char* MQTT_PASS     = "";
// ------------------------------------------------------------------

const char* TOPIC_CMD    = "vigil/car/cmd";
const char* TOPIC_ALERT  = "vigil/alert";
const char* TOPIC_STATE  = "vigil/car/state";
const char* TOPIC_EVENT  = "vigil/car/event";
const char* TOPIC_STATUS = "vigil/esp32/status";

// ---------- PINS (-1 = not connected) ----------
const int  HAZARD_PIN        = 8;      // onboard LED on most C3 Mini boards
const bool HAZARD_ACTIVE_LOW = true;   // onboard LED lights when pin is LOW
const int  BRAKE_PIN         = -1;     // brake light LED, e.g. 4
const bool BRAKE_ACTIVE_LOW  = false;
const int  BUZZER_PIN        = -1;     // active buzzer, e.g. 5
// -----------------------------------------------

// ---------- SIMULATED CAR ----------
const float CRUISE_KMH       = 100.0;
const float DEFAULT_DECEL    = 2.0;    // m/s^2 - gentle braking (~14 s from 100 to 0)
const float MIN_DECEL        = 0.5;
const float MAX_DECEL        = 4.0;    // never allow harsh braking from a command
const unsigned long HAZARD_BLINK_MS = 400;
const unsigned long STATE_MS_BRAKING = 500;   // publish state often while braking
const unsigned long STATE_MS_IDLE    = 5000;  // heartbeat otherwise

struct Car {
  float speed    = CRUISE_KMH;
  float target   = CRUISE_KMH;
  float decel    = DEFAULT_DECEL;
  bool  braking  = false;
  bool  hazard   = false;
  bool  stopped  = false;
  bool  emergency = false;
} car;
int alertLevel = 0;
String currentCmdId = "";   // "id" of the command being handled, echoed in events
// -----------------------------------

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);
String clientId;
unsigned long lastWifiAttempt = 0, lastMqttAttempt = 0;

float round1(float x) { return roundf(x * 10.0f) / 10.0f; }

void setOutput(int pin, bool on, bool activeLow) {
  if (pin < 0) return;
  digitalWrite(pin, (on != activeLow) ? HIGH : LOW);
}

// ================= PUBLISH (mock prints, real sends) =================
void publish(const char* topic, const String& payload, bool retain = false) {
#if MOCK_MODE
  Serial.printf("[PUB] %s %s\n", topic, payload.c_str());
#else
  if (mqtt.connected()) mqtt.publish(topic, payload.c_str(), retain);
#endif
}

void sendEvent(const char* event, const char* detail = nullptr) {
  JsonDocument d;
  d["event"]      = event;
  d["speed_kmh"]  = round1(car.speed);
  d["target_kmh"] = round1(car.target);
  d["braking"]    = car.braking;
  d["hazard"]     = car.hazard;
  d["emergency"]  = car.emergency;
  if (car.braking) d["decel_mps2"] = car.decel;
  if (detail) d["detail"] = detail;
  if (currentCmdId.length()) d["id"] = currentCmdId;
  String s; serializeJson(d, s);
  publish(TOPIC_EVENT, s);
  Serial.printf("[EVENT] %s\n", s.c_str());
}

void publishState() {
  JsonDocument d;
  d["speed_kmh"]  = round1(car.speed);
  d["target_kmh"] = round1(car.target);
  d["braking"]    = car.braking;
  d["hazard"]     = car.hazard;
  d["stopped"]    = car.stopped;
  d["emergency"]  = car.emergency;
  d["alert"]      = alertLevel;
  String s; serializeJson(d, s);
  publish(TOPIC_STATE, s);
}

// ================= COMMANDS =================
void handleCommand(const String& json) {
  JsonDocument doc;
  if (deserializeJson(doc, json)) { sendEvent("error", "invalid_json"); return; }

  String cmd = doc["cmd"] | "";
  currentCmdId = doc["id"] | "";

  if (cmd == "emergency") {
    // Emergency protocol: hazards first, then gentle braking to a full stop
    car.emergency = true;
    car.hazard    = true;
    car.target    = 0.0f;
    car.decel     = constrain(doc["decel_mps2"] | DEFAULT_DECEL, MIN_DECEL, MAX_DECEL);
    car.braking   = car.speed > 0.0f;
    car.stopped   = !car.braking;
    if (alertLevel < 3) alertLevel = 3;
    Serial.println("[EMERGENCY] hazards ON, braking to 0");
    sendEvent(car.braking ? "emergency_braking_started" : "stopped");

  } else if (cmd == "cancel_emergency") {
    // Driver responded / server cancels: stop braking, keep current speed, hazards stay on
    car.emergency = false;
    car.braking   = false;
    car.target    = car.speed;
    sendEvent("emergency_cancelled");

  } else if (cmd == "slow_down") {
    if (car.emergency) { sendEvent("error", "emergency_active"); currentCmdId = ""; return; }
    car.target  = constrain(doc["target_kmh"] | 0.0f, 0.0f, car.speed);
    car.decel   = constrain(doc["decel_mps2"] | DEFAULT_DECEL, MIN_DECEL, MAX_DECEL);
    car.braking = car.speed > car.target;
    car.stopped = car.speed <= 0.0f;
    sendEvent(car.braking ? "braking_started" : "target_reached");

  } else if (cmd == "hazard") {
    bool on = doc["on"] | true;
    if (!on && car.emergency) { sendEvent("error", "emergency_active"); currentCmdId = ""; return; }
    car.hazard = on;
    if (!car.hazard) setOutput(HAZARD_PIN, false, HAZARD_ACTIVE_LOW);
    sendEvent(car.hazard ? "hazard_on" : "hazard_off");

  } else if (cmd == "resume") {
    if (car.emergency) { sendEvent("error", "emergency_active"); currentCmdId = ""; return; }
    car.braking = false;
    car.target  = car.speed;
    sendEvent("resumed");

  } else if (cmd == "reset") {
    car = Car();
    setOutput(HAZARD_PIN, false, HAZARD_ACTIVE_LOW);
    sendEvent("reset");

  } else {
    sendEvent("error", "unknown_cmd");
    currentCmdId = "";
    return;
  }
  publishState();
  // keep the id while an emergency/brake is running, so "stopped" carries it too
  if (!car.braking) currentCmdId = "";
}

void handleAlert(int level) {
  if (level < 0 || level > 3) { sendEvent("error", "invalid_alert"); return; }
  alertLevel = level;
  Serial.printf("[ALERT] level = %d\n", level);
}

// ================= SIMULATION / OUTPUTS =================
void updateCar() {
  static unsigned long last = millis();
  unsigned long now = millis();
  float dt = (now - last) / 1000.0f;
  last = now;

  if (car.braking) {
    car.speed -= car.decel * 3.6f * dt;          // m/s^2 -> km/h per second
    if (car.speed <= car.target) {
      car.speed = car.target;
      car.braking = false;
      if (car.speed <= 0.0f) {
        car.speed = 0.0f;
        car.stopped = true;
        sendEvent("stopped");
      } else {
        sendEvent("target_reached");
      }
      publishState();
      currentCmdId = "";
    }
  }

  // Periodic state
  static unsigned long lastState = 0;
  unsigned long period = car.braking ? STATE_MS_BRAKING : STATE_MS_IDLE;
  if (now - lastState >= period) { lastState = now; publishState(); }
}

void updateOutputs() {
  unsigned long now = millis();

  // Hazard lights blink. Otherwise the same LED shows the alert level sent by ML2 on vigil/alert
  // (no buzzer is wired, so without this the alert levels would have no visible effect)
  if (car.hazard) {
    bool on = (now / HAZARD_BLINK_MS) % 2 == 0;
    setOutput(HAZARD_PIN, on, HAZARD_ACTIVE_LOW);
  } else {
    bool on = alertLevel == 1 || (alertLevel == 2 && (now / 250) % 2 == 0) || (alertLevel == 3 && (now / 60) % 2 == 0);
    setOutput(HAZARD_PIN, on, HAZARD_ACTIVE_LOW);
  }

  // Brake light while braking
  setOutput(BRAKE_PIN, car.braking, BRAKE_ACTIVE_LOW);

  // Buzzer: 2 = short beeps, 3 = fast beeps
  if (BUZZER_PIN >= 0) {
    bool on = false;
    if (alertLevel == 2) on = (now % 1000) < 100;
    if (alertLevel == 3) on = (now % 300)  < 150;
    digitalWrite(BUZZER_PIN, on ? HIGH : LOW);
  }
}

// ================= MOCK INPUT =================
void mockLoop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length()) {
      Serial.printf("[IN] %s\n", line.c_str());
      if      (line.startsWith("{"))     handleCommand(line);
      else if (line == "emergency")      handleCommand("{\"cmd\":\"emergency\",\"id\":\"mock\"}");
      else if (line == "cancel")         handleCommand("{\"cmd\":\"cancel_emergency\"}");
      else if (line == "brake")          handleCommand("{\"cmd\":\"slow_down\"}");
      else if (line == "hazard on")      handleCommand("{\"cmd\":\"hazard\",\"on\":true}");
      else if (line == "hazard off")     handleCommand("{\"cmd\":\"hazard\",\"on\":false}");
      else if (line == "resume")         handleCommand("{\"cmd\":\"resume\"}");
      else if (line == "reset")          handleCommand("{\"cmd\":\"reset\"}");
      else if (isDigit(line[0]))         handleAlert(line.toInt());
      else Serial.println("  commands: emergency | cancel | brake | hazard on | hazard off | resume | reset | 0..3 | {json}");
    }
  }

#if AUTO_DEMO
  static bool demoDone = false;
  if (!demoDone && millis() > 5000) {
    demoDone = true;
    Serial.println("[DEMO] simulating server: emergency protocol");
    handleCommand("{\"cmd\":\"emergency\",\"id\":\"demo\"}");
  }
#endif
}

// ================= REAL (Wi-Fi + MQTT) =================
void scanNetworks() {
  Serial.println("[WiFi] Networks the ESP32 can see (2.4 GHz only):");
  int n = WiFi.scanNetworks();
  if (n <= 0) Serial.println("   none found");
  for (int i = 0; i < n; i++) {
    Serial.printf("   \"%s\"  %d dBm  ch %d\n", WiFi.SSID(i).c_str(), WiFi.RSSI(i), WiFi.channel(i));
  }
  WiFi.scanDelete();
}

void connectWifi() {
  Serial.printf("[WiFi] Connecting to \"%s\"", WIFI_SSID);
  WiFi.disconnect(true);                 // stop any previous attempt ("sta is connecting" warning)
  delay(100);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(false);          // we handle retries ourselves in realLoop()
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);    // C3 Super Mini fix: full power often fails to connect
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) { delay(500); Serial.print("."); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WiFi] Connected, IP: %s, RSSI %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    // 1 = network not found (name wrong / 5 GHz / too far), 4 or 6 = usually wrong password
    Serial.printf("[WiFi] Failed, status=%d, will retry\n", WiFi.status());
    scanNetworks();
  }
}

void onMessage(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  msg.trim();
  Serial.printf("[MQTT] %s -> %s\n", topic, msg.c_str());

  if      (strcmp(topic, TOPIC_CMD) == 0)   handleCommand(msg);
  else if (strcmp(topic, TOPIC_ALERT) == 0) handleAlert(msg.toInt());
}

bool connectMqtt() {
  Serial.printf("[MQTT] Connecting to %s:%u ... ", MQTT_HOST, MQTT_PORT);
  bool ok = mqtt.connect(clientId.c_str(),
                         strlen(MQTT_USER) ? MQTT_USER : nullptr,
                         strlen(MQTT_PASS) ? MQTT_PASS : nullptr,
                         TOPIC_STATUS, 1, true, "offline");
  if (ok) {
    Serial.println("OK");
    mqtt.publish(TOPIC_STATUS, "online", true);
    mqtt.subscribe(TOPIC_CMD, 1);
    mqtt.subscribe(TOPIC_ALERT, 1);
    publishState();
  } else {
    Serial.printf("failed, state=%d\n", mqtt.state());   // -2 unreachable, 5 not authorized
  }
  return ok;
}

void realLoop() {
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWifiAttempt > 5000) { lastWifiAttempt = millis(); connectWifi(); }
    return;
  }
  if (!mqtt.connected()) {
    if (millis() - lastMqttAttempt > 3000) { lastMqttAttempt = millis(); connectMqtt(); }
    return;
  }
  mqtt.loop();
}

// ================= MAIN =================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== Vigil ESP32-C3 car actuator ===");

  if (HAZARD_PIN >= 0) { pinMode(HAZARD_PIN, OUTPUT); setOutput(HAZARD_PIN, false, HAZARD_ACTIVE_LOW); }
  if (BRAKE_PIN  >= 0) { pinMode(BRAKE_PIN, OUTPUT);  setOutput(BRAKE_PIN, false, BRAKE_ACTIVE_LOW); }
  if (BUZZER_PIN >= 0) { pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW); }

#if MOCK_MODE
  Serial.println("MOCK MODE. Commands: emergency | cancel | brake | hazard on | hazard off | resume | reset | 0..3 | {json}");
#else
  clientId = "vigil-esp32-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setKeepAlive(15);
  mqtt.setBufferSize(512);
  connectWifi();
#endif
}

void loop() {
#if MOCK_MODE
  mockLoop();
#else
  realLoop();
#endif
  updateCar();      // runs in both modes, even while reconnecting
  updateOutputs();
}