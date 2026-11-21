// Vigil — ESP32 ascultă vigil/alert: 0 stins, 1 LED aprins, 2 LED clipește, 3 LED clipește rapid + buzzer (opțional).
// WiFi: copiază wifi_secrets.h.example în wifi_secrets.h și completează (fișierul nu se urcă pe git).
#include <WiFi.h>
#include <PubSubClient.h>
#include "wifi_secrets.h"

const char* BROKER_IP = "172.22.118.134";  // IP-ul laptopului cu Mosquitto — singurul loc de schimbat
// ESP32-C3 (SuperMini): LED-ul de pe placă e pe GPIO 8 și se aprinde pe LOW.
// ESP32 clasic: LED_PIN 2, LED_ON HIGH.
const int LED_PIN = 8;
const int LED_ON = LOW;
const int BUZZER_PIN = 4;  // buzzer activ (HIGH = sună)

WiFiClient net;
PubSubClient mqtt(net);
int level = 0;

void onMessage(char* topic, byte* payload, unsigned int len) {
  if (len >= 1 && payload[0] >= '0' && payload[0] <= '3') {
    level = payload[0] - '0';
    Serial.printf("alert %d\n", level);  // confirmare pentru esp32_test.py
  }
}

void ensureConnected() {
  while (WiFi.status() != WL_CONNECTED) {
    Serial.printf("wifi: conectare la %s\n", WIFI_SSID);
    WiFi.disconnect();  // altfel un begin() peste o conectare în curs dă "cannot set config"
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    WiFi.setTxPower(WIFI_POWER_8_5dBm);  // ESP32-C3 SuperMini nu se conectează la putere maximă
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) delay(500);
    if (WiFi.status() == WL_CONNECTED) Serial.printf("wifi: ok %s\n", WiFi.localIP().toString().c_str());
    else Serial.printf("wifi: eșuat, status %d (1 = rețea negăsită, 4 = parolă greșită?)\n", WiFi.status());
  }
  while (!mqtt.connected()) {
    if (mqtt.connect("vigil-esp32")) {
      mqtt.subscribe("vigil/alert");
      Serial.println("mqtt: ok");
    } else {
      Serial.printf("mqtt: eroare %d, reîncerc\n", mqtt.state());
      delay(2000);
      if (WiFi.status() != WL_CONNECTED) return;  // WiFi picat: îl reface ensureConnected data viitoare
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  WiFi.mode(WIFI_STA);
  // diagnostic: rețeaua apare doar dacă emite pe 2.4 GHz
  int n = WiFi.scanNetworks();
  bool seen = false;
  for (int i = 0; i < n; i++) {
    if (WiFi.SSID(i) == WIFI_SSID) {
      seen = true;
      Serial.printf("wifi: văd %s, canal %d, semnal %d dBm\n", WIFI_SSID, WiFi.channel(i), WiFi.RSSI(i));
    }
  }
  if (!seen) Serial.printf("wifi: NU văd %s (%d rețele în jur) — e pe 2.4 GHz?\n", WIFI_SSID, n);
  mqtt.setServer(BROKER_IP, 1883);
  mqtt.setCallback(onMessage);
}

void loop() {
  ensureConnected();
  mqtt.loop();
  bool blink = (millis() / 250) % 2;
  bool fastBlink = (millis() / 60) % 2;  // nivel 3 fără buzzer: tot se vede ca urgență
  bool ledOn = level == 1 || (level == 2 && blink) || (level == 3 && fastBlink);
  digitalWrite(LED_PIN, ledOn ? LED_ON : !LED_ON);
  digitalWrite(BUZZER_PIN, level == 3 && blink);
}
