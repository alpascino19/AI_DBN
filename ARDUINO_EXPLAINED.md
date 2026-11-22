# Vigil — The Arduino / ESP32 Hardware, Explained

This document explains the embedded (Arduino) part of AI_DBN (Vigil): the two ESP32-C3 sketches, what each one does, the MQTT topics they use, how the simulated car works, how to configure, flash and test the boards, and the C fuzzy table that ML1 generates for an ECU.

> Related docs: `ML2/README.md` §2.4 and §4.7 (Romanian), `ML1/docs/PROBABILITY_ECU.md` §3. Companion files in this folder: `VISION_EXPLAINED.md`, `ML1_EXPLAINED.md`, `ML2_EXPLAINED.md`.

---

## Table of contents

1. [Overview: three embedded pieces](#1-overview-three-embedded-pieces)
2. [Hardware and tools](#2-hardware-and-tools)
3. [Where they sit in the system](#3-where-they-sit-in-the-system)
4. [Sketch 1 — `vigil_alert.ino` (alert LED / buzzer)](#4-sketch-1--vigil_alertino-alert-led--buzzer)
5. [Sketch 2 — `ECU_Arduino.ino` (simulated car actuator)](#5-sketch-2--ecu_arduinoino-simulated-car-actuator)
6. [The C fuzzy lookup table (`ML1/ecu/`)](#6-the-c-fuzzy-lookup-table-ml1ecu)
7. [Configuration: Wi-Fi, broker IP, pins](#7-configuration-wi-fi-broker-ip-pins)
8. [Building and flashing](#8-building-and-flashing)
9. [Testing](#9-testing)
10. [Embedded concepts used](#10-embedded-concepts-used)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limitations and things to check](#12-known-limitations-and-things-to-check)

---

## 1. Overview: three embedded pieces

| Piece | Location | Role |
|---|---|---|
| **Alert board** | `ML2/esp32/vigil_alert/vigil_alert.ino` | the original, minimal sketch: listens to `vigil/alert` (0–3) and drives an LED + optional buzzer — the physical "warning light" in the car |
| **Car actuator (ECU simulator)** | `ECU_Arduino/ECU_Arduino.ino` | a richer sketch: simulates a car (speed, braking, hazard lights), obeys commands on `vigil/car/cmd`, reports `vigil/car/state` / `vigil/car/event`, and also shows the alert level on the LED when hazards are off |
| **Fuzzy ECU table** | `ML1/ecu/vigil_fuzzy_lut.h` + `test_lut.c` | not a sketch: a generated C99 header that lets any microcontroller run ML1's fuzzy speed controller without Python or an LLM |

The ECU simulator is a superset of the alert board (it also listens to `vigil/alert`), so one board can do both jobs at the demo.

---

## 2. Hardware and tools

| Item | Detail |
|---|---|
| Board | **ESP32-C3** (e.g. *SuperMini*), RISC-V, Wi-Fi 2.4 GHz only, USB-CDC serial |
| Onboard LED | **GPIO 8, active LOW** (lights when the pin is LOW) |
| Optional buzzer | active buzzer, GPIO 4 in `vigil_alert`; not wired by default in `ECU_Arduino` (`BUZZER_PIN = -1`) |
| Optional brake light | an LED on a free GPIO (`BRAKE_PIN`, −1 = not wired) |
| Libraries | **PubSubClient** (Nick O'Leary) for MQTT; **ArduinoJson v7** (Benoît Blanchon) for `ECU_Arduino` |
| Toolchain | **arduino-cli** (or Arduino IDE) with the `esp32:esp32` core |
| Board target | `--fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc` (USB CDC On Boot = Enabled), e.g. on COM13 |
| Network | the team's own phone hotspot on **2.4 GHz**; Mosquitto broker on a laptop |

For a classic ESP32 (not C3): `--fqbn esp32:esp32:esp32`, LED on GPIO 2, active HIGH.

---

## 3. Where they sit in the system

```
 Vision ─vigil/vision─► ML1 ─vigil/decision─► ML2 ─vigil/alert (0..3, retained)──────────► ESP32 LED / buzzer
                         │                     │
                         └─vigil/ecu (5 Hz)───►│ CarLink ─vigil/car/cmd (only changes)──► ECU_Arduino (simulated car)
                                               │                                               │
 ML1, dashboard ◄──────────── vigil/car/state (0.5–5 s), vigil/car/event ◄─────────────────────┘
                              vigil/esp32/status: "online"/"offline" (retained, last will)
```

- ML2 publishes the **alert level** right away (it doesn't wait in the action queue).
- ML1 computes a continuous **fuzzy car command**; ML2 turns it into **discrete commands** for the board; the board **simulates** the car and reports back; ML1 puts the real car state into its status and into the 112 report; the dashboard draws "what ML1 asks vs what the ECU does".

---

## 4. Sketch 1 — `vigil_alert.ino` (alert LED / buzzer)

About 70 lines; everything happens in `loop()`.

**Behaviour:**

| Level on `vigil/alert` | LED (GPIO 8) | Buzzer (GPIO 4) |
|---|---|---|
| 0 | off | off |
| 1 | on continuously | off |
| 2 | blinks (toggles every 250 ms) | off |
| 3 | blinks fast (toggles every 60 ms) | beeps (on/off every 250 ms) |

**How it works:**
- `setup()`: serial at 115200; LED and buzzer pins as outputs; **scans Wi-Fi networks** and prints whether it sees yours, on which channel and with what signal (a network on 5 GHz never appears); sets the broker and the message callback.
- `onMessage()`: takes the first character `'0'..'3'` as the level and prints `alert N` on serial — this line is the **confirmation that `esp32_test.py` waits for**.
- `ensureConnected()`: blocking reconnect loop for Wi-Fi (with diagnostics: status 1 = network not found, 4 = wrong password) and for MQTT (prints `mqtt: ok` or `mqtt: eroare <code>`, retries every 2 s; if Wi-Fi drops it goes back to reconnecting Wi-Fi).
- `loop()`: `ensureConnected()` → `mqtt.loop()` → compute the LED/buzzer state from `millis()`.
- **Non-blocking blinking**: blink phases are derived from `millis() / period % 2`, so there is no `delay()` in the normal loop.
- **TX power reduced to 8.5 dBm** (`WiFi.setTxPower(WIFI_POWER_8_5dBm)`): many ESP32-C3 SuperMini boards fail to connect at full power (a known board/antenna issue).
- The broker keeps `vigil/alert` **retained**, so after a reboot the board immediately gets the current level; ML2 sends `0` at startup so it never stays on an old level.

Configuration: `BROKER_IP` (single place to change), `LED_PIN`, `LED_ON`, `BUZZER_PIN`; Wi-Fi in `wifi_secrets.h` (`WIFI_SSID`, `WIFI_PASS`). MQTT client id: fixed `vigil-esp32`.

---

## 5. Sketch 2 — `ECU_Arduino.ino` (simulated car actuator)

This sketch makes the ESP32 behave like the **brake + hazard-light controller of a car**, so the safety chain "driver collapses → car slows down and stops on the shoulder" can be shown physically.

### 5.1 Topics

| Topic | Direction | Content |
|---|---|---|
| `vigil/car/cmd` | IN | JSON commands (below); any command may carry an `"id"` that is echoed back |
| `vigil/alert` | IN | `"0".."3"` — alert level from ML2, shown on the LED when hazards are off (+ buzzer if wired) |
| `vigil/car/state` | OUT | `{"speed_kmh":72.4,"target_kmh":0,"braking":true,"hazard":true,"stopped":false,"emergency":true,"alert":3}` |
| `vigil/car/event` | OUT | `{"event":"emergency_braking_started","speed_kmh":100,"target_kmh":0,"decel_mps2":2,"hazard":true,"emergency":true,"id":"e1"}` |
| `vigil/esp32/status` | OUT | `"online"` / `"offline"` — **retained**, `"offline"` set as the MQTT **last will** |

### 5.2 Commands

| Command | Effect |
|---|---|
| `{"cmd":"emergency","decel_mps2":2.5,"id":"…"}` | **emergency protocol**: hazards ON first, then gentle braking to a full stop; alert raised to ≥ 3 |
| `{"cmd":"cancel_emergency"}` | driver responded: stop braking, keep current speed, **hazards stay on** |
| `{"cmd":"slow_down","target_kmh":60,"decel_mps2":2.0}` | brake down to a target speed (refused during an emergency) |
| `{"cmd":"hazard","on":true\|false}` | hazards on/off (turning them off is refused during an emergency) |
| `{"cmd":"resume"}` | stop braking, keep current speed (refused during an emergency) |
| `{"cmd":"reset"}` | demo reset: cruise speed 100 km/h, hazards off |

Events: `emergency_braking_started, emergency_cancelled, braking_started, target_reached, stopped, hazard_on, hazard_off, resumed, reset, error` (with `detail`: `invalid_json`, `unknown_cmd`, `emergency_active`, `invalid_alert`).

### 5.3 The simulated car

```c
struct Car { speed = 100, target = 100, decel = 2.0, braking, hazard, stopped, emergency };
```
- `updateCar()` runs every loop: while braking, `speed -= decel · 3.6 · dt` (m/s² converted to km/h per second, `dt` from `millis()`), until it reaches the target → event `target_reached` or `stopped`.
- **Safety limits on deceleration**: any requested `decel_mps2` is clamped to **0.5 … 4.0 m/s²** — a command can never cause harsh braking. Default 2.0 m/s² (~14 s from 100 to 0); ML2 uses 2.5 for MRM, 2.0 urgent, 0.5 normal.
- `slow_down` targets are clamped to `[0, current speed]` (it can only slow down; there is no "accelerate" — `reset` returns to cruise).
- **Emergency is sticky**: while `emergency` is true, `slow_down`, `resume` and `hazard off` are refused with an `error` event — only `cancel_emergency` or `reset` end it.
- **State publishing**: every **500 ms while braking**, every **5 s** otherwise (heartbeat), plus after every command.
- The command `id` is kept while braking, so the final `stopped` event still carries the id of the command that started it (end-to-end traceability).

### 5.4 Outputs

- **Hazard lights** blink every 400 ms on the onboard LED when `hazard` is on.
- When hazards are off, the **same LED shows the alert level** (1 on, 2 blink 250 ms, 3 fast blink 60 ms) — needed because no buzzer is wired by default.
- Optional **brake light** on while braking; optional **buzzer**: level 2 short beeps, level 3 fast beeps.

### 5.5 MOCK mode (no Wi-Fi needed)

`#define MOCK_MODE 1` turns off Wi-Fi/MQTT: you type commands in the Serial Monitor (line ending: Newline): `emergency | cancel | brake | hazard on | hazard off | resume | reset | 0..3 | {json}`. Everything the board would publish is printed as `[PUB] topic payload`. With `AUTO_DEMO 1` the board triggers the emergency protocol by itself 5 s after boot. The car simulation runs in both modes.

### 5.6 Networking details

- Non-blocking reconnect in `realLoop()`: Wi-Fi retried every 5 s, MQTT every 3 s; `updateCar()` and `updateOutputs()` keep running **while reconnecting** (the simulated car never freezes).
- Wi-Fi: `WiFi.disconnect(true)` before `begin` (avoids "sta is connecting"), auto-reconnect disabled (handled manually), 15 s connect timeout, TX power 8.5 dBm, network scan printed on failure.
- MQTT: unique client id `vigil-esp32-<chip MAC>`, keep-alive 15 s, buffer 512 bytes (enough for the JSON), subscriptions with QoS 1, last will `offline`.

---

## 6. The C fuzzy lookup table (`ML1/ecu/`)

ML1's fuzzy controller (Mamdani, 8 rules, centroid defuzzification — see `ML1_EXPLAINED.md` §10) is compiled into a **lookup table** for embedded use:

- `python ML1/run_ml1.py ecu --export-c` generates `ML1/ecu/vigil_fuzzy_lut.h` — **do not edit it by hand**.
- Grid of **15 × 15 × 9** points over the inputs (attention `a`, risk, confidence), each holding 3 outputs as **uint8** → **6,075 bytes** in flash.
- `vigil_fuzzy_eval(a, risk, conf, out[3])` does **trilinear interpolation** between the 8 surrounding grid points: no dynamic allocation, no recursion, bounded execution time, no floating-point library beyond basic arithmetic.
- Outputs: `speed_factor` [0, 1], `authority` (lane assist) [0, 1], `headway_s` [1.5, 3.5].
- Grid nodes include the corners of the membership functions (where the output has "knees"). Error vs exact inference on 4,000 random points: mean < 0.7 %, max 0.07–0.14 at isolated points.
- `test_lut.c` reads `a risk conf` lines from stdin and prints the 3 outputs; `ML1/tests/test_probability_ecu.py` compiles it with `gcc -std=c99 -Wall -Wextra -Werror` (if gcc exists) and compares against Python.
- The deterministic **supervisor** (severity caps, MRM, rate limits) is intentionally **not** in the header — it belongs to the ECU's own safety layer.

This header is not used by the two sketches yet; it is ready to be dropped into an ECU firmware.

---

## 7. Configuration: Wi-Fi, broker IP, pins

1. **Wi-Fi secrets** (never committed — the repo is public):
   - `ML2/esp32/vigil_alert/`: copy `wifi_secrets.h.example` → `wifi_secrets.h`, defines `WIFI_SSID`, `WIFI_PASS` (`#define`s);
   - `ECU_Arduino/`: copy `wifi_secrets.h.example` → `wifi_secrets.h`, defines `WIFI_SSID`, `WIFI_PASSWORD` (`const char*`).
2. **Hotspot on 2.4 GHz**, WPA2-Personal. On a phone hotspot, set the band to 2.4 GHz, then turn the hotspot off and on.
3. **Broker IP** = the laptop running Mosquitto (`ipconfig`, Wi-Fi adapter):
   - `vigil_alert.ino`: `BROKER_IP`;
   - `ECU_Arduino.ino`: `MQTT_HOST` (+ `MQTT_USER`/`MQTT_PASS` if the broker ever needs auth).
4. **Broker** must listen on the network: `listener 1883 0.0.0.0` + `allow_anonymous true` (`ML2/mosquitto.conf`). Only one broker instance must be listening.
5. **Pins**: `LED_PIN`/`LED_ON`/`BUZZER_PIN` in `vigil_alert`; `HAZARD_PIN`/`HAZARD_ACTIVE_LOW`/`BRAKE_PIN`/`BUZZER_PIN` in `ECU_Arduino` (−1 = not connected).

---

## 8. Building and flashing

```powershell
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install PubSubClient
arduino-cli lib install ArduinoJson          # only for ECU_Arduino

arduino-cli board list                        # find the port, e.g. COM13

# alert board
arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc -u -p COM13 ML2\esp32\vigil_alert
# car actuator
arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc -u -p COM13 ECU_Arduino
```
In the Arduino IDE: board "ESP32C3 Dev Module", Tools → USB CDC On Boot: Enabled. Close the Serial Monitor before uploading or running Python tests on the port.

---

## 9. Testing

| Test | How |
|---|---|
| **Alert board end-to-end** | `python ML2/esp32_test.py COM13`: resets the board over serial, waits for `mqtt: ok` (≤ 30 s), clears the retained level, sends 1, 2, 3, 0 and checks each `alert N` confirmation with its latency. Expected: LED on → blinking → fast blinking → off |
| **ECU without network** | flash with `MOCK_MODE 1`, open the Serial Monitor at 115200, type `emergency`, `cancel`, `brake`, `reset`, `2`… and watch the `[PUB]` / `[EVENT]` lines |
| **ECU through MQTT** | `mosquitto_pub -h <broker> -t vigil/car/cmd -m '{"cmd":"emergency","id":"t1"}'` and `mosquitto_sub -t 'vigil/car/#' -v` |
| **Full chain** | run ML1 (`bridge`), ML2 (`executor.py`) and the board; trigger a collapse scenario → `vigil/ecu` MRM → `emergency` → hazards blink, speed falls to 0 → `stopped` event with the `ml2-N` id |
| **C fuzzy table** | `gcc -std=c99 -Wall -Wextra -O2 -o test_lut ML1/ecu/test_lut.c` or `python ML1/run_ml1.py test` |

---

## 10. Embedded concepts used

| Concept | Where |
|---|---|
| **MQTT publish/subscribe** with **retained** messages and **last will** | `vigil/alert` retained; `vigil/esp32/status` last will "offline" |
| **Non-blocking loop** (`millis()` instead of `delay()`) | blinking, car simulation, state timers, reconnect timers |
| **Automatic reconnection** | Wi-Fi and broker, with diagnostics on serial |
| **Command / event protocol with IDs** | `vigil/car/cmd` → `vigil/car/event` echoes `id` |
| **Physical simulation with dt** | speed integration using real elapsed time |
| **Input clamping / safety envelope** | decel 0.5–4 m/s², target ≤ current speed, emergency lock |
| **Active-low outputs** | onboard LED lights on LOW (`setOutput(pin, on, activeLow)`) |
| **Mock / hardware-in-the-loop switch** | `MOCK_MODE` compile-time flag |
| **Lookup table + trilinear interpolation** | fuzzy controller in 6 KB of flash, bounded time |
| **Secrets out of source control** | `wifi_secrets.h` in `.gitignore` |

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `NU văd <network>` / scan doesn't list it | network on 5 GHz → switch the hotspot to 2.4 GHz and restart it |
| Wi-Fi status 1 | network not found (name wrong, 5 GHz, too far) |
| Wi-Fi status 4 or 6 | usually wrong password |
| `mqtt: eroare -2` / `state=-2` | broker unreachable: wrong IP (laptop IP changed?), broker stopped, or broker listening only on localhost |
| `state=5` | not authorized → `allow_anonymous true` or set user/password |
| `mqtt: ok` but nothing arrives | two brokers running — `netstat -ano | findstr :1883` must show exactly one `0.0.0.0:1883 LISTENING` |
| `could not open port 'COM13': Access is denied` | port busy (Arduino Serial Monitor) |
| board doesn't answer after reset | ESP32-C3 may stay in the bootloader — unplug and replug USB |
| LED/buzzer on at boot | old level retained in the broker → ML2 sends `0` at start; manually `mosquitto_pub -t vigil/alert -m 0 -r` |
| C3 never connects at all | TX power fix (8.5 dBm) is already applied; try closer to the hotspot |

---

## 12. Known limitations and things to check

- **Broker IPs differ between files**: `vigil_alert.ino` uses `172.22.118.134`, `ECU_Arduino.ino` uses `172.22.118.20` (same as `Vision/config.yaml`). Set both to the IP of the laptop that actually runs Mosquitto before flashing.
- **Different secret names**: `vigil_alert` expects `WIFI_PASS`, `ECU_Arduino` expects `WIFI_PASSWORD` — each sketch has its own `wifi_secrets.h.example`, don't swap them.
- `vigil_alert` uses a **fixed client id** (`vigil-esp32`): two such boards on one broker would disconnect each other. `ECU_Arduino` derives a unique id from the chip MAC.
- `vigil_alert`'s reconnect loop is **blocking**; `ECU_Arduino`'s is non-blocking.
- The car is **simulated**: no real vehicle bus (CAN), no `highway-env` closed loop yet; there is no "accelerate" command (only `reset` back to cruise).
- The fuzzy C header is **generated and tested but not yet used** by the sketches.
- No buzzer is wired in the default `ECU_Arduino` configuration, so alert levels are only visible on the LED.
