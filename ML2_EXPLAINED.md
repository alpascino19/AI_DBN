# Vigil — The ML2 Module (Action Executor & Voice), Explained

This document explains everything inside the `ML2/` folder of AI_DBN (Vigil): how decisions from ML1 are turned into real actions, how the priority queue and interruptions work, how the voice (TTS), the microphone (STT) and the conversation with the driver work, the assistant Virgil, the simulated 112 call, the link to the ECU board, and how to run and test it.

> Related docs: `ML2/README.md` (very detailed, Romanian). Companion files in this folder: `VISION_EXPLAINED.md`, `ML1_EXPLAINED.md`, `ARDUINO_EXPLAINED.md`.

---

## Table of contents

1. [What ML2 does](#1-what-ml2-does)
2. [Technology stack](#2-technology-stack)
3. [Folder structure](#3-folder-structure)
4. [Architecture](#4-architecture)
5. [The executor (`executor.py`)](#5-the-executor-executorpy)
6. [The tools (`actions.py`)](#6-the-tools-actionspy)
7. [Speech: TTS and STT](#7-speech-tts-and-stt)
8. [Conversation with the driver (`ask_driver`)](#8-conversation-with-the-driver-ask_driver)
9. [The assistant Virgil](#9-the-assistant-virgil)
10. [The simulated 112 call and the contact notification](#10-the-simulated-112-call-and-the-contact-notification)
11. [Link to the car ECU (`CarLink`)](#11-link-to-the-car-ecu-carlink)
12. [MQTT contracts](#12-mqtt-contracts)
13. [Running ML2](#13-running-ml2)
14. [Tuning](#14-tuning)
15. [Testing](#15-testing)
16. [Key design principles](#16-key-design-principles)
17. [Glossary](#17-glossary)
18. [Known limitations and troubleshooting](#18-known-limitations-and-troubleshooting)

---

## 1. What ML2 does

ML2 is the **hands and voice** of Vigil. ML1 publishes **what** should be done (`vigil/decision`); ML2 **does it**:

- speaks to the driver in Romanian;
- plays generated alarm sounds (3 levels);
- lights the LED / buzzer on the ESP32 (`vigil/alert`);
- holds a short spoken conversation and reports each answer back to ML1 (`vigil/driver_response`);
- simulates the 112 emergency call (report on screen + saved to disk);
- translates ML1's continuous ECU command into discrete commands for the simulated car (`vigil/car/cmd`);
- runs the voice assistant **Virgil** (time, weather, traffic, general questions).

```
CV ──vigil/vision──► ML1 (decides) ──vigil/decision──► ML2 (executes) ──vigil/alert──► ESP32 (LED / buzzer)
                         ▲                                  │      └──vigil/car/cmd──► ECU_Arduino (simulated car)
                         └───────vigil/driver_response──────┘
```

Everything runs on the laptop; exceptions are the Edge TTS voice (sends only the spoken text to Microsoft), the optional internet lookups of the assistant, and the optional contact notification.

---

## 2. Technology stack

| Library / program | Used for |
|---|---|
| **Python** (tested on 3.14, Windows 11), stdlib `threading`, `queue`, `heapq` | executor, priority queue, interruptions |
| **paho-mqtt 2.x** | receive decisions, publish alerts / responses / car commands |
| **edge-tts** (`ro-RO-EmilNeural`) + **ffmpeg** | main neural voice (needs internet) |
| **piper-tts** (`ro_RO-mihai-medium`) | local offline fallback voice |
| **pyttsx3** | last-resort voice (Windows SAPI) |
| **faster-whisper** (`small`) + **Silero VAD** | speech-to-text in Romanian + voice activity detection |
| **sounddevice**, **NumPy** | microphone/speaker, alarm synthesis, volume measurement |
| **Ollama** (`qwen2.5:1.5b` default, `gemma3:4b` better Romanian) | generating the conversation replies |
| **pyserial** | `esp32_test.py` reads the board's confirmations |
| **Mosquitto** | MQTT broker (TCP 1883 + WebSocket 9001 for the dashboard) |
| **arduino-cli** | compiling and flashing the ESP32 |
| Optional online services | Open-Meteo (weather), RSS (news), Waze via OpenWebNinja, Google Maps Routes, Tavily search via OmniRoute + Groq models, TomTom, CallMeBot (Telegram), `adb` (Android call) |

---

## 3. Folder structure

| File | Role |
|---|---|
| `executor.py` | **main program**: listens to decisions, validates, queues and runs actions; `CarLink` for the ECU |
| `actions.py` | the tools: voice, alarm, conversation, 112 (simulated), assistant |
| `info.py` | offline/simple lookups for the dashboard assistant buttons (weather, traffic estimate, time) |
| `info_assistant.py` | standalone "Hei Virgil" information assistant (weather, news, Waze traffic, routes, web search) |
| `emergency_notify.py` | optional separate process: notifies a contact person on `call_112` (never 112) |
| `fake_llm.py` | publishes fake decisions to test ML2 without ML1 |
| `voice_test.py` | the voice assistant Virgil at the microphone, without ML1 |
| `esp32_test.py` | end-to-end test: laptop → broker → ESP32 → serial confirmation |
| `test_*.py` | automated tests (executor, voice, info, assistant, notify) |
| `esp32/vigil_alert/` | the simple ESP32 LED/buzzer sketch (see `ARDUINO_EXPLAINED.md`) |
| `mosquitto.conf` | broker config: listen on the network (1883) + WebSocket (9001) |
| `.env.example` | template for the contact-notification settings (`.env` is never committed) |

Files created at runtime and never committed: `wifi_secrets.h`, `recordings/*.wav`, `voice_live.json`, `reports/112_*.txt`, `voices/`.

---

## 4. Architecture

```
vigil/decision ─► validate decision ─► for each action:
                                         ├─ validate action (known tool? correct args?) ── no ──► dropped + logged
                                         ├─ set_esp32 ──► published IMMEDIATELY on vigil/alert (bypasses the queue)
                                         ├─ debounce: same tool at same severity in last 10 s? ──► dropped
                                         └─ PriorityQueue (by severity, then arrival order)
                                                         │
                                          ONE worker thread runs one action at a time
                                          (threading.Event interrupt for emergencies)

vigil/assistant ─► assistant request (rank 2.5) ──► same queue
vigil/ecu ───────► CarLink (only changes) ──────► vigil/car/cmd
```

Why one worker: there is one speaker and one microphone — voice and alarm must never overlap.

---

## 5. The executor (`executor.py`)

**At startup:**
1. connects to the broker (`localhost` or `VIGIL_BROKER`);
2. subscribes to `vigil/decision`, `vigil/assistant`, `vigil/ecu`;
3. publishes `0` on `vigil/alert` (retained) so the ESP32 doesn't stay on an old level, and `reset` to the car;
4. preloads the voice models (Piper, Whisper, the LLM) in the background — otherwise the first question would be delayed 15–50 s;
5. enters the worker loop.

**Validation of a decision:** valid JSON, `severity` an int 0–3, `actions` a list; anything else is rejected and logged without crashing. `severity: 0` = "back to normal" → ESP32 off.

**Validation of an action** (protection against an LLM inventing tools or arguments):
- only the 6 known tools: `speak`, `play_alarm`, `ask_driver`, `set_esp32`, `call_112`, `suggest_break`;
- exactly the required arguments with the right type (`True` is not accepted as a number);
- texts cannot be empty; alarm level 1–3, ESP32 level 0–3.

**Debounce:** the same tool at the same severity runs at most once every **10 s** (`DEBOUNCE_SEC`). The key includes the severity, so **escalation passes** (alarm 3 is not blocked by a recent alarm 2).

**Priority queue:** `queue.PriorityQueue` ordered by `(−severity, sequence)` → severity 3 first; FIFO within the same severity.

**Emergencies (severity 3):**
1. **interrupt** the action in progress if it is less important (speech or alarm stop in ~50 ms, even while the voice is still being generated) via a shared `threading.Event`;
2. **drop less important actions** still waiting in the queue (after a 112 call, "Are you tired?" makes no sense);
3. never interrupt another emergency.
A race is handled too: if an emergency arrives between `get()` and starting a weaker action, the weaker action is skipped.

**Assistant priority:** assistant requests have rank **2.5** — they pass ahead of queued level 1–2 alerts and can cut a running reminder/alarm 1–2; severity 3 always interrupts Virgil; only one assistant request waits at a time (the latest click wins); 2 s debounce against double clicks.

---

## 6. The tools (`actions.py`)

`executor.py` calls `actions.run(tool, args, interrupt, publish)`.

| Tool | What it does |
|---|---|
| `speak(text)` | generates the voice (Edge → Piper → pyttsx3) on a separate thread and plays it; interruptible |
| `play_alarm(level)` | alarm synthesized in code with NumPy: 1 = one short quiet beep, 2 = three beeps, 3 = two-tone siren, 4 s, loud |
| `ask_driver(question)` | short conversation (§8) |
| `set_esp32(level)` | not in actions: the executor publishes it directly on `vigil/alert` |
| `call_112(report)` | **simulated only** (§10) |
| `suggest_break()` | says one of 3 break phrases, chosen at random so it doesn't sound repetitive |

---

## 7. Speech: TTS and STT

### 7.1 Text-to-speech (fallback chain)

1. **edge-tts**, voice `ro-RO-EmilNeural` — most natural, needs internet; decoded with ffmpeg.
2. If Edge doesn't answer within **5 s**, switch to **Piper** `ro_RO-mihai-medium` (local ONNX model) for 60 s, then try Edge again.
3. If Piper fails too: **pyttsx3** (Windows voice).

Synthesis runs on its own thread so an emergency can abandon it immediately.

### 7.2 Speech-to-text (`listen` → `record` → `transcribe`)

- **`record`**: reads the microphone in 250 ms chunks (16 kHz). After each chunk, `utterance_state` uses **Silero VAD** (voice activity detection) to decide:
  - `wait` — keep listening;
  - `silence` — the driver didn't **start** speaking within **4 s** (`LISTEN_SEC`);
  - `done` — the sentence ended (~1 s of silence, `END_SILENCE_SEC`), max 15 s.
- **`transcribe`**: **faster-whisper `small`** on CPU, `language="ro"`, beam size 5, a context prompt, strict VAD. Then each text segment is **filtered**:
  - `nesigur` (unsure): Whisper invents text on noise — a false "answered" could hide an unconscious driver, so low-confidence segments are dropped; known YouTube-style hallucinations ("abonați-vă", "subtitrare") are removed by a regex;
  - `departe` (far): volume below `NEAR_DBFS` → a background voice, not the driver's.
- **Clipping warning**: if peaks hit the maximum, the microphone gain is too high.

---

## 8. Conversation with the driver (`ask_driver`)

1. Speak the question received from ML1.
2. **Start the microphone only after the voice has finished** (so it doesn't hear itself).
3. Listen: 4 s to start answering; once started, until the sentence ends (max 15 s).
4. Transcribe with Whisper.
5. **Publish immediately** on `vigil/driver_response`: `{"answered": true/false, "latency_sec": ..., "text": ...}` — ML1 learns at once if the driver says "mă doare pieptul".
6. If the driver is **silent**: stop; ML1 decides the escalation.
7. If he answered: the local LLM (Ollama, `VIGIL_MODEL`) generates the next line in the **Virgil persona** (calm, polite, like JARVIS), with a JSON schema `{"reply", "end"}`, e.g. "De când conduceți fără pauză?". Repeat, with **at most 3 driver answers** (`MAX_TURNS`).
8. Stop when the LLM sets `end=true` (driver is fine, or must stop) or the reply contains no question.

Notes for ML1: several `driver_response` messages may arrive for one question; open questions work better than yes/no (a short "da" is badly recognized).

---

## 9. The assistant Virgil

There are three faces of the assistant:

| Where | What |
|---|---|
| `voice_test.py` | the full wake-word assistant: **idle** listens continuously and reacts only to **"Hei Virgil"** (fuzzy match: Whisper writes "Vigil", "Virgilă"…); in conversation, 4 s of silence to a question → **alarm + ESP32 level 2**, keeps listening; "sunt bine", "mulțumesc", "gata" → back to idle, ESP32 off. Writes `voice_live.json` (live state, volume `speech_dbfs`, rejections) and `recordings/*.wav` |
| `info_assistant.py` | standalone information assistant: weather (Open-Meteo), news (RSS HotNews/Digi24), traffic (Waze alerts within 10 km via OpenWebNinja + Tavily search), travel time (Google Maps Routes API with live traffic, start from Windows location service), anything else via Tavily search through OmniRoute; answers phrased by Groq models via OmniRoute (~2–3 s, model fallback chain). After each answer a beep and it listens for another question. It **stops talking and stays silent 60 s on any alert** (severity ≥ 1) |
| dashboard buttons (`vigil/assistant`) | "Întreabă-l pe Virgil", "Ora", "Vremea", "Traficul" handled by the **executor**: time from the laptop clock; with `OMNIROUTE_API_KEY` the same answers as `info_assistant`; otherwise `info.py` — Open-Meteo weather, TomTom traffic if `TOMTOM_KEY`, else an hour-based estimate *said as an estimate*, free questions to the local LLM with real data only. Lookups run on a separate thread (an emergency never waits for slow HTTP), 20 s timeout. Status published on `vigil/assistant/reply` (`listening / working / speaking / done / interrupted`). A dashboard request never publishes on `vigil/driver_response` |

Location is fixed to Cluj-Napoca by default (`VIGIL_LAT`, `VIGIL_LON`, `VIGIL_PLACE`). API keys are environment variables, never in code.

---

## 10. The simulated 112 call and the contact notification

**`call_112` in the executor is only simulated.** It never calls or sends anything outside the laptop:
- prints the report with "APEL 112 SIMULAT" and "AVARII PORNITE (simulat)";
- saves it to `reports/112_DATE_TIME.txt`;
- announces by voice that the hazards were turned on and the emergency alert was sent (simulated).
The dashboard shows the report from `vigil/decision`.

**`emergency_notify.py`** (optional, separate process) listens to `vigil/decision` and, **only on `call_112`**, notifies the contact person from `ML2/.env` (a teammate, **never 112**):
- Telegram message with the report via **CallMeBot** (the recipient sends `/start` once to the bot);
- optionally a GSM call from an Android phone over USB (`adb`).
One notification per emergency (5 min lock), up to 3 retries at 70 s if CallMeBot answers `FLOOD`. The forwarded message is trimmed and leaves out the observed medical signs and the location (tested in `test_emergency_notify.py`). If internet or phone fail, the executor is not affected. Modes: `--test`, `--dry-run`.

---

## 11. Link to the car ECU (`CarLink`)

ML1 publishes a **continuous** fuzzy command on `vigil/ecu` (5 Hz). The ECU board understands **discrete** commands on `vigil/car/cmd`. `CarLink` sends **only changes**:

| ML1 `vigil/ecu` | → `vigil/car/cmd` |
|---|---|
| `mode` becomes `MRM` | `{"cmd":"emergency","decel_mps2":2.5}` (hazards + controlled stop) |
| leaving `MRM` | `cancel_emergency` |
| `hazards` changes | `{"cmd":"hazard","on":...}` |
| `target_kmh` drops ≥ 5 km/h below the last target / cruise (100 km/h) | `{"cmd":"slow_down","target_kmh":...,"decel_mps2":0.5 or 2.0}` (2.0 when hazards are on) |
| back to `NORMAL` without hazards after slowing | `reset` (back to cruise) |

Every command carries an `id` (`ml2-N`) that the board echoes in `vigil/car/event`, so the chain can be traced. Decelerations match ML1's constants (MRM 2.5, urgent 2.0, normal 0.5 m/s²). Invalid `vigil/ecu` messages are rejected and logged.

---

## 12. MQTT contracts

| Topic | Direction | Content |
|---|---|---|
| `vigil/decision` | ML1 → ML2 | `{"a":0.2,"severity":3,"actions":[{"tool":"play_alarm","args":{"level":3}}],"reason":"..."}` |
| `vigil/driver_response` | ML2 → ML1 | `{"answered":false,"latency_sec":null,"text":null}` |
| `vigil/alert` | ML2 → ESP32 | `0`–`3`, retained |
| `vigil/assistant` | dashboard → ML2 | `{"request": "listen" \| "time" \| "weather" \| "traffic"}` |
| `vigil/assistant/reply` | ML2 → dashboard, ML1 | `{"request", "status", "heard", "text", "source"}` |
| `vigil/ecu` | ML1 → ML2 | continuous fuzzy command |
| `vigil/car/cmd` | ML2 → ECU board | discrete commands (only changes) |
| `vigil/car/state`, `vigil/car/event` | ECU board → ML1, dashboard | real (simulated) speed, braking, hazards; events with command `id` |

Tools and arguments: `speak {"text"}`, `play_alarm {"level":1-3}`, `ask_driver {"question"}`, `set_esp32 {"level":0-3}` (immediate), `call_112 {"report"}` (simulated), `suggest_break {}`.

The broker (`mosquitto.conf`): `listener 1883 0.0.0.0` + `allow_anonymous true` (Mosquitto 2.x only listens on localhost by default, so the ESP32 couldn't connect) and `listener 9001` with `protocol websockets` for the React dashboard. Make sure only **one** broker is listening (the Windows service), otherwise modules talk to different brokers.

---

## 13. Running ML2

From `ML2/` (with the root `.venv`):

| Command | What it does |
|---|---|
| `python executor.py` | the executor — wait for "modelele de voce sunt încărcate" |
| `python fake_llm.py distragere oboseala medical fara_raspuns reset -d 15` | demo without ML1 (scenarios: `distragere` 1, `oboseala` 2, `medical` 3, `fara_raspuns` 3 + 112, `invalid`, `reset`) |
| `python fake_llm.py oboseala medical -d 2` | shows that the emergency cuts the fatigue message |
| `python voice_test.py` | Virgil at the microphone (wake word) |
| `python info_assistant.py` | information assistant (needs `OMNIROUTE_API_KEY`) |
| `python esp32_test.py COM13` | board test: resets it, sends levels 1,2,3,0 and checks the serial confirmations with timings |
| `python emergency_notify.py [--test \| --dry-run]` | optional contact notification |

Install: `pip install -r ML2/requirements.txt`; external: Mosquitto, Ollama (`qwen2.5:1.5b`, optionally `gemma3:4b`), ffmpeg, arduino-cli; Piper voice with `python -m piper.download_voices ro_RO-mihai-medium --download-dir ML2\voices`; Whisper `small` downloads by itself.

---

## 14. Tuning

At the top of `actions.py`:

| Constant | Value | Meaning |
|---|---|---|
| `LISTEN_SEC` | 4 | seconds for the driver to **start** answering |
| `END_SILENCE_SEC` | 1.0 | silence that ends a sentence |
| `MAX_UTTERANCE_SEC` | 15 | max sentence length |
| `NEAR_DBFS` | −28 | volume below which a voice counts as "background" — **calibrate on the demo mic** (~10 dB under `speech_dbfs`) |
| `WHISPER_MODEL` | `small` (`VIGIL_WHISPER`) | `large-v3-turbo` is better but ~3× slower |
| `PROMPT` | context + "Hei Virgil" | guides Whisper — no example answers (Whisper copies them on short audio) |
| `EDGE_VOICE` | `ro-RO-EmilNeural` | or `ro-RO-AlinaNeural` |
| `LLM_MODEL` | `VIGIL_MODEL`, default `qwen2.5:1.5b` | ~2.9 s/reply, weak free Romanian; `gemma3:4b` ~4.4 s, correct Romanian |
| `MAX_TURNS` | 3 | driver answers per conversation |
| `PERSONA` | — | Virgil's personality and rules (no Romanian „…" quotes — the model copies them and breaks the JSON) |

In `executor.py`: `DEBOUNCE_SEC = 10`, `VIGIL_BROKER`.

---

## 15. Testing

| Test | Covers |
|---|---|
| `test_executor.py` | validation, bad payloads ignored, priority and direct ESP32, debounce with escalation, severity-3 interruption (also during slow synthesis), dropping stale actions, conversation (silence, continuation, limit, LLM down, reply without question), wake word, idle → alarm → idle, assistant priority, `CarLink` |
| `test_voice.py` | close voice accepted, far voice and silence rejected, end-of-sentence detection (uses Piper-synthesized speech, no mic) |
| `test_info.py`, `test_info_assistant.py` | lookups and routing of questions, offline |
| `test_emergency_notify.py` | one notification per emergency, retries, dry-run |

All run without hardware, microphone or internet. "decizie respinsă" lines during tests are expected (tests send bad data on purpose). Pre-demo checklist: hotspot on 2.4 GHz, ESP32 on USB, Ollama running, Arduino Serial Monitor closed.

---

## 16. Key design principles

1. **Never trust the LLM's output blindly** — strict validation of every decision and action.
2. **One speaker, one worker** — actions never overlap.
3. **Safety first in the queue** — severity 3 interrupts, flushes weaker actions, and can't be blocked by debounce.
4. **The ESP32 is immediate** — the visual alert doesn't wait behind speech.
5. **Report answers as they happen** — each answer or silence goes to ML1 at once.
6. **Fail soft** — Edge → Piper → pyttsx3; online lookups → offline sources; broker down → assistant continues without ESP32.
7. **A false "answered" is dangerous** — aggressive filtering of Whisper hallucinations and background voices.
8. **112 is never really called** — simulation only; the optional notification goes to a teammate.
9. **Send only changes to the car** — the board gets discrete, traceable commands, not a 5 Hz stream.
10. **Secrets stay out of git** — `.env`, `wifi_secrets.h`, API keys as environment variables.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **TTS / STT** | text-to-speech / speech-to-text |
| **VAD** | Voice Activity Detection — decides when someone is speaking |
| **dBFS** | decibels relative to full scale (0 = maximum); used to tell near from far voices |
| **Whisper hallucination** | text Whisper outputs on silence/noise that nobody said |
| **Debounce** | ignore repeats of the same action within a time window |
| **Priority queue** | a queue where the most important item is served first |
| **Interrupt event** | a shared flag (`threading.Event`) that tells a running action to stop |
| **Retained message** | MQTT message the broker keeps and delivers to new subscribers |
| **Wake word** | the phrase that activates the assistant ("Hei Virgil") |
| **MRM** | Minimal Risk Maneuver — controlled stop with hazards |

---

## 18. Known limitations and troubleshooting

**Not done yet:** the closed loop with a real vehicle or `highway-env` (the ECU board only simulates a car); "Hei Virgil" inside the executor (wake-word mode exists only in `voice_test.py`; in the executor a conversation starts only when ML1 sends `ask_driver`); speaker identification (Virgil listens to the closest voice); calibrating `NEAR_DBFS` on the demo microphone.

| Symptom | Cause / fix |
|---|---|
| poor recognition | laptop mic picks up nearby people almost as loud as the driver — use a headset; lower mic gain to 60–70 % on clipping warnings |
| long pauses in conversation | LLM reply takes ~2.9 s (qwen) / ~4.4 s (gemma); ML1 and ML2 slow each other if they call Ollama together |
| nonsense replies | `qwen2.5:1.5b` is too small for free Romanian — switch ML2 to `gemma3:4b` |
| robotic voice | no/slow internet → automatic switch to Piper |
| first question late | first model load 15–50 s — wait for the "loaded" message |
| ESP32 "NU văd <network>" | network on 5 GHz — set the hotspot to 2.4 GHz and restart it |
| ESP32 `mqtt: eroare -2` | broker unreachable: wrong `BROKER_IP` or broker stopped |
| ESP32 `mqtt: ok` but receives nothing | two brokers running — `netstat -ano | findstr :1883` must show one LISTENING line |
| `COM13 Access is denied` | port busy (Arduino Serial Monitor) |
| LED/buzzer on at boot | last level retained in the broker — executor sends `0` at start; manually `mosquitto_pub -t vigil/alert -m 0 -r` |
