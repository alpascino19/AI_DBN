# Vigil — The ML1 Module (Decision Engine), Explained

This document explains everything inside the `ML1/` folder of AI_DBN (Vigil): what the decision engine does, how a camera frame becomes a decision, which algorithms and models are used, how the LLM is kept safe, the probability matrix and the fuzzy ECU controller, how it is evaluated, and how to run it.

> Related docs: `ML1/README.md` (commands, Romanian), `ML1/docs/TECHNICAL_DEEP_DIVE.md` (English, very detailed, with code line links), `ML1/docs/ARCHITECTURE.md`, `ML1/docs/PROBABILITY_ECU.md`, `ML1/docs/ML_METHODS.md`, `ML1/docs/EVALUATION.md`, `ML1/docs/MODEL_CHOICE.md`, `ML1/docs/API.md`, `ML1/PITCH.md`. Companion files in this folder: `VISION_EXPLAINED.md`, `ML2_EXPLAINED.md`, `ARDUINO_EXPLAINED.md`.

---

## Table of contents

1. [What ML1 does](#1-what-ml1-does)
2. [Technology stack](#2-technology-stack)
3. [Folder structure](#3-folder-structure)
4. [Architecture and data flow](#4-architecture-and-data-flow)
5. [The pipeline, step by step](#5-the-pipeline-step-by-step)
6. [The LLM layer](#6-the-llm-layer)
7. [Safety architecture (guards and invariants)](#7-safety-architecture)
8. [The engine (state machine, timing)](#8-the-engine-state-machine-timing)
9. [Probability matrix (HMM)](#9-probability-matrix-hmm)
10. [Fuzzy ECU controller](#10-fuzzy-ecu-controller)
11. [Input / output: MQTT and HTTP](#11-input--output-mqtt-and-http)
12. [Reports, dashboard, test data](#12-reports-dashboard-test-data)
13. [Evaluation and results](#13-evaluation-and-results)
14. [Running ML1](#14-running-ml1)
15. [Key design principles](#15-key-design-principles)
16. [ML concepts used (linked to the course)](#16-ml-concepts-used)
17. [Glossary](#17-glossary)
18. [Known limitations](#18-known-limitations)

---

## 1. What ML1 does

ML1 is the **brain** of Vigil. It receives what the camera sees (`vigil/vision`) and what the driver answered (`vigil/driver_response`), and decides:

- **how serious the situation is** — a severity from 0 to 3;
- **what should happen** — a list of actions (speak, alarm, light the ESP32, ask a question, call 112 — simulated);
- **how the car should behave** — a continuous command for the ECU (speed, lane assist, following distance, emergency stop on the shoulder).

It then publishes the decision for **ML2** (which executes it) and the dashboard.

| Severity | Meaning | Standard actions |
|---|---|---|
| 0 | attentive | none |
| 1 | distraction (phone, looking away) | a fixed voice reminder |
| 2 | fatigue (PERCLOS, microsleep, frequent yawning, low score) | alarm 2, ESP32 2, suggest a break, "Doriți o pauză?" |
| 3 | medical emergency / no reaction | alarm 3, ESP32 3, "Sunteți bine?"; **112 only if there is no answer or the driver says he is not OK** |

Everything runs **locally**: the LLM is a small model (Qwen2.5 1.5B) served by Ollama on the laptop.

---

## 2. Technology stack

| Component | Used for |
|---|---|
| **Python 3.10+, standard library** | the whole core (fusion, rules, LLM client, HTTP API) uses only the stdlib (`json`, `urllib`, `threading`, `http.server`…) |
| **Ollama** | local LLM inference server (`http://localhost:11434`), with structured output |
| **Qwen2.5 1.5B** (Q4_K_M, ~1 GB, Apache 2.0) | the LLM; fallback `llama3.2:1b` (`VIGIL_MODEL` env var) |
| **paho-mqtt** | MQTT bridge to the Mosquitto broker |
| **NumPy, Matplotlib** | only for the evaluation and its figures |
| **gcc** (optional) | compiles and tests the C fuzzy lookup table |
| **LaTeX** | builds `docs/TECHNICAL_DEEP_DIVE.pdf` |

Why a small local model: privacy (no camera data leaves the car), no internet dependency, predictable latency, and it fits the target hardware — an in-car cockpit / driver-monitoring controller (Snapdragon Cockpit, NVIDIA DRIVE Orin), not a classic microcontroller ECU. Measured on an RTX 4050: median ~260–364 ms per decision, 100 % valid JSON. Details in `docs/MODEL_CHOICE.md`.

---

## 3. Folder structure

```
ML1/
├── run_ml1.py            the single entry point (all commands)
├── requirements.txt      numpy, matplotlib, paho-mqtt
├── PITCH.md              3-minute pitch + likely jury questions
├── ml1/
│   ├── cv_adapter.py     CV / Spring Boot messages → internal form (names, units, cleaning)
│   ├── calibration.py    personal baseline from the first 30 s (median / IQR)
│   ├── fusion.py         attention score a ∈ [0,1] + confidence
│   ├── rules.py          deterministic severity ladder (the safety path)
│   ├── profile.py        thresholds aligned with the CV semantics
│   ├── prompt.py         system prompt + few-shot examples + per-frame context line
│   ├── llm_client.py     Ollama client (options, warmup, timeouts)
│   ├── actions.py        the action menu the LLM chooses from + JSON schema
│   ├── tools.py          the 6 tools of the vigil/decision contract
│   ├── validate.py       parse + validate + safety guards; fallback to rules
│   ├── engine.py         DecisionEngine: ties everything together
│   ├── probability.py    probability matrix (HMM) + statistics; prob_matrix.json = learned matrices
│   ├── ecu_fuzzy.py      fuzzy controller + supervisor for the ECU, C export
│   ├── service.py        transport-independent service (ordering, duplicates, decision stream)
│   ├── mqtt_bridge.py    MQTT transport
│   ├── http_api.py       REST + SSE transport (for a Spring Boot server)
│   ├── reports.py        112 report and post-trip report (templates)
│   ├── dashboard.py, web/index.html, mock_pose.py   ML1's own replay dashboard
│   └── evaluate.py, bench.py, fake_vision.py, replay.py   evaluation, benchmark, test data
├── ecu/                  vigil_fuzzy_lut.h (generated C table) + test_lut.c
├── tests/                137 tests + tests/data/*.jsonl scenarios
└── docs/                 documentation, figures, PDF, LaTeX source
```

---

## 4. Architecture and data flow

```
vigil/vision ─┐
              ├─► normalize ─► calibrate ─► fuse (score a, confidence) ─► RULES (safety path) ─┬─► vigil/decision
driver_resp ──┘                                                     │                          │
                                                                    └─► LLM (only on change) ─► guards ─┘
                     rules + a ─► probability matrix (HMM) ─► fuzzy + supervisor ─► vigil/ecu (5 Hz)
```

Three ideas carry the whole design:

1. **Deterministic core, probabilistic advisor.** The rules decide what is safety-relevant; the LLM refines *within limits the rules set*. This follows ISO 26262: the part that can cause harm (a false 112 call, a missed emergency) must be testable and explainable.
2. **Time is a signal.** Most hard cases are about duration: a 1 s glance vs 6 s on the phone, a 1.5 s chest touch vs 5 s with heavy breathing, one lost face frame vs 1 s of absence. Smoothing, persistence thresholds, hysteresis and timeouts hold most of the logic.
3. **Measure, don't assume.** Prompt size, the LLM's value, the calibration effect and latencies were all measured; several "obvious" choices turned out wrong.

---

## 5. The pipeline, step by step

### 5.1 Normalize — `cv_adapter.py`

The **single boundary** between the CV schema and ML1. If CV renames something, only this file changes.
- Event aliases: `phone_use` → `phone_in_hand` (CV2's name vs ML1's name).
- Nested fields flattened: `breathing.state` → `breathing_state`, `posture.state` → `posture_state`.
- Cleaning: non-finite numbers → `None`; values **clipped** to physical ranges (clipping keeps information, dropping a frame loses it).
- Landmarks removed — they never reach the rules or the LLM.
- Also accepts Spring Boot camelCase / millisecond / envelope formats.
- `cv_demo_messages()` imports the Vision team's own fake generator, so ML1 is tested on exactly the shape the camera produces.

### 5.2 Calibrate — `calibration.py`

Problem: thresholds tuned for an "average" driver misfire on someone with naturally narrow eyes (EAR 0.22, measured PERCLOS 0.15) — in the evaluation this produced 21 false alarms/hour.

`DriverBaseline` learns the driver's normal EAR, blink rate and PERCLOS offset from the first 30 s:
- **only clean frames** (face visible, gaze on road, no events, no phone) — otherwise a yawn would be learned as normal;
- **robust statistics**: median and IQR, not mean/std (one long blink barely moves a median);
- **plausibility bounds**: a baseline PERCLOS above 0.20 is rejected (a driver who is already drowsy must not teach the system that drowsy is normal);
- **fit on the past only** (train/transform discipline applied in time);
- fallback to population values after 3 × 30 s without enough clean frames.

### 5.3 Fuse — `fusion.py`: one attention score

Each available signal gets a sub-score in [0, 1] (1 = attentive), combined with weights:

| Signal | Weight |
|---|---|
| eyes (PERCLOS, EAR vs baseline) | 0.30 |
| gaze, head, phone | 0.15 each |
| events (hand on chest, eyes closed, head drop…) | 0.10 |
| blink rate, voice, driving | 0.05 each |

Mechanics:
- **Only available signals count**; weights are renormalized over those present.
- **Caps** (`CAPS`): a serious signal *limits* the total instead of being averaged away. Example from a real trace: weighted mean 0.903 (looks attentive) but hand-on-chest + heavy breathing caps it at `0.40 + 0.60·0.28 = 0.568`.
- **Asymmetric smoothing** (EMA): `α = 1 − exp(−Δt/τ)`, with **τ = 0.7 s when falling, 4.0 s when rising** — a real problem shows up within ~1 s, but one good frame can't "un-alarm" the system.
- **`critical` bypass**: head dropped + eyes closed, a near-certain severe event, or no answer + eyes closed skip smoothing entirely.
- **Missing data is not neutral**: camera off (privacy choice) → lower *confidence*, not inattention; face missing with the camera on for ≥ 1 s → "not looking at the road".
- **Confidence** is coverage-based: camera 0.6, voice 0.2, driving 0.2 (a heuristic, not a calibrated probability). `camera_cov` reflects which laptops are sending (both ≈ 1, face only ≈ 0.8, body only ≈ 0.45).

### 5.4 Rules — `rules.py`: the severity ladder

`rule_severity` is an **ordered ladder: first match wins**, from most to least severe. Order is meaning — moving a line silently changes behaviour (there is a test exercise about exactly this).

```
3  critical (fusion) | head drop + eyes closed | head_on_wheel ≥ 2 s | slumped & still ≥ 2 s | slumped_side
3  eyes closed ≥ 3 s, continuous
3  eyes closed ≥ 1.5 s AND (no answer | distress)
3  hand on chest ≥ 3 s AND heavy breathing          (→ 2 if a quick coherent "sunt bine")
3  (no answer | distress) AND a < 0.5
2  eyes closed ≥ 1.5 s (microsleep) | PERCLOS ≥ 0.15 | a < 0.5
2  frequent yawning | low blink rate ≥ 5 s + PERCLOS ≥ 0.12 | yawn + PERCLOS ≥ 0.12
1  phone ≥ 0.6 | phone event ≥ 2 s | gaze off ≥ 2 s | off_road_ratio ≥ 0.5 | bad posture | out of position
0  short medical sign (< 3 s): watch, don't remind
1  a < 0.75 for ≥ 4 s
0  otherwise
```

Details:
- Eye closures during a yawn or while looking down (phone) are **artifacts** and never count as microsleep.
- "PERCLOS" prefers a 15 s closure fraction (from CV or ML1's own estimate), else CV's 60 s value corrected by calibration.
- `answer_status` turns ML2's answer into `none / no_answer / distress / slow / ok` with two regular expressions (deterministic, testable, but crude).
- `wants_112`: severity 3 **and** (no answer or distress) **to "Sunteți bine?"** specifically. Silence after "Doriți o pauză?" first escalates to an urgent check, never straight to 112.
- All CV-dependent thresholds live in `profile.py` (one place to change when CV changes a definition).

---

## 6. The LLM layer

A 1.5B model is unreliable at free text and at native function calling, so its job is narrowed to **choosing a severity and picking items from a menu**.

### 6.1 Action menu — `actions.py`

The LLM never writes what the driver hears. It picks IDs; ML1 expands them into tool calls with **pre-written Romanian phrases**:

| ID | Tool | Argument |
|---|---|---|
| `remind_road` | speak | "Vă rugăm să păstrați privirea la drum." |
| `remind_phone` | speak | "Vă rugăm să lăsați telefonul și să priviți drumul." |
| `remind_posture`, `remind_position` | speak | posture / back to driving position |
| `alarm` | play_alarm | level derived from severity |
| `esp32` | set_esp32 | level derived from severity |
| `suggest_break` | suggest_break | — |
| `ask_tired` | ask_driver | "Păreți obosit. Doriți să faceți o pauză?" |
| `ask_ok` | ask_driver | "Sunteți bine?" |
| `call_112` | call_112 | report built from a template |

Result: ~25 output tokens instead of ~80, zero hallucinations in voice messages (an automotive HMI requirement).

### 6.2 Structured output (schema-constrained decoding)

The JSON schema (`severity ∈ {0..3}`, `actions` from the enum, `reason` string) is sent as Ollama's `format`. llama.cpp compiles it into a grammar and **masks every token that would violate it** during sampling — the model physically *cannot* produce invalid JSON or an unknown action. It can still pick a wrong *valid* action, which is why validation exists.

### 6.3 Prompt — `prompt.py`

- **Static prefix**: system prompt + 5 fixed few-shot examples, byte-identical on every call, so the inference server reuses the **KV cache**. Measured: 976 prompt tokens processed in only 9–18 ms; generation is the real cost.
- **Per-frame line** (~80 new tokens), e.g.
  `a=0.58 conf=0.60 cam=on | perclos=0.05 ear=0.29 blink=34 | gaze_on_road=yes head_drop=no phone=0.00 | breathing=heavy posture=upright | events: hand_on_chest 5s heavy | a 10s ago=0.39 | answer: no answer | hint=3`
- **`hint=`** gives the rule severity: it anchors the small model, whose job becomes "agree, or argue down by one level with context".

### 6.4 Client settings — `llm_client.py`

| Option | Value | Why |
|---|---|---|
| `temperature`, `top_k`, `seed` | 0, 1, 42 | deterministic, reproducible |
| `num_predict` | 48 | caps worst-case latency |
| `num_ctx` | 2048 | was 1024 until a 976-token prompt overflowed and Ollama silently dropped the start of the prompt (the rules) |
| `keep_alive` | 60 min | no reload between calls |
| `warmup()` | one call at start | first load ~3 s (~90 s cold); changing any option reloads the model |

---

## 7. Safety architecture

### 7.1 `validate.py` — the trust boundary

`decide()` **never raises**. Pipeline:

```
LLM call in a worker thread, timeout 2.5 s ──fail──► fallback_decision (rules)
  └► parse_llm_output   JSON from text / Ollama message / tool_calls (tolerates fences, chatter)
  └► expand_actions     IDs → tool calls
  └► validate_decision  types, ranges, unknown tools dropped; fatal → rules
  └► apply_safety       guards, in order:
       1  floor     severity ≥ (3 if rules = 3 else rules − 1)
       1b ceiling   severity ≤ rules (LLM_MAX_RAISE = 0); 0 if rules = 0
       1c evidence  phone/road reminders need the matching sign in the data
       1d voice     severity ≥ 2 gets a question; at 3 it is "Sunteți bine?"
       2  112       removed without evidence; added when due
       3  levels    alarm/ESP32 level ≤ severity; severity 3 forces level 3
       4  empty     severity > 0 with no valid action → standard actions
```

Every intervention is logged in `safety_notes` (shown on the dashboard). 26 adversarial LLM outputs are tested in `tests/data/llm_outputs.json`.

### 7.2 Invariants (always true)

| Invariant | Where |
|---|---|
| A clear severity-3 case is never downgraded by the LLM | floor guard |
| The LLM never raises severity above the rules | ceiling guard |
| 112 only at severity 3 **and** no answer / distress to "Sunteți bine?" | `wants_112` + guard 2 |
| 112 at most once per episode; no question repeated while waiting | engine `_publish` |
| Severity 3 never waits for the LLM | engine fast path |
| The driver only hears pre-written phrases | action menu |
| Any LLM failure (timeout, crash, bad JSON) yields a valid rule-based decision | `decide` fallback |

Standards behind this: **ISO 26262** (functional safety — the LLM is "QM", it has no safety rating, so it's only an advisor) and **ISO 21448 / SOTIF** (sensor and ML performance limits — hence calibration, persistence thresholds, reduced confidence without camera).

---

## 8. The engine (state machine, timing)

`engine.py → DecisionEngine` holds the state of one trip: last published decision, last question and answer, whether 112 was called, calm/downgrade timers, the LLM signature and busy flag, calibration, and the real car state reported by the ECU.

**Per frame:**

```
normalize → fusion.update → calibration.update → build context → rule_severity
1. FAST PATH   rules = 3 and (not already 3 | new severe reason | "Sunteți bine?" not yet asked | 112 now due)
               → publish the rules decision IMMEDIATELY
2. ALL CLEAR   rules = 0 and a ≥ 0.85 for 5 s → publish severity 0, reset the episode
3. LLM         only if the situation signature changed
               (rule severity, active events, a in quarters, answer status), or a 15 s heartbeat while non-zero;
               ≥ 1 s apart, at most one call in flight
```

| Timer | Value | Effect |
|---|---|---|
| `ask_timeout` | 8 s | silence after a question = "no answer" (works even if ML2's microphone is down) |
| all clear | 5 s calm | no flicker between episodes |
| downgrade | 5 s at a lower level, never while awaiting an answer | stable severity (hysteresis) |
| min gap / heartbeat | 1 s / 15 s | ~11 LLM calls per 150 frames of drowsiness, **0 when driving normally** |

**Async mode** (MQTT bridge, dashboard): the LLM runs in a worker thread. When an answer returns, it is **discarded** if a newer decision was published meanwhile or if it falls below the *current* rule floor — a slow LLM can never overwrite a newer fast-path decision.

**Publishing** strips a repeated `call_112`, questions after 112 or while awaiting an answer, and LLM decisions that add nothing new for the driver.

---

## 9. Probability matrix (HMM) — `probability.py`

**Problem:** the published severity is a *decision*, not a probability. On test data a published "2" means real fatigue in only 43 % of frames — the rest is hysteresis (the driver is already attentive). Fine for an alarm, not fine for braking a car.

**Model:** a **Hidden Markov Model**.
- Hidden states: the true severity 0–3 (attentive, distracted, tired, emergency).
- Observation per frame: 15 symbols = (rule level 0, 1, 2, 3 or **0+**) × (a ≥ 0.75 | 0.5–0.75 | < 0.5). "0+" = rules still say 0 but see a sub-threshold medical sign (hand on chest).
- **Emission matrix** `E[s, o] = P(o | s)`: counted on training data, Laplace smoothing, weighted by camera coverage.
- **Transition** `T(dt) = exp(Q·dt)`: a continuous-time Markov chain; mean durations learned: attentive 106 s, distracted 6 s, tired 30 s, emergency 208 s.

**Filter (forward algorithm)**, pure Python, ~4–6 µs per frame:
```
b ← normalize( E[:, o]^w ⊙ (T(dt)ᵀ · b) ),   w = min(1, dt·evidence_hz) · min(1, coverage / 0.7)
```
- Evidence counted **per second** (`evidence_hz = 1`, chosen on validation by log-loss), not per frame: neighbouring frames of the same gesture are not independent, and the same data at 10 Hz vs 5 Hz now changes the belief by 0.02 instead of 0.88.
- One-camera frames move the belief less (coverage weight) but still move it — a collapse seen only by the body camera is caught.
- Camera off → prediction only (missing observation is not evidence of attention).

**Bayes decision with asymmetric cost:** `sev* = argmin_p Σ_s b[s]·COST[s, p]` (missing an emergency costs ~30× a false alarm). Example: `b = [0.8, 0, 0, 0.2]` → most likely state is 0, but the Bayes decision is 3.

**Results (per frame, test / shift):** Cohen's kappa 0.922 / 0.910 vs rules 0.789 / 0.707; cost per frame 0.037 / 0.049 vs 0.099 / 0.122; emergency recall 1.00. The matrix feeds the **ECU and dashboard only** — `vigil/decision` still comes from rules + LLM. Learned matrices are in `ml1/prob_matrix.json`.

---

## 10. Fuzzy ECU controller — `ecu_fuzzy.py`

Turns the driver state into a car command (published on `vigil/ecu` at 5 Hz; ML2 translates it into commands for the ECU board — see `ML2_EXPLAINED.md` and `ARDUINO_EXPLAINED.md`).

**Inputs:** attention score `a`, risk = `0.5·P(tired) + P(emergency)` from the HMM, measurement confidence.
**Outputs:** speed factor (fraction of set speed), lane-keeping authority (0 = driver, 1 = system), headway in seconds.

**Mamdani inference:** AND = min, implication = min (clipping), aggregation = max, **centroid defuzzification**. Eight readable rules cover every input combination (checked by tests):

```
IF a high   AND risk low  AND conf high → speed normal, authority driver, headway normal
IF a medium AND risk low                → reduced, shared, large
IF a low    AND risk low                → slow, shared, large
IF a high   AND risk medium             → reduced, shared, large
IF a medium AND risk medium             → slow, shared, max
IF a low    AND risk medium             → slow, system, max
IF risk high                            → crawl, system, max
IF conf low AND risk low                → reduced, shared, large     (camera off: caution)
```

| Situation | Speed | Assist | Headway | Mode |
|---|---|---|---|---|
| attentive | 100 % | 0.00 | 1.8 s | NORMAL |
| phone in hand | 73 % | 0.50 | 2.5 s | CAUTION |
| drowsy | 60 % | 1.00 | 3.3 s | ASSIST |
| medical emergency | 30 % → MRM | 1.00 | 3.3 s | ASSIST + hazards → MRM |
| camera off | 82 % | 0.50 | 2.5 s | CAUTION |

**Deterministic supervisor** on top (same philosophy as `rules.py`):
- published severity 3 → speed ≤ 50 %, hazards on, lane assist 100 %, immediately;
- **MRM — Minimal Risk Maneuver** (pull over to the shoulder, speed → 0 at 2.5 m/s²) after the 112 call, or after 10 s of severity 3 **confirmed by the matrix** (P(emergency) ≥ 0.5);
- rate limits: 0.5 m/s² gentle slowdown, 2 m/s² at high risk, 1 m/s² when recovering.

Results: attentive 97 % of set speed, tired 65 %, emergency 24 %; MRM reached in all emergencies (median ~10–11 s), **0 false MRMs**.

`vigil/ecu` message: `mode` (NORMAL | CAUTION | ASSIST | MRM), `target_kmh`, `speed_ref_kmh`, `decel_mps2`, `lka_authority`, `headway_s`, `hazards`, `lane`, plus the inputs, raw fuzzy outputs and HMM state.

### On a real ECU: the C lookup table

`python run_ml1.py ecu --export-c` compiles the fuzzy controller into `ecu/vigil_fuzzy_lut.h`: a **15×15×9 grid of uint8** values (6 KB in flash) with **trilinear interpolation** in C99 — no allocation, no recursion, bounded time, no LLM, no NumPy. Mean error vs exact inference < 0.7 %. `tests/test_probability_ecu.py` compiles it with `gcc -Wall -Wextra -Werror` and compares C and Python outputs. The supervisor is not in the header; it belongs to the ECU safety layer.

---

## 11. Input / output: MQTT and HTTP

`service.py` is transport-independent (ordering, duplicates, timestamps, decision stream); MQTT and HTTP both use it.

### MQTT topics (`mqtt_bridge.py`)

| Topic | Direction | Content |
|---|---|---|
| `vigil/vision` | Vision → ML1 | camera signals (10 Hz) |
| `vigil/driver_response` | ML2 → ML1 | `{"answered", "latency_sec", "text"}` |
| `vigil/decision` | ML1 → ML2, dashboard | `{"a", "severity", "actions":[{"tool","args"}], "reason", "source", "ts", "latency_ms"?, "safety_notes"?}` |
| `vigil/ml1/status` | ML1 → dashboard | score, confidence, calibration, state, stats (1 Hz) |
| `vigil/ecu` | ML1 → ML2, dashboard | fuzzy car command (5 Hz) |
| `vigil/car/state`, `vigil/car/event` | ECU → ML1 | real (simulated) car speed, braking, hazards; used in status and 112 report |
| `vigil/ml1/cmd` | dashboard → ML1 | `{"cmd":"reset"}` = new trip, recalibrate |
| `vigil/assistant/reply` | ML2 → ML1 | Virgil's state (silence while Virgil talks doesn't count as "no answer") |

### HTTP API (`http_api.py`, port 8090, for a Spring Boot server)

`POST /api/v1/vision` (one frame or a list), `POST /api/v1/driver-response`, `GET /api/v1/decisions?since=N`, `GET /api/v1/decisions/latest`, `GET /api/v1/decisions/stream` (**Server-Sent Events**), `GET /api/v1/status`, `POST /api/v1/reset`, `GET /api/v1/health`. Optional token. Java examples in `docs/API.md`.

---

## 12. Reports, dashboard, test data

- **`reports.py`** — the 112 report and the post-trip report are built from **templates**: instant, and they contain only observed facts ("signs observed by the camera, not a diagnosis"), plus the car state from the ECU.
- **ML1 dashboard** (`dashboard.py` + `web/index.html`) — replays a scenario in real time through the same engine, with a simulated body skeleton (`mock_pose.py`), the score chart, signal contributions, decisions, safety notes and the 112 report.
- **Test data** (`tests/data/*.jsonl`) — scenarios `demo, cv_demo, phone, drowsy, chest, collapse, privacy, noisy, glance, calibration`; regenerated with a fixed seed by `run_ml1.py data`. `fake_vision.py` and `replay.py` publish/replay them.

---

## 13. Evaluation and results

`evaluate.py` (`run_ml1.py evaluate`) builds a full evaluation on simulated drivers:

- **Simulator** (`TripSim`): a latent state per frame (fatigue level, phone, chest, collapse…) → a CV-format message with driver-specific noise; events only after CV's minimum durations; it imitates the two-laptop `merge.py` output (missing camera fields, `cameras`), posture events, breathing disabled on half the trips, and benign 2–5 s chest gestures.
- **Closed loop**: simulated driver answers depend on the true state, seeded so compared variants get identical answers.
- **Split by driver** (unseen drivers in test) + a **shift** domain (different drivers, e.g. narrow eyes).
- **Grace windows**: frames just after a true transition are excluded (designed latency and hysteresis), latency measured separately.
- **Metrics**: per-class precision/recall/F1, macro-F1, **cost-weighted confusion** (under-estimating by k levels costs 5k, ×2 when the truth is 3; over-estimating costs k), kappa, MCC.
- **Bootstrap CIs** by resampling *trips* (not frames), 400×.
- **Baselines**: softmax regression (gradient descent, L2, class weights) and a depth-4 Gini decision tree; thresholds on `a` tuned on validation.

Key results (simulated, unseen drivers):

| | macro-F1 | cost |
|---|---|---|
| legacy thresholds | 0.75 | 0.171 |
| CV-aligned rules + calibration | 0.83 | 0.090 |
| + LLM | 0.89 | 0.078 |

- **0 missed emergencies, 0 false 112 calls.**
- Calibration on the shift domain: macro-F1 0.58 → 0.82, false alarms 62 → 20 per hour.
- Detection latency (simulated): collapse 0.2–2 s, chest 3 s; 112 about 8–11 s after onset.
- Full tables: `docs/EVALUATION_RESULTS.md`, figures in `docs/figures/`.

---

## 14. Running ML1

From `ML1/` (uses the root `.venv` automatically):

| Command | What it does |
|---|---|
| `python run_ml1.py check` | checks Ollama, the model, Python packages, the broker |
| `python run_ml1.py test` | all 137 tests (~1 min, no Ollama, no broker) |
| `python run_ml1.py dashboard --autostart demo --speed 2 --llm` | ML1 dashboard with a simulated body |
| `python run_ml1.py scenario collapse [--llm]` | one scenario in the console with the 112 report |
| `python run_ml1.py bridge --llm [--host IP]` | **ML1 live on MQTT** (connected to Vision and ML2) |
| `python run_ml1.py api --llm` | HTTP + SSE on port 8090 |
| `python run_ml1.py evaluate [--llm] [--quick]` | full evaluation → `docs/EVALUATION_RESULTS.md` |
| `python run_ml1.py bench` | LLM latency and validity on this machine |
| `python run_ml1.py prob [--fit]` | show / relearn the probability matrices |
| `python run_ml1.py ecu [--bench] [--export-c]` | fuzzy controller on typical cases / export the C table |
| `python run_ml1.py data` | regenerate test scenarios |
| `python run_ml1.py pdf` | rebuild the technical PDF |

Install: `pip install -r requirements.txt` and `ollama pull qwen2.5:1.5b`. Another model: `set VIGIL_MODEL=llama3.2:1b`. Without Ollama, ML1 still works with rules only. The full system is started from `START/`.

---

## 15. Key design principles

1. **Rules are the safety path; the LLM is an advisor** that can only refine within bounds.
2. **The LLM chooses, it doesn't write** — action IDs and fixed phrases; schema-constrained decoding.
3. **Never wait for the LLM in an emergency** — fast path from the rules.
4. **Call the LLM only when the situation changes** — cheap, quiet, and no voice spam.
5. **Every failure has a fallback** — timeout/crash/bad JSON → rule decision; no Ollama → rules only; no broker → HTTP.
6. **Hysteresis and timers everywhere** — stable decisions, no flicker, no repeated questions or 112s.
7. **Personal calibration with robust statistics**, learned only from clean frames.
8. **Missing data has meaning** — camera off ≠ face lost ≠ inattentive.
9. **Probabilities for actuation, decisions for alarms** — the HMM + Bayes cost decides how hard the car reacts.
10. **Embedded-friendly** — fuzzy compiled to a 6 KB C table; HMM in microseconds.

---

## 16. ML concepts used

`docs/ML_METHODS.md` links each method to the course lectures:

| Lecture | Applied in ML1 |
|---|---|
| C1 — problem framing, inductive bias, no free lunch | rules as a strong prior; when to use an LLM |
| C2 — data and preprocessing | clipping, missing data mechanisms (MCAR vs MNAR), robust scaling (median/IQR), fit on train/past only |
| C3 — linear regression, optimization | gradient descent for the softmax baseline |
| C4 — classification, logistic regression | softmax, cross-entropy, decision thresholds vs probabilities |
| C5 — evaluation and generalization | split by subject, per-class metrics, cost-based thresholds, bootstrap, calibration, distribution shift |
| C6 — classical ML (k-NN, SVM, trees) | Gini decision tree baseline |
| outside the course | HMM / forward filter, Bayes risk decision, Mamdani fuzzy logic, schema-constrained decoding, KV cache, ISO 26262 / SOTIF |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **a (attention score)** | ML1's 0–1 summary of how attentive the driver is |
| **Severity** | 0–3 decision level that drives the actions |
| **Fast path** | publishing a rule decision immediately, without waiting for the LLM |
| **Structured output** | constraining LLM decoding with a JSON schema / grammar |
| **KV cache / prefix caching** | reusing the attention keys/values of an unchanged prompt prefix |
| **Few-shot examples** | example input→output pairs placed in the prompt |
| **Guard** | a deterministic check that corrects or rejects an LLM output |
| **Hysteresis** | different conditions for going up vs coming down a level |
| **EMA** | exponential moving average with time constant τ |
| **HMM** | Hidden Markov Model: hidden states, emissions, transitions |
| **Forward filter** | recursive Bayesian update of the state belief `b` |
| **Bayes decision** | choose the action minimizing expected cost under the belief |
| **Mamdani fuzzy** | rule-based fuzzy inference with min/max and centroid defuzzification |
| **MRM** | Minimal Risk Maneuver: controlled stop on the shoulder with hazards |
| **LKA** | Lane-Keeping Assist (authority 0 = driver, 1 = system) |
| **Headway** | following distance in seconds to the car ahead |
| **Macro-F1, kappa, MCC** | class-balanced agreement metrics |
| **Bootstrap CI** | confidence interval from resampling trips |
| **ISO 26262 / SOTIF (ISO 21448)** | automotive functional-safety and intended-functionality standards |
| **SSE** | Server-Sent Events: one-way HTTP stream from server to client |

---

## 18. Known limitations

1. **All evaluation data is simulated** — good for comparing variants and catching regressions, not a real-world accuracy estimate. Next step: real labelled recordings (`docs/EVALUATION.md` §6).
2. **`confidence` is a coverage heuristic**, not a calibrated probability.
3. **Fusion weights, caps and time constants are hand-chosen**; only a few parameters were chosen from data.
4. **`answer_status` uses keyword regexes**; real speech-to-text output will be messier.
5. **The LLM's gain is not statistically significant** on test, and `LLM_MAX_RAISE` was chosen after seeing the test result.
6. **Severity-2 recall ~0.79**: a tired driver who stays silent is escalated to 3 (intended, but costs precision).
7. **One driver per process**; a driver change needs `reset`.
8. **Matrices learned on the same simulator** that produces test data; on real data the emission matrix must be relearned.
9. **The ECU loop is not closed with a real car or `highway-env`** — the board in `ECU_Arduino/` only simulates the car.
