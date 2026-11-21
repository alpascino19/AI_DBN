# ML1 technical deep dive: a guided reading path

> For: the ML1 owner and anyone who must explain, debug or extend the decision engine.
> Assumes Python and the ML course basics (C1–C6). Every number below was measured on the demo
> laptop (RTX 4050, Qwen2.5 1.5B) unless marked otherwise. Code links point to the exact lines.

## 0. How to read this

| You have | Read |
|---|---|
| 30 min | §1 (mental model) → §2 (one frame's journey) → §6 (limitations) |
| 2 h | add §3.3 fusion, §3.4 rules, §3.7 validate, §3.8 engine, and run exercises 1–4 in §7 |
| a day | everything, in order, with the code open next to it, plus all exercises |

Companion documents: [ARCHITECTURE.md](ARCHITECTURE.md) (what talks to what),
[ML_METHODS.md](ML_METHODS.md) (why, per lecture), [EVALUATION.md](EVALUATION.md) (how we measure).

---

## 1. The mental model

```
 CV (perception)                ML1 (decision)                                   ML2 (action)
 camera → numbers   ──vision──► normalize → calibrate → fuse → RULES ─────┐  ──decision──► voice, alarm,
                                                                 │         ├─► engine ──►   ESP32, 112
                                                                 └─► LLM ─► validate ─┘
                    ◄──────────────────── driver_response (answer to "Sunteți bine?") ───────────
```

Three ideas carry the whole design. If you remember only these, you can reconstruct the rest.

1. **Deterministic core, probabilistic advisor.** The rules ([rules.py](../ml1/rules.py)) decide what
   is safety-relevant. The LLM refines inside limits the rules set. This is the ISO 26262 pattern: the
   part that can cause harm (a false 112 call, a missed emergency) must be testable and explainable.
2. **Time is a signal.** Almost every hard case is about duration: a 1 s glance vs 6 s on the phone,
   a 1.5 s chest touch vs 5 s with heavy breathing, one lost face frame vs 1 s of absence. Smoothing,
   persistence thresholds, hysteresis and timeouts are where most of the logic lives.
3. **Measure, don't assume.** Several "obvious" choices were wrong until measured: the prompt size,
   the LLM's value, the calibration effect, a latency of 0.1 s. §2 and §6 show those.

### Invariants: what must always stay true

| Invariant | Enforced in |
|---|---|
| A clear severity-3 case is never downgraded by the LLM | [validate.py:255](../ml1/validate.py#L255) (floor) |
| The LLM never raises severity above the rules | [validate.py:263](../ml1/validate.py#L263) (ceiling, `LLM_MAX_RAISE = 0`) |
| 112 only at severity 3 **and** no answer / distress to "Sunteți bine?" | [rules.py:127](../ml1/rules.py#L127), [validate.py:307](../ml1/validate.py#L307) |
| 112 at most once per episode; no question repeated while waiting | [engine.py:223](../ml1/engine.py#L223) |
| Severity 3 never waits for the LLM | [engine.py:118](../ml1/engine.py#L118) (fast path) |
| The driver only hears pre-written Romanian phrases | [actions.py:17](../ml1/actions.py#L17) |
| Any LLM failure (timeout, crash, bad JSON) yields a valid rule-based decision | [validate.py:341](../ml1/validate.py#L341) |

---

## 2. One frame's journey (real trace)

Input: a frame from the CV team's own fake generator (`Vision/vigil_vision/fake`), phase `chest`,
t = 45.0 s, after the earlier phases of their demo. Reproduce it with exercise 1 (§7).

### 2.1 Raw CV message (`vigil/vision`)

```json
{"ear": 0.29, "perclos": 0.05, "blink_rate": 34.0, "gaze_on_road": true, "head_drop": false, "phone": 0.0,
 "breathing": {"state": "heavy", "rate_bpm": 32.0, "amp_ratio": 3.1, "mouth_open": true},
 "posture": {"state": "upright", "still_sec": 0.0},
 "events": [{"event": "hand_on_chest", "duration_sec": 5.0, "confidence": 0.8, "breathing": "heavy"},
            {"event": "heavy_breathing", "duration_sec": 4.0, "confidence": 0.75}],
 "ts": 45.0, "camera_on": true, "face_detected": true, "landmarks": ["...468 points..."]}
```

### 2.2 Normalize ([cv_adapter.py:44](../ml1/cv_adapter.py#L44))

Landmarks removed; `breathing.state` → `breathing_state: "heavy"`; `posture.state` →
`posture_state: "upright"`; numbers clipped to physical ranges. Event names would be mapped here
(`phone_use` → `phone_in_hand`), but this frame needs no mapping.

### 2.3 Fuse ([fusion.py:209](../ml1/fusion.py#L209))

Each available signal gets a sub-score in [0, 1] (1 = attentive):

| Signal | Base weight | Sub-score | Why |
|---|---|---|---|
| eyes | 0.30 | 1.000 | PERCLOS 0.05, EAR 0.29 ≈ the driver's calibrated 0.30 |
| gaze | 0.15 | 1.000 | on road |
| head | 0.15 | 1.000 | not dropped |
| phone | 0.15 | 1.000 | 0.0 |
| blink | 0.05 | 0.686 | 34/min vs personal normal 15.4: upper limit max(25, 1.8×15.4) = 27.7, ramp to 47.7 → (47.7−34)/20 |
| events | 0.10 | 0.280 | hand on chest + heavy: penalty max(0.6×0.8, 0.9×0.8) × duration factor 1.0 = 0.72 |

Only available signals count (voice and driving are absent), so the weights are re-normalized over
0.90 (eyes 0.30/0.90 = 0.333 …):

```
weighted = 0.333·1 + 0.167·1 + 0.167·1 + 0.167·1 + 0.056·0.686 + 0.111·0.28 = 0.903
```

0.903 would look like an attentive driver. That's why **caps** exist ([fusion.py:61](../ml1/fusion.py#L61)):
a serious signal limits the total instead of being averaged away.

```
events cap = 0.40 + 0.60 × 0.28 = 0.568   →   a_raw = min(0.903, 0.568) = 0.568   (capped_by = "events")
```

Then **asymmetric smoothing** ([fusion.py:280](../ml1/fusion.py#L280)), an exponential moving average
with a time constant that depends on direction:

```
α = 1 − exp(−Δt / τ),   τ = 0.7 s when falling, 4.0 s when rising
a ← a + α (a_raw − a)
```

Result: **a = 0.585** (a little behind a_raw: smoothing lag). Falling fast means a real problem shows
up within ~1 s; rising slowly means one good frame can't "un-alarm" the system.
**Confidence = 0.60**: camera only (voice and driving sources add 0.2 each when present).

### 2.4 Calibrate ([calibration.py:46](../ml1/calibration.py#L46))

Already done in the first 30 s of this trip: `ear = 0.30, blink = 15.4, perclos_offset = 0.0`.
That's where the blink sub-score's personal limits come from.

### 2.5 Rules ([rules.py:74](../ml1/rules.py#L74))

```
rule_severity → (3, "mână pe piept + respirație grea")
```

Hand on chest ≥ 3 s with confidence ≥ 0.6, and heavy breathing (three sources accepted: the event's
`breathing` attribute, a `heavy_breathing` event, or `breathing_state`). The context also contains:

```json
"driver_response": {"answered": false, "timeout": true, "question": "Sunteți bine?"}
```

The engine asked "Sunteți bine?" at 35.6 s. In the CV demo nobody answers (ML2 isn't connected), so
after 8 s of silence the engine records "no answer" ([engine.py:182](../ml1/engine.py#L182)).

### 2.6 Prompt ([prompt.py:59](../ml1/prompt.py#L59))

```
a=0.58 conf=0.60 cam=on | perclos=0.05 ear=0.29 blink=34 | gaze_on_road=yes head_drop=no phone=0.00
| breathing=heavy posture=upright | events: hand_on_chest 5s heavy, heavy_breathing 4s
| a 10s ago=0.39 | answer: no answer | hint=3
```

Messages = fixed system prompt + 5 fixed examples + this line. **976 prompt tokens** (measured with
Ollama's `prompt_eval_count`), but the prompt phase took only **9–18 ms**: the fixed prefix is reused
from the KV cache. The whole call took ~330 ms, so **generating tokens is the cost**.

### 2.7 LLM (Qwen2.5 1.5B, [llm_client.py:48](../ml1/llm_client.py#L48))

```json
{"severity": 3, "actions": ["alarm", "esp32", "ask_ok"], "reason": "cap închis, răsfoarea grea 5 s"}
```

42 output tokens. Three things to notice:
* the JSON is valid and the actions come from the menu: guaranteed by schema-constrained decoding (§3.5);
* the Romanian `reason` is garbled — this is why the model never writes what the driver hears;
* it **forgot `call_112`**, although the line says `answer: no answer` and the rules say 112 is due.

### 2.8 Validate and guard ([validate.py:341](../ml1/validate.py#L341))

1. `parse_llm_output`: extract the JSON;
2. `expand_actions`: `"alarm"` → `{"tool": "play_alarm", "args": {"level": 3}}` (level from severity);
3. `validate_decision`: known tools, typed arguments, severity in 0..3;
4. `apply_safety`: rules say 112 is due (severity 3 + no answer to "Sunteți bine?"), the LLM omitted
   it, so the guard **adds `call_112`** with the template report. Note logged:
   `call_112 adăugat: severitate 3 și șoferul nu a răspuns`.

### 2.9 Engine ([engine.py:93](../ml1/engine.py#L93))

In the real run the **fast path** had already published 112 at 43.6 s, the moment the rules saw
"severity 3 + no answer", without waiting for the LLM. So when this LLM decision arrives, `_publish`
strips the repeated `call_112` (112 once per episode), and the de-duplication drops it entirely
because it adds nothing new for the driver.

---

## 3. Module by module

### 3.1 `cv_adapter.py`: the boundary

* [`normalize_vision`](../ml1/cv_adapter.py#L44): the single place where the CV schema meets ML1.
  If CV renames a field or event, only `EVENT_ALIASES` ([line 23](../ml1/cv_adapter.py#L23)) changes.
* Cleaning (C2): non-finite → `None`; clip to `RANGES` ([line 31](../ml1/cv_adapter.py#L31)).
  Clipping, not dropping: a clipped value still carries information, a dropped frame doesn't.
* [`cv_demo_messages`](../ml1/cv_adapter.py#L99) imports the CV team's generator from `Vision/`, so
  ML1 is tested on exactly the shape the camera produces.

### 3.2 `calibration.py`: per-driver normalization

The problem: thresholds tuned on an "average" driver misfire on a driver with naturally narrow eyes
(EAR 0.22, measured PERCLOS 0.15). In the evaluation's shift domain, that produced 21 false alarms per hour.

How [`DriverBaseline.update`](../ml1/calibration.py#L66) solves it:
* **Fit on the past only**: learn from the first 30 s and apply to later frames, never the reverse
  (the train/transform discipline from C2, applied in time).
* **Only clean frames** ([`is_clean`](../ml1/calibration.py#L56)): face visible, gaze on road, no
  events, no phone. Otherwise a glance or a yawn would be learned as "normal".
* **Robust statistics**: median and IQR, not mean and standard deviation. One long blink (EAR 0.01)
  barely moves a median; it drags a mean.
* **Plausibility bounds** ([lines 21–24](../ml1/calibration.py#L21)): a baseline PERCLOS above 0.20
  is rejected. A driver who is already drowsy at start must not teach the system that drowsy is normal.
* **Fallback**: after 3 × 30 s without enough clean frames, use population values.

What uses it: the fusion (`baseline_ear`, `baseline_blink`, `perclos_offset`) and the rules
(`perclos_cal`, set in [engine.py:182](../ml1/engine.py#L182)).

### 3.3 `fusion.py`: from signals to one score

The constants are the model. Know where each comes from:

| Constant | Value | Meaning | Origin |
|---|---|---|---|
| `WEIGHTS` | eyes 0.30, gaze/head/phone 0.15, events 0.10, blink/voice/driving 0.05 | relative trust | hand-chosen (domain) |
| `CAPS` | eyes 0.35, events 0.40 | floor of the cap line `floor + (1−floor)·s` | hand-chosen, checked on scenarios |
| `TAU_DOWN` / `TAU_UP` | 0.7 s / 4.0 s | smoothing time constants | hand-chosen; the evaluation confirms no missed emergencies |
| `EVENT_SEVERITY` | head_drop/head_on_wheel 1.0, eyes_closed 0.8, … | penalty per event type | `eyes_closed` lowered 1.0 → 0.8 after the CV demo |
| `FACE_MISSING_SEC` | 1.0 s | lost face counts as "not looking" only after this | chosen from the evaluation (shift false alarms) |
| `LOW_A` + `LOW_A_PERSIST_SEC` | 0.75 for 4 s | generic "reduced attention" rule | chosen from the evaluation |
| `SOURCE_CONFIDENCE` | camera 0.6, voice 0.2, driving 0.2 | coverage-based confidence | hand-chosen (**not calibrated**, see §6) |

Two subtleties:
* **`critical`** ([fusion.py:274](../ml1/fusion.py#L274)) bypasses smoothing: head dropped + eyes closed,
  a near-certain severe event, or no answer + eyes closed. Waiting 1 s for the EMA is unacceptable there.
* **Missing data is not neutral.** Camera off (privacy, a user choice) → lower confidence, *not*
  inattention. Face missing with the camera on → after 1 s, gaze = 0. Same "missing value", different
  mechanism (C2: MCAR vs MNAR), different meaning.

### 3.4 `rules.py`: the decision ladder

[`rule_severity`](../ml1/rules.py#L74) is an ordered ladder: **first match wins**, from most to least
severe. Order is semantics. Moving the "eyes closed ≥ 1.5 s → 2" line above the "eyes closed + no
answer → 3" line would silently remove an emergency path.

```
3  critical (fusion) | head drop + eyes closed | head_on_wheel ≥ 2 s | slumped & still ≥ 2 s
3  eyes closed ≥ 3 s, continuous in the EAR too (not a tracker-merged episode)
3  eyes closed ≥ 1.5 s AND (no answer | distress)
3  hand on chest ≥ 3 s AND heavy breathing        (→ 2 if a quick coherent "sunt bine")
3  (no answer | distress) AND a < 0.5
2  eyes closed ≥ 1.5 s (microsleep) | microsleep | PERCLOS ≥ 0.15 | a < 0.5
2  frequent yawning | low blink rate ≥ 5 s + PERCLOS ≥ 0.12 | yawn + PERCLOS ≥ 0.12
1  phone ≥ 0.6 | phone event ≥ 2 s | gaze off ≥ 2 s | off_road_ratio ≥ 0.5
0  short medical sign (< 3 s): watch, don't remind
1  a < 0.75 for ≥ 4 s
0  otherwise
```

Eye closures during a yawn or while looking down (phone) are **artifacts** and never count as microsleep,
including a closure episode that began during one. "PERCLOS" means the 15 s closure fraction when
available (from CV, or ML1's estimate checked against CV), else CV's 60 s value corrected by calibration.
All of these thresholds live in [`profile.py`](../ml1/profile.py); see [CV_INTEGRATION.md](CV_INTEGRATION.md).

[`answer_status`](../ml1/rules.py#L60) turns ML2's answer into `none / no_answer / distress / slow / ok`
with two regular expressions. It's crude (see §6) but deterministic and testable.

[`wants_112`](../ml1/rules.py#L127): severity 3 and (no answer or distress) **to "Sunteți bine?"**.
Silence after "Doriți o pauză?" escalates to an urgent check first, never straight to 112.

### 3.5 `actions.py` + `prompt.py`: constraining a small model

A 1.5B model is unreliable at free-form generation and at native function calling. The design narrows
its job to choosing a severity and picking items from a menu.

* **Menu** ([actions.py](../ml1/actions.py)): 8 action IDs; levels derived from severity; phrases fixed.
* **Schema-constrained decoding** ([actions.py:43](../ml1/actions.py#L43)): the JSON schema is sent as
  Ollama's `format`. llama.cpp compiles it to a grammar and masks every token that would violate it
  while sampling. The model *cannot* emit invalid JSON or an action outside the enum. It can still pick
  the wrong valid action (as in §2.7), which is why validation exists.
* **Static prefix** ([prompt.py:19](../ml1/prompt.py#L19)): system prompt + 5 examples are byte-identical
  on every call, so the inference server reuses their KV cache. Measured: prompt phase 9–18 ms for
  976 tokens. Any per-call content (time, IDs) in the prefix would break this.
* **`hint=`**: the rule severity is given to the model. It anchors a small model, and the model's job
  becomes "agree, or argue down by one level with context".
* **Context window**: `num_ctx = 2048` ([llm_client.py:23](../ml1/llm_client.py#L23)). It was 1024 until
  the trace showed a 976-token prompt; with up to 48 output tokens that overflows, and Ollama then
  drops the *beginning* of the prompt, i.e. the rules. The token estimate was recalibrated (2.8 chars/token).

### 3.6 `llm_client.py`: inference settings

| Option | Value | Why |
|---|---|---|
| `temperature`, `top_k`, `seed` | 0, 1, 42 | deterministic, reproducible decisions |
| `num_predict` | 48 | a compact answer is ~25–42 tokens; caps the worst-case latency |
| `num_ctx` | 2048 | see above |
| `keep_alive` | 60 min | no model reload between calls |
| `warmup()` | one call with the same options | loading takes ~3 s (~90 s cold after reboot); changing any option above **reloads the model** |

### 3.7 `validate.py`: the trust boundary

[`decide`](../ml1/validate.py#L341) never raises. The pipeline:

```
llm_fn(ctx) in a worker thread, timeout 2.5 s ──fail──► fallback_decision (rules)
   └► parse_llm_output   (JSON from text / Ollama message / tool_calls; fences and chatter tolerated)
   └► expand_actions     (IDs → tool calls)
   └► validate_decision  (types, ranges, unknown tools dropped; fatal → rules)
   └► apply_safety       (guards, in this order:)
        1  floor     severity ≥ (3 if rules=3 else rules−1)
        1b ceiling   severity ≤ rules + LLM_MAX_RAISE (= rules); 0 if rules = 0
        1c evidence  phone/road reminders need the matching sign in the data
        1d voice     severity ≥ 2 gets a question; at 3 the question is "Sunteți bine?"
        2  112       removed without evidence; added when due
        3  levels    alarm/ESP32 level ≤ severity; severity 3 forces level 3
        4  empty     severity > 0 with no valid action → standard actions
```

Every intervention is logged in `safety_notes`, shown on the dashboard. On the 5 benchmark contexts
the guards intervene in 20% of answers (the drowsy case omits the voice question).

### 3.8 `engine.py`: the state machine

**State per trip:** last published decision (severity, time, tools), last question + time, last
answer, whether 112 was called, the timers `_calm_since` / `_lower_since`, the LLM signature and
busy flag, and the calibration.

**Per frame** ([engine.py:93](../ml1/engine.py#L93)):

```
normalize → fusion.update → track timeline → calibration.update → build ctx → rule_severity
1. FAST PATH  rules = 3 and (not already 3, or a new severe reason, or "Sunteți bine?" not yet asked,
              or 112 now due) → publish the rules decision immediately
2. ALL CLEAR  rules = 0 and a ≥ 0.85 continuously for 5 s → publish severity 0, reset the episode
3. LLM        signature = (rule severity, active events, a in quarters, answer status)
              call only if the signature changed (or 15 s heartbeat while non-zero), ≥ 1 s apart,
              and no call already in flight
```

**Timers:**

| Timer | Value | Effect |
|---|---|---|
| `ask_timeout` | 8 s | silence after a question = "no answer" (works even if ML2/STT is down) |
| all clear | 5 s of calm | prevents flicker between episodes |
| downgrade | 5 s at a lower level, never while awaiting an answer | stable severity |
| `min_gap` / `heartbeat` | 1 s / 15 s | LLM rate limit: ~11 calls per 150 frames of drowsiness, 0 when calm |

**Async mode** (`sync=False`, used by the dashboard and the MQTT bridge): the LLM runs in a worker
thread. When its answer returns, [`_llm_job`](../ml1/engine.py#L200) discards it if a newer decision was
published meanwhile, or if it would fall below the *current* rule floor. A slow LLM can never overwrite a
newer fast-path decision.

**Publishing** ([`_publish`](../ml1/engine.py#L223)): strips a repeated `call_112`, strips questions after
112 or while awaiting an answer, and (for LLM decisions) drops anything that adds nothing new for the driver.

### 3.9 The rest

* [`reports.py`](../ml1/reports.py): 112 and post-trip reports from templates. Facts only; "suspicion"
  is phrased as "signs observed by the camera, not a diagnosis".
* [`mqtt_bridge.py`](../ml1/mqtt_bridge.py): `Bridge` holds all logic and can be tested without a broker;
  `run_mqtt` only wires it to paho. Not yet run against a real broker.
* [`dashboard.py`](../ml1/dashboard.py) + [`web/index.html`](../ml1/web/index.html): replays a scenario in
  real time through the same engine; the body comes from [`mock_pose.py`](../ml1/mock_pose.py).

---

## 4. Evaluation internals ([evaluate.py](../ml1/evaluate.py))

* **Simulator** ([`TripSim`](../ml1/evaluate.py#L122)): latent state per frame (fatigue level D, phone,
  chest, collapse…) → CV-format message with driver-specific noise; events emitted only after CV's
  minimum durations. Timestamps are exactly `index / 5` (a drift bug here once produced fake 0.1 s latencies).
* **Closed loop** ([`respond`](../ml1/evaluate.py#L252)): answers depend on the true state, drawn from a
  generator seeded per (trip, episode, question type), so compared variants get identical answers.
* **Labels and grace** ([`_grace`](../ml1/evaluate.py#L235)): frames in the first 4 s after a true
  escalation and 12 s after a de-escalation are excluded (designed latency and hysteresis); latency is
  measured separately.
* **Metrics**: per-class precision/recall/F1 from the confusion matrix; macro-F1 = mean over present
  classes; cost = Σ confusion ⊙ `COST` / frames ([line 49](../ml1/evaluate.py#L49)), where under-estimating
  by k levels costs 5k (×2 when the truth is 3) and over-estimating costs k.
* **Bootstrap** ([line 450](../ml1/evaluate.py#L450)): resample *trips* (not frames) with replacement
  400×, recompute the metric from summed confusion matrices, take the 2.5–97.5 percentiles.
* **Thresholds on validation** ([`tune_a_thresholds`](../ml1/evaluate.py#L534)): grid over 3 cut points
  on `a`, minimizing cost on validation frames.
* **Classical models**: standardization fit on train ([`Standardizer`](../ml1/evaluate.py#L330));
  softmax regression with full-batch gradient descent, L2 and class weights
  ([`fit_softmax`](../ml1/evaluate.py#L347)); a weighted-Gini tree with depth 4 ([`Tree`](../ml1/evaluate.py#L371)).
* **How to read the near-perfect softmax (0.995)**: it learned the simulator that also produced the labels.
  That's an upper bound for "what a learned model could do with labels", not a real-world estimate.

---

## 5. Numbers cheat sheet

| Quantity | Value | Source |
|---|---|---|
| frame rate in ML1 tests / CV publish rate | 5 Hz / 10 Hz | simulator / `Vision/config.yaml` |
| CV event minimum durations | chest 2 s, breathing 3 s, phone 1 s, head on wheel 2 s | `Vision/config.yaml` |
| ML1 chest threshold | 3 s | rules |
| LLM latency | median ~260 ms, p95 ~370 ms; prompt phase 9–18 ms | measured |
| prompt / output tokens | ~976 / 25–42 | measured |
| model memory | 1.1 GB, 100% GPU, context 2048 | `ollama ps` |
| LLM timeout → rules | 2.5 s | validate |
| detection latency (simulated) | collapse 0.2–2 s, chest 3 s | evaluation |
| 112 after onset (simulated) | 8–11 s | evaluation |
| test: legacy thresholds / CV-aligned rules / + LLM | macro-F1 0.75 / 0.83 / 0.89; cost 0.171 / 0.090 / 0.078 | evaluation (CV-realistic simulator) |
| shift: no calibration / with calibration | macro-F1 0.58 / 0.82; false alarms 62 / 20 per hour | evaluation |

---

## 6. Known limitations and open questions

1. **All evaluation data is simulated.** Good for regressions and comparisons, not for real accuracy.
2. **`confidence` is a coverage heuristic, not a calibrated probability.** "0.6" means "camera only",
   not "60% likely correct".
3. **Fusion weights, caps and time constants are hand-chosen.** Only `FACE_MISSING_SEC`,
   `LOW_A_PERSIST_SEC`, `LLM_MAX_RAISE` and the `a` cut points were chosen from data.
4. **`answer_status` uses keyword regexes.** "Nu, sunt bine" ("No, I'm fine") contains "nu" but no distress
   keyword, so it's fine; "nu știu" ("I don't know") becomes "slow". Real STT output will be messier.
5. **The LLM's measured gain isn't statistically significant on test**, and `LLM_MAX_RAISE` was chosen
   after the test result had been seen.
6. **Severity-2 recall is ~0.79**: a tired driver who stays silent gets escalated to 3 (intended, but it
   costs severity-3 precision).
7. **MQTT bridge untested against a broker**; it needs `paho-mqtt`.
8. **Single trip state.** The engine assumes one driver per process; a driver change mid-trip would need
   recalibration.
9. **Prefix caching is assumed from timings** (prompt phase 9–18 ms), not from server internals; a
   different inference server may behave differently.

---

## 7. Hands-on exercises

Run from the repo root. Each exercise builds one intuition.

1. **Trace a frame yourself.** Replay CV's demo and print the fusion contributions at the chest phase:
   ```
   python -c "from ml1.cv_adapter import cv_demo_messages as m; from ml1.engine import DecisionEngine as E; e=E(); [e.on_vision(x) for x in m() if x['ts']<=45]; print(e.last_result.to_dict())"
   ```
   Recompute `a_raw` by hand from the contributions and the events cap.
2. **Watch smoothing.** `python -m ml1.replay tests/data/glance.jsonl --every 1`: a 1 s glance barely
   moves `a`. Now set `TAU_DOWN = 0.1` in `fusion.py` and replay. What would the driver hear?
3. **Break the ladder order.** In `rules.py`, move the "eyes closed ≥ 1.5 s → 2" line above the "eyes closed
   + no answer → 3" line and run `python -m unittest tests.test_integration_cv`. Which test catches it?
4. **See the guards work.** `python -m unittest tests.test_validate -v`: each of the 26 cases in
   `tests/data/llm_outputs.json` is a way an LLM can fail. Pick three and predict the outcome first.
5. **Feel calibration.** Run `python -m ml1.evaluate --quick` and compare the "rules fără calibrare" and
   "rules+calibrare" rows on `shift`. Then set `PERCLOS_MAX = 0.40` in `calibration.py` and rerun:
   what risk does that create?
6. **Measure the LLM.** `python -m ml1.bench --runs 5`, then change `num_predict` to 16 in
   `llm_client.py` (warm up first) and rerun. Latency drops; does validity?
7. **Async race.** In `tests/test_engine.py`, read `test_fast_path_does_not_wait_for_slow_llm` and
   `test_stale_llm_answer_discarded`, then explain `_llm_job`'s two discard conditions in your own words.
8. **Dashboard intuition.** `python -m ml1.dashboard --autostart cv_demo --speed 2`, then hover the score
   chart around the chest phase: find the moment `capped_by` switches to `events`.

---

## 8. Further reading

**Course (in this repo):**
* `ML_C2.pdf`: missing data mechanisms, robust scaling, fit/transform without leakage → §3.2, §3.3
* `ML_C4.pdf`: probability vs decision threshold, softmax, cross-entropy → §3.7, §4
* `ML_C5.pdf`: splits by subject, per-class metrics, cost-based thresholds, bootstrap, calibration,
  distribution shift → §4 and [EVALUATION.md](EVALUATION.md)
* `ML_C6.pdf`: decision trees and inductive bias → the tree baseline in §4

**External** (look these up in the original; titles given so you can find them):
* Ollama documentation, "Structured outputs" (the `format` parameter), and llama.cpp's GBNF grammars
  README: how schema-constrained decoding works.
* Qwen2.5 Technical Report (arXiv 2412.15115) and the Llama 3.2 model card: model capabilities and licenses.
* Dinges & Grace (1998), "PERCLOS: A valid psychophysiological measure of alertness as assessed by
  psychomotor vigilance" (FHWA tech brief): the origin of PERCLOS as a drowsiness measure.
* Åkerstedt & Gillberg (1990), "Subjective and objective sleepiness in the active individual":
  the Karolinska Sleepiness Scale (KSS), used for labeling real drowsiness data.
* Regulation (EU) 2019/2144 (General Safety Regulation): the DDAW/ADDW requirements behind the pitch.
* ISO 26262 (functional safety) and ISO 21448 / SOTIF (safety of the intended functionality, including
  sensor and ML performance limits): the reasoning behind §1 and the guards.
* Efron & Tibshirani, "An Introduction to the Bootstrap": the CI method in §4.
