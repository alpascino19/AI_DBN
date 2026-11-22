# Vigil — The Vision Module, Explained

This document explains everything inside the `Vision/` folder of AI_DBN (Vigil): what it does, how the data flows, which libraries and models are used, the algorithms and concepts behind each detector, how it talks to the rest of the system, and how to run, test and tune it.

> Related docs: `Vision/README.md` (short file map, Romanian), `Vision/COMPUTER_VISION.md` (user guide, Romanian), `docs/vision-doua-camere.md` (the two-camera MQTT design). This file is the "why and how" companion to them.

---

## Table of contents

1. [What Vision does](#1-what-vision-does)
2. [Technology stack](#2-technology-stack)
3. [Folder structure](#3-folder-structure)
4. [Architecture and data flow](#4-architecture-and-data-flow)
5. [Core infrastructure (camera, context, smoothing)](#5-core-infrastructure)
6. [Perception layer (the ML models)](#6-perception-layer-the-ml-models)
7. [CV1 — Face and eyes](#7-cv1--face-and-eyes)
8. [CV2 — Body, hands and objects](#8-cv2--body-hands-and-objects)
9. [From detections to events (EventTracker)](#9-from-detections-to-events-eventtracker)
10. [Local alerts (AlertManager)](#10-local-alerts-alertmanager)
11. [Output: the `vigil/vision` message and MQTT](#11-output-the-vigilvision-message-and-mqtt)
12. [Two-camera setup and `merge.py`](#12-two-camera-setup-and-mergepy)
13. [Live video for the dashboard](#13-live-video-for-the-dashboard)
14. [Run modes and commands](#14-run-modes-and-commands)
15. [Configuration (`config.yaml`)](#15-configuration-configyaml)
16. [Testing](#16-testing)
17. [Key design principles](#17-key-design-principles)
18. [Concepts glossary](#18-concepts-glossary)
19. [Known limitations and open points](#19-known-limitations-and-open-points)

---

## 1. What Vision does

Vision watches the driver through one or two webcams and turns pixels into **numbers, detections and events** that describe the driver's state. It answers questions like:

| Question | Detector |
|---|---|
| Are the eyes closed? For how long? Is the driver drowsy? | `face/eyes.py` (EAR, PERCLOS, blink rate) |
| Is the driver looking at the road? | `face/head_pose.py` (yaw/pitch/roll + iris + eyeLookDown) |
| Is the driver yawning (often)? | `face/yawn.py` (MAR + jawOpen) |
| Did the driver lose consciousness? | `face/unconscious.py` (head drop + closed eyes + stillness) |
| Is a hand pressed on the chest (possible heart problem)? | `body/chest.py` |
| Is the breathing heavy? | `body/breathing.py` (optical flow + FFT) — *currently disabled* |
| Is the driver using a phone? | `body/phone.py` (YOLO + hands + gaze) |
| Has the driver collapsed on the wheel / to the side / left the seat? | `body/slump.py` |

The results are:
- **published on MQTT** (`vigil/vision`) at ~10 Hz for **ML1** (the decision engine: rules + probability + local LLM), **ML2** (actions / voice / emergency) and the **React dashboard**;
- **shown locally** in an OpenCV window with a side panel, coloured warnings and beeps (useful for the demo and as a fallback if the LLM is slow).

**Privacy:** no video frame is ever saved. On the decision channel (`vigil/vision`) only numbers and a reduced face skeleton travel. The only exception is the optional JPEG stream for the dashboard (`vigil/video/*`, see §13), which can be turned off with `--no-video`.

---

## 2. Technology stack

| Library | Used for |
|---|---|
| **Python 3.10+** (tested on 3.13) | everything |
| **OpenCV** (`opencv-contrib-python`) | camera capture, colour conversion, drawing the debug/monitor windows, Farneback optical flow, Sobel gradients, JPEG encoding |
| **MediaPipe Tasks** (`mediapipe`) | three neural models: FaceLandmarker, PoseLandmarker, HandLandmarker |
| **Ultralytics YOLOv8** (`ultralytics`, brings PyTorch) | object detection — only the COCO class `cell phone` |
| **NumPy** | all the maths: distances, medians, FFT, PCA (eigen-decomposition), polyfit |
| **paho-mqtt** | publishing to the Mosquitto broker |
| **PyYAML** | loading `config.yaml` |
| **FastAPI + uvicorn** | a skeleton web dashboard in `Vision/dashboard/` (the real dashboard is the React app in `/dashboard`) |
| **winsound** (Windows stdlib) | alert beeps |
| `pytest` / `unittest` | tests |

### Models (in `Vision/models/`, auto-downloaded on first run)

| File | Size | Model | Output |
|---|---|---|---|
| `face_landmarker.task` | ~3.7 MB | MediaPipe Face Landmarker (with iris) | 478 3D face points, 52 blendshapes, 4×4 head transform matrix |
| `pose_landmarker_lite.task` | ~5.8 MB | MediaPipe Pose (lite) | 33 body points with visibility |
| `hand_landmarker.task` | ~7.8 MB | MediaPipe Hands | 21 points per hand, up to 2 hands, handedness |
| `yolov8n.pt` | ~6.5 MB | YOLOv8 nano, COCO-trained | bounding boxes (we keep class 67 "cell phone") |

All MediaPipe models run in **VIDEO mode** (they use temporal tracking between frames and need strictly increasing timestamps — the code enforces this with `max(ts, last_ts + 1)`).

---

## 3. Folder structure

```
Vision/
├── config.yaml              # all tunable thresholds (overrides defaults in config.py)
├── requirements.txt
├── start_face.bat / start_body.bat   # Windows launchers for the two-camera apps
├── models/                  # .task / .pt model files
├── dashboard/               # FastAPI skeleton (not the main dashboard)
├── tests/                   # ~90 unit tests on synthetic data, no camera needed
└── vigil_vision/
    ├── __main__.py          # `python -m vigil_vision` (single-camera pipeline / --fake)
    ├── camera.py            # capture thread, latest-frame-only, multiple readers
    ├── context.py           # FrameContext, Landmark, Hand, BBox, Detection, FaceOutput
    ├── schemas.py           # the vigil/vision message contract (VisionMessage, Event)
    ├── config.py            # defaults + YAML merge
    ├── smoothing.py         # One Euro filter, EMA, PoseSmoother
    ├── pipeline.py          # main loop for the single-camera mode
    ├── publisher.py         # MQTT publisher (console fallback)
    ├── merge.py             # combines vigil/vision/face + /body -> vigil/vision
    ├── video.py             # JPEG live stream for the dashboard
    ├── panel.py             # panel data (face_panel / body_panel) for the dashboard
    ├── debug_view.py        # CV2 debug window
    ├── run_face.py          # app: face camera only  -> vigil/vision/face
    ├── run_body.py          # app: body camera only  -> vigil/vision/body
    ├── perception/          # thin wrappers around the models
    │   ├── face.py          #   FaceLandmarker
    │   ├── body.py          #   Pose + Hands
    │   └── objects.py       #   YOLO
    ├── face/                # CV1 analysers
    │   ├── eyes.py  head_pose.py  yawn.py  unconscious.py  calibration.py
    │   ├── alerts.py        #   alert rules (face + body)
    │   ├── stack.py         #   FaceStack: whole CV1 for one frame -> FaceOutput
    │   └── monitor.py       #   live monitor app (face + body + alerts + sounds)
    ├── body/                # CV2 analysers
    │   ├── geometry.py  chest.py  breathing.py  phone.py  slump.py
    ├── events/tracker.py    # per-frame detections -> events with duration & confidence
    └── fake/fake_vision.py  # fake data generator with demo scenarios
```

Team split: **CV1** = face & eyes, **CV2** = body, hands, objects, plus shared infrastructure. A hard rule: *CV1 and CV2 never import each other*; they exchange data only through `FrameContext` / `FaceOutput`.

---

## 4. Architecture and data flow

### 4.1 Layers

Every frame goes through four layers:

```
1. Capture      camera.py          one thread, keeps only the newest frame
2. Perception   perception/*.py    each ML model runs ONCE per frame
3. Analysis     face/*, body/*     pure logic over landmarks -> signals + Detection objects
4. Temporal     events/tracker.py  detections -> events with duration/confidence
                face/alerts.py     detections -> local warnings (levels 1–3)
5. Output       publisher.py       VisionMessage JSON on MQTT at publish_hz (10 Hz)
```

The separation matters: perception is the expensive part, so it is done once and stored in `FrameContext`; analysers are cheap, deterministic and easy to unit-test with synthetic values.

### 4.2 Single-camera mode (`python -m vigil_vision`)

```
Camera ──► reader("cv1") ──► FaceWorker thread: FaceStack.process() ──► latest FaceOutput ─┐
       └─► reader("cv2") ──► Pipeline.run (main loop)                                      │
                              BodyPerception (Pose + Hands) → PoseSmoother                 │
                              ObjectDetector (YOLO every N frames)                         │
                              + latest FaceOutput (if < 0.5 s old)  ◄──────────────────────┘
                              analysers: BodyGeometry → Chest → [Breathing] → Phone → Slump
                              EventTracker → VisionMessage → Publisher ──► vigil/vision
```

Face and body run on **different threads, on the same frames**. If YOLO makes the body side slow, the face side is not slowed down, because each consumer has its own `FrameReader`.

CV1 plugs into CV2 through a **hook**: the pipeline imports `vigil_vision.face` and calls `create(cfg)`, which returns a `FaceStack` with `.process(frame) -> FaceOutput`. If the face module fails to load, the pipeline keeps working with body only (graceful degradation via `_try_build`).

### 4.3 Two-camera mode (used at the demo)

```
Laptop "face":  run_face --publish ─► vigil/vision/face ─┐
                                                         ├─► merge.py ─► vigil/vision ─► ML1, ML2, dashboard
Laptop "body":  run_body --publish ─► vigil/vision/body ─┘
```

A camera in front of the face sees eyes well; a camera at ~45° sees hands, chest and phone well. Each laptop publishes only what it sees well and `merge.py` combines them (§12). `START/run_vigil.py` starts all of this automatically.

### 4.4 Cross-module signals (CV1 → CV2)

The body analysers read some face signals via `ctx.signal(key)`:

| Signal from CV1 | Used by | Why |
|---|---|---|
| `jaw_open` | breathing | open mouth is part of the heavy-breathing score |
| `gaze_on_road` | phone | phone + looking away = more dangerous |
| `chin`, `mouth` | chest | a hand at the chin/mouth is not a hand on the chest |
| `face.detected` | slump | face disappearing supports "head on wheel" |

---

## 5. Core infrastructure

### 5.1 `camera.py` — capture thread, latest frame only

- **One process opens the camera once.** A background thread calls `cap.read()` in a loop.
- It keeps **only the newest frame** (no queue). If processing is slow, old frames are dropped instead of piling up → **no accumulated latency**. `CAP_PROP_BUFFERSIZE = 1` minimises driver-side buffering too.
- **Multiple readers:** `cam.reader("cv1")`, `cam.reader("cv2")`, `cam.reader("video0")`… Each reader remembers the last `frame_id` it saw and gets every *new* frame at most once. Synchronisation uses a `threading.Condition` (`wait` / `notify_all`).
- `Frame` is immutable (`frozen=True`) and shared — never draw on it in place (`.copy()` first). It caches the RGB conversion so MediaPipe models don't each convert again.
- Auto-reconnect if the camera drops; mirror flip (`mirror: true`) so it behaves like a mirror; on Windows uses the `CAP_DSHOW` backend.
- Each reader reports its own effective FPS.

### 5.2 `context.py` — the shared data model

- `Landmark(x, y, z, visibility)` — coordinates **normalised 0..1** (x right, y down), as MediaPipe.
- `Hand` (21 landmarks, handedness, `palm_center()` = mean of wrist + 4 finger bases).
- `BBox` (normalised box with `contains(x, y, pad)`).
- `Detection(name, active, confidence, attrs)` — what an analyser sees **in this frame** (no duration).
- `FaceOutput` — what CV1 hands over (signals, detections, skeleton landmarks).
- `FrameContext` — everything about one frame: pose, hands, objects, face, `signals`, `detections`, `events`, `geom`. Analysers implement the `Analyzer` protocol: `update(ctx) -> None`.
- `P` — named indices of the MediaPipe Pose points used (nose 0, shoulders 11/12, wrists 15/16, hips 23/24…).

### 5.3 `smoothing.py` — removing landmark jitter

MediaPipe points tremble a few pixels between frames. Two filters:

- **One Euro Filter** (Casiez et al., 2012): an adaptive low-pass filter. Cut-off frequency = `min_cutoff + beta·|speed|`. When the point is still → low cut-off → strong smoothing; when it moves fast → high cut-off → little lag. Ideal for shoulders and hands. Parameters in `config.yaml → smoothing`.
- **EMA with time constant τ** (`Ema(tau)`): `α = 1 − exp(−dt/τ)`. Because α depends on real `dt`, smoothing is correct **regardless of FPS**.

`PoseSmoother` applies One Euro to x/y of all 33 pose points and EMA to visibility, and resets if the person disappears > 0.5 s. It runs right after Pose, so all body analysers get stable points.

### 5.4 `config.py` + `config.yaml`

`config.py` holds all defaults; `config.yaml` overrides only what you change (deep merge). Helpers: `camera_cfg(cfg, "face"|"body")` merges `face_camera`/`body_camera` over `camera`; `model_path(cfg, key)`. CV1 thresholds live in dataclasses (`EyeConfig`, `HeadConfig`, `YawnConfig`, `UnconsciousConfig`).

---

## 6. Perception layer (the ML models)

### 6.1 `perception/face.py` — MediaPipe FaceLandmarker

Returns a `FaceResult`:
- `landmarks` (478 × 3): 468 face mesh points + 10 iris points (468–477);
- `blendshapes`: 52 expression scores 0..1 (e.g. `eyeBlinkLeft`, `jawOpen`, `eyeLookDownLeft`, `mouthSmileLeft`, `eyeSquintLeft`);
- `matrix`: 4×4 facial transformation matrix.

`Idx` names the indices used: 6 points per eye in EAR order (right eye 33,160,158,133,153,144; left 362,385,387,263,373,380), iris centres 468/473, nose tip 1, chin 152, forehead 10, mouth corners 61/291, lips 13/14.

### 6.2 `perception/body.py` — Pose + Hands

One call → `(pose | None, [Hand…])`. Pose gives 33 points with **visibility** (how sure the model is that the point is visible); Hands gives 21 points per hand with left/right label. Confidence thresholds in `perception:`.

### 6.3 `perception/objects.py` — YOLOv8n

- Filters to class `cell phone` only (`classes=[67]`), `imgsz=320`, `conf=0.35`.
- YOLO is the slowest model, so it runs **every N frames** (`every_n_frames: 5`); in between it returns the last boxes, as long as they are younger than `max_age_sec` (1 s). This keeps the phone analyser fed every frame at a fraction of the cost.

---

## 7. CV1 — Face and eyes

All of CV1 is glued together by `face/stack.py → FaceStack.process(frame)`:

```
FaceDetector → EyeAnalyzer → HeadPoseAnalyzer → YawnAnalyzer
             → Calibrator (first N seconds) → UnconsciousAnalyzer → FaceOutput
```

### 7.1 Eyes (`face/eyes.py`)

**EAR — Eye Aspect Ratio** (Soukupová & Čech, 2016):

```
EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)
```
vertical eyelid distances divided by eye width. ~0.25–0.35 open, drops towards ~0.1 closed. It is scale-invariant (does not depend on distance to camera).

**Closure score (0..1)** instead of a fixed EAR threshold — more robust:
1. EAR normalised to *this driver*: `(ear_open − ear) / (ear_open − ear_closed)`.
   `ear_open` comes from calibration; `ear_closed` is **learned from the driver's own blinks** (median of the minimum EAR of the last blinks, clamped to 25–60 % of `ear_open`).
2. Combined with MediaPipe's `eyeBlink` blendshape: `0.6·EAR_part + 0.4·eyeBlink`.
3. **Smile / squint compensation:** when smiling, the cheeks push the lower eyelid up and EAR drops although the eyes are open. If `mouthSmile` or `eyeSquint` > 0.35, the "closed" threshold rises from 0.72 to 0.85.

**Hysteresis:** closed when score > 0.72, open again only when < 0.50. Prevents flickering around one threshold.

**Blinks:** a closure lasting 0.05–0.5 s counts as a blink. Longer = a real closure (≥1.5 s = "long closed" / microsleep).

**PERCLOS** — PERcentage of eye CLOSure: fraction of *time* (sum of `dt`, not number of frames) with eyes closed over the last **60 s**. A classic drowsiness metric (≥15 % → fatigue). Frames without a visible face are not counted.

**Blink rate** (blinks/min over 60 s): < 6–8 → staring / drowsiness; > 22 → abnormal hyper-blinking. Not reported during the first 10 s (too little data).

**Fast recovery:** fatigue measured over 60 s would stay on for a whole minute. So there is a 15 s "recovery" window: if eyes closed ≤5 %, ≥1 normal blink, face visible ≥80 % → `recovered = True` and the fatigue alert turns off.

### 7.2 Head pose and gaze (`face/head_pose.py`)

**Angles directly from 3D landmarks** (no `solvePnP`):
- face centre = mean of eye outer corners + mouth corners; vector `n = nose_tip − centre` points out of the face;
- `yaw = atan2(n.x, −n.z)`, `pitch = atan2(n.y, −n.z)`, `roll` = angle of the line between the eye corners.

All angles are **relative to the neutral pose from calibration** (important: a laptop camera is usually lower than the face, so "looking at the road" is not 0°). EMA smoothing (0.35).

**Eyes on top of head:** the driver can keep the head straight and move only the eyes.
- **Iris offset**: iris position between the eye corners, −0.5..0.5. Effective yaw = `yaw + iris·60°`. Ignored when the head is turned > 15° (the eye is seen in profile and the iris position becomes misleading).
- **eyeLookDown** blendshape: effective pitch = `pitch + eye_down·35°` — catches "phone in the lap, only eyes down". Ignored when the head is tilted up (the eyes go down just to compensate).

**Direction** = up / down / right / left / center (yaw limit 23°, down 18°, up 20°). `gaze_on_road = (direction == center)`. Also reported: `off_road_sec` (continuous time off-road) and `off_road_ratio` (fraction of the last 10 s).

### 7.3 Yawning (`face/yawn.py`)

- **MAR — Mouth Aspect Ratio** = lip-to-lip distance / mouth width, plus `jawOpen` blendshape.
- Mouth wide = MAR > 0.55 or jawOpen > 0.40; closed with hysteresis (×0.7).
- A yawn = mouth wide for **1.5–8 s** (shorter = talking/laughing; longer = something else, e.g. heavy breathing). A short 0.4 s gap (hand in front of the mouth) does not break it.
- `yawns_recent` = yawns in the last 5 min; ≥ 3 → `frequent_yawning`.

### 7.4 Loss of consciousness (`face/unconscious.py`)

A fainting driver: head **falls** (forward or sideways), often **suddenly**, eyes **closed**, and **no movement**. Instead of an AND of rules, a **weighted score 0..1**:

| Component | Weight | Ramp |
|---|---|---|
| head dropped (pitch or relative roll) | 0.35 | 25° → 45° |
| eyes closed | 0.35 | 1 s → 3 s |
| stillness (angle spread < 3° over 1 s) | 0.20 | 1 s → 3 s |
| sudden fall (> 40°/s within ~0.5 s) | 0.10 | binary |

States: `ok` / `suspect` (≥0.45) / `unconscious` (≥0.65, exits below 0.40 — hysteresis). Without closed eyes the score cannot really reach 0.65, so looking down with open eyes is not a faint. Neutral roll is learned slowly (people keep the head slightly tilted). If the face disappears while unconscious (head on the wheel), the state is **held for 10 s**. `head_drop` = head dropped + eyes closed ≥1 s.

### 7.5 Calibration (`face/calibration.py`)

During the first N seconds (30 by default, `--calib 5` for tests) the driver sits straight, looks ahead, neutral face. The calibrator collects samples and takes the **median** (robust to outliers):
- `ear_open` (ignoring EAR < 0.15, i.e. blinks) → `EyeAnalyzer.set_open_baseline`
- `yaw0`, `pitch0`, `iris0`, `down0` → `HeadPoseAnalyzer.set_neutral`

This makes thresholds **personal**: small eyes, glasses, a camera placed low — all handled.

### 7.6 FaceOutput

`FaceStack` produces:
- **signals**: `ear, perclos, blink_rate, gaze_on_road, head_drop, jaw_open, eyes_closed, closed_sec, direction, yaw, pitch, yawning, yawns_recent, consciousness, unconscious_score, chin, mouth, calibrated`;
- **detections**: `eyes_closed, microsleep, gaze_off_road, head_drop, unconscious, yawn, hyper_blinking, low_blink_rate`;
- **landmarks**: a ~70-point **skeleton subset** (eyes, nose, chin, lips, face contour, eyebrows) — enough for the dashboard drawing, but not the full 478-point mesh (privacy + bandwidth).

---

## 8. CV2 — Body, hands and objects

Analyser order in the pipeline matters: `BodyGeometry → ChestAnalyzer → [BreathingAnalyzer] → PhoneAnalyzer → SlumpAnalyzer`.

### 8.1 Body geometry (`body/geometry.py`) — camera-angle independence

In a car the camera is rarely straight in front; at ~45° or from the side the shoulders overlap in the image, so "shoulder width in x" is a bad unit. `BodyGeometry` computes once per frame (`ctx.geom`):

- **`unit`** — body size in the image: 2D shoulder distance, or nose–shoulders distance × a ratio learned during calibration (whichever is larger), EMA-smoothed. **All body thresholds are expressed in "body units"**, so they work at any distance from the camera.
- **`facing`** — +1/−1/0, where the body faces in the image (nose position relative to shoulders, with hysteresis); default from `view.camera_side` + mirror.
- **`near`/`far` shoulder** — from MediaPipe z (smaller z = closer to camera).
- **`torso_angle`** — torso tilt vs vertical, if hips are visible.
- `view.mode`: `front` | `three_quarter` | `side`.

### 8.2 Hand on chest (`body/chest.py`)

The **chest zone** is built from geometry: horizontally from the shoulders, extended towards the *front* of the body (`view_forward[view]·unit`) and less towards the back (`view_back`); vertically from slightly above the shoulder line (`zone_top = −0.25`) down to 60 % of the torso, or `zone_bottom·unit` if hips aren't visible (usual in a car). The zone corners are smoothed with several EMAs so the box doesn't "breathe" with MediaPipe noise.

A hand counts as "on the chest" only if it passes **all** anti-false-alarm filters:

| # | Filter | Rejection label |
|---|---|---|
| 0 | not holding a phone (YOLO box) | `phone` |
| 1 | **not at the face** — fingers/palm far from chin & mouth (from CV1, or estimated from Pose) | `face` |
| 3 | **whole hand inside** — ≥60 % of the 21 points in the zone | `partial` |
| 3b | fingertips not above the top of the zone | `fingers_up` |
| 6 | **not near the camera** — palm size / body unit ≤ `max_hand_scale` (a hand held in front of the lens looks huge) | `close` |
| 4 | **hand is resting** — moved < 0.25 unit in the last 0.5 s | `moving` |
| 5 | depth — wrist not far in front of shoulders (z) — **off by default**, Pose z is too noisy | `depth` |

Hands (21 points) is the primary source; Pose (wrist + index) is only a **fallback when Hands sees no hand**, and off-image Pose points are ignored. Confidence grows with how central the hand is in the zone. Attributes: which side (left side = heart), both hands. The monitor shows *why* a hand was rejected — very useful for tuning.

### 8.3 Heavy breathing (`body/breathing.py`) — *disabled in `config.yaml`* (`breathing.enabled: false`, too many false alarms)

The most signal-processing-heavy part:

- **Source A — optical flow (primary):** dense **Farneback optical flow** between consecutive grayscale frames (downscaled to 320 px) inside the chest zone. Only pixels with texture (Sobel gradient > 20) are used; median motion (dx, dy). The **head's motion is subtracted** (removes whole-body motion and car vibration). The motion is integrated over time to get chest displacement. Jumps (a hand crossing) are rejected.
- **Source B — landmarks (fallback):** vertical shoulders–nose distance in body units, used when the shirt has no texture.
- **Analysis**, 5×/s over a ~10 s window: resample to 10 Hz → remove linear trend (`polyfit`) → **FFT** restricted to the breathing band **0.13–1.0 Hz (8–60 breaths/min)**. For 2D flow, **PCA** (eigenvectors of the covariance matrix) finds the main direction of chest motion automatically — up/down when seen from the front, forward/back from the side. Hanning window, peak frequency → **rate (bpm)**, peak power share → **periodicity**, band-passed std → **amplitude**.
- Each source learns its **normal amplitude** in the first `calib_sec` (30th percentile, so one deep breath doesn't spoil it).
- **Score:** amplitude ratio vs normal (last 4 s) 0.50 + rate 0.20 + periodicity 0.15 + open mouth (`jaw_open` from CV1) 0.15. Without large amplitude the score is capped below the exit threshold. Heavy = score ≥ 0.55 for 4 evaluations in a row; exits < 0.40 after 6 (hysteresis + debounce).
- If a hand is on the chest, the `hand_on_chest` event receives `"breathing": "heavy"|"normal"` (as the contract requires).

### 8.4 Phone (`body/phone.py`)

A phone seen by YOLO is not necessarily being used (it could be in a holder). The score:

```
raw = yolo_conf · (0.4 + 0.35·in_hand + 0.15·near_face + 0.25·looking_away)
```
- `in_hand`: palm centre / fingertips / Pose wrist inside the (padded) phone box;
- `near_face`: box centre within 0.6 unit of the nose → **call**;
- `looking_away`: `gaze_on_road == False` from CV1.

A phone that is neither in hand nor near the face is capped below the threshold (holder). Usage: `call` / `looking` / `holding`. Score rises instantly and decays smoothly. Output: `phone` (0..1) field and `phone_use` detection (≥ 0.4). ML1 renames `phone_use` → `phone_in_hand` in its `cv_adapter.py`.

### 8.5 Posture / collapse (`body/slump.py`)

When the driver falls on the wheel, the face disappears but Pose still sees the shoulders. After calibrating the upright posture (5 s), it measures:

- **drop** — how much the head (mean of nose/eyes/ears) came down towards the shoulders: `1 − gap/base_gap` (0 = upright, 1 = head at shoulder level);
- **lean** — max of: shoulders lower, body bigger (closer to camera), head in front of shoulders, torso tilted (each a ramp from `slump.lean`);
- **face missing** (from CV1 or head points gone);
- **stillness** — std of nose/shoulders/wrists over 1 s in body units (unseen points are NaN so their noise doesn't count).

`score = 0.6·drop + 0.2·face_missing + 0.2·lean`; reported **only if still for ≥ 2 s** (someone picking something up moves). Once slumped, small movements don't clear it; if Pose loses the person after a collapse, the state is held 15 s.

Detections produced:
| Detection | Meaning |
|---|---|
| `head_on_wheel` | collapsed forward, still |
| `slumped_side` | fell sideways (shoulder tilt / head-shoulder angle ≥ 25° vs normal) + still |
| `bad_posture` | leaning/hunched (event after 10 s) |
| `out_of_position` | body not visible, or much farther than normal (left the seat / leaned back) |

The posture line in the window: grey = upright, orange = leaning, red = slumped. `head_drop` in the message is true if **either** CV1 (face) **or** CV2 (`head_on_wheel`) sees it.

---

## 9. From detections to events (EventTracker)

`events/tracker.py` turns noisy per-frame booleans into **events with duration and confidence** — what the LLM actually reasons about. It is generic (any detection name, CV1 or CV2).

- An event starts at the first active frame.
- **Gap tolerance:** inactive for less than `gap_sec` (1 s) does not end it (hand slips for a moment, YOLO misses a frame).
- **Minimum duration** before it appears in the message (`events.min_duration_sec`): `hand_on_chest` 2 s, `heavy_breathing` 3 s, `phone_use` 1 s, `head_on_wheel` 2 s, `bad_posture` 10 s, default 1 s. A brief touch never reaches the LLM.
- **Confidence** = smoothed mean of per-frame confidences × √(fraction of frames active) × a ramp growing with duration.
- Keeps the last 20 ended events (for logs/dashboard).

Example output: `{"event": "hand_on_chest", "duration_sec": 4.0, "confidence": 0.8, "side": "left", "breathing": "normal"}`.

---

## 10. Local alerts (AlertManager)

`face/alerts.py` is a small **rule engine** that gives immediate feedback (demo + Plan B if the LLM is slow). The *official* decision stays with ML1.

Each `Rule` has: `code`, `level` (1–3), `message`, `condition(signals)`, `min_sec` (must hold that long), `repeat_sec`, `cooldown_sec`, `hides` (suppresses less severe/redundant alerts), `show_sec` (max display time).

| Level | Colour | Sound | Examples |
|---|---|---|---|
| 3 | red | 3 high beeps | eyes closed ≥3 s, unconscious (score ≥0.65, not with phone in hand), hand on chest ≥6 s, head on wheel, slumped sideways |
| 2 | orange | 2 beeps | eyes closed ≥1.5 s (not while yawning or looking down), head drop + eyes closed, fatigue (PERCLOS ≥15 % or low blink rate, 5 s), hyper-blinking, frequent yawning, distracted (≥50 % of 10 s off-road), hand on chest ≥2 s, heavy breathing, phone + looking away, out of position ≥10 s |
| 1 | yellow | 1 beep | single yawn, looking left/right/down/up ≥2 s, no face ≥3 s, phone in hand, out of position ≥3 s, bad posture ≥10 s |

Examples of `hides`: "unconscious" hides "eyes closed", "look down", "no face", "fatigue"; "phone + looking" hides "look down". Rules are split into `face_rules()` and `body_rules()` so the two-camera apps each use only theirs. Sounds use `winsound.Beep` in a separate thread (non-blocking, no overlap), falling back to the console bell.

---

## 11. Output: the `vigil/vision` message and MQTT

### 11.1 The contract (`schemas.py`)

```json
{
  "ear": 0.27, "perclos": 0.05, "blink_rate": 17.0,
  "gaze_on_road": true, "head_drop": false,
  "phone": 0.0,
  "breathing": {"state": "normal", "rate_bpm": 14},
  "posture": {"state": "upright", "score": 0.1, "drop": 0.0, "still_sec": 0.0, "calibrated": true},
  "events": [{"event": "hand_on_chest", "duration_sec": 4.0, "confidence": 0.8}],
  "landmarks": [[0.51, 0.42], ...],          // face skeleton subset (CV1)
  "body_landmarks": [[0.48, 0.61], ...],     // first 25 pose points (CV2)
  "ts": 1716300000.512, "camera_on": true,
  "face_detected": true, "body_detected": true, "fps": 24.8
}
```

Rules of the contract:
- Missing data is **`null`, never invented** — if the face isn't seen, `ear` is null and ML1 lowers its confidence.
- Extra fields (`face_panel`, `body_panel`, `cameras`, `fps`) are optional; consumers that don't know them ignore them.
- Published at `publish_hz` (10 Hz), not on every frame. If the camera is lost > 2 s, a message with `camera_on: false` is sent.
- No dependency on pydantic: plain dataclasses + `to_dict()` → `json.dumps`, rounded numbers.

### 11.2 `publisher.py`

- paho-mqtt (supports both 1.x and 2.x APIs), `connect_async` + `loop_start` → connects in the background and **reconnects automatically** (1–10 s backoff).
- **Unique client ID** `vigil-vision-<hostname>-<topic suffix>` — two laptops with the same ID would kick each other off the broker ("session taken over").
- QoS 0 (latest data matters more than guaranteed delivery).
- **Console fallback:** without a broker (or `--no-mqtt`) it prints a summary at most once a second, or immediately when the event list changes; landmarks are not printed.
- `publish_bytes()` for video frames: if the previous frame on that topic hasn't been sent yet, the new one is skipped → video stays live instead of lagging.

---

## 12. Two-camera setup and `merge.py`

Problem (analysed in `docs/vision-doua-camere.md`): two Vision instances publishing on the same `vigil/vision` would make ML1 alternately see "no face" and "no body", making decisions oscillate.

Solution: each laptop publishes on its own topic and `merge.py` (on the central PC with the broker) publishes one combined message at 10 Hz:

- From the **face** message it takes: `ear, perclos, blink_rate, gaze_on_road, landmarks, face_detected, face_panel`.
- From the **body** message: `phone, breathing, posture, body_landmarks, body_detected, body_panel`.
- `head_drop` = OR of both; `events` = union, with duplicates resolved by keeping the longest duration and the max confidence; `fps` = the minimum.
- A camera silent for > **1 s (STALE_SEC)** is considered lost: its fields disappear and `*_detected` becomes false.
- `cameras: {"face": bool, "body": bool}` lets ML1 distinguish "the face laptop lost Wi-Fi" (we don't know where the driver looks) from "camera works but no face visible" (driver not looking at the road).
- Subscribes to `vigil/vision/+` which does **not** match `vigil/vision` itself → no feedback loop. Publishes nothing until the first camera message arrives.

---

## 13. Live video for the dashboard (`video.py`)

The React dashboard can show the camera feed with the skeleton drawn on top:
- topics `vigil/video/face` and `vigil/video/body`, separate from `vigil/vision` (ML1 never receives images);
- JPEG, 640 px width, quality 60, up to 20 fps (~30 KB/frame, ~0.6 MB/s per camera);
- runs on **its own thread with its own camera reader**, so the body camera (Pose + Hands + YOLO ≈ 7–11 fps processing) can still stream at 20 fps;
- the same mirrored frame that the landmarks are computed on → drawn points align exactly;
- QoS 0, no retain (nothing stays on the broker); `--no-video` disables it (numbers only); `--video-fps`, `--video-width` for weak hotspots.

`panel.py` also adds `face_panel` / `body_panel` to the messages: the same values the local monitor draws (bars, states, active alerts), as clean JSON for the dashboard.

---

## 14. Run modes and commands

Install once (from the `Vision/` folder): `python -m pip install -r requirements.txt`. On Windows, if diacritics break the console: `set PYTHONUTF8=1`.

| Command | What it does |
|---|---|
| `python -m vigil_vision.face.monitor --show --calib 5` | **Main live monitor**: one camera, face + body + alerts + sounds. Options: `--no-body`, `--publish`, `--mute`, `--min-sound N`, `--index N` |
| `python -m vigil_vision --show` | CV2 pipeline with CV1 hooked in, debug window, publishes on `vigil/vision` |
| `python -m vigil_vision --no-mqtt` | same, console output only |
| `python -m vigil_vision --fake --scenario demo` | **fake data, no camera** — scenarios `demo, normal, phone, drowsy, chest, unconscious` (demo cycles through all) so ML1/ML2/dashboard can be developed without a camera |
| `python -m vigil_vision.run_face --show --calib 5 [--publish --mqtt-host IP]` | face-only app → `vigil/vision/face` (`start_face.bat`) |
| `python -m vigil_vision.run_body --show [--publish --mqtt-host IP]` | body-only app → `vigil/vision/body` (`start_body.bat`) |
| `python -m vigil_vision.merge [--broker IP]` | combine the two cameras into `vigil/vision` |
| `python -m vigil_vision.camera --show` | camera test with two simulated consumers |
| `python -m vigil_vision.perception.face --show` / `python -m vigil_vision.face.eyes --show` | quick model / EAR tests |

The whole system (broker, Vision, merge, ML1, ML2, dashboard) is started from `START/run_vigil.py` and the `.bat` files in `START/` (see `START/3_LAPTOPURI.md` for the 3-laptop setup).

### The monitor window

```
┌────────────────────────────┬──────────────────────┐
│  camera: eye points        │ VIGIL - SOFER  [OK]  │
│  (green open / red closed),│ EYES / GAZE /        │
│  hands, chest box,         │ CONSCIOUSNESS /      │
│  phone box, posture line   │ MOUTH / BODY bars    │
├────────────────────────────┴──────────────────────┤
│ ALERTS (band coloured by level)                   │
└───────────────────────────────────────────────────┘
```
Corner label: `CALIBRARE xx%` → `OK` → `NIVEL 1/2/3`. Bars: green normal, orange near threshold, red over threshold. During calibration: sit straight, look ahead, don't smile, breathe calmly.

---

## 15. Configuration (`config.yaml`)

| Section | Contents |
|---|---|
| `camera`, `face_camera`, `body_camera` | index, 640×480, 30 fps, mirror |
| `mqtt`, `publish_hz` | broker host/port/topic, console fallback, 10 Hz |
| `models` | model file names |
| `view` | `mode` (front / three_quarter / side), `camera_side`, body calibration time |
| `smoothing` | One Euro `min_cutoff`, `beta` |
| `perception` | Pose/Hands confidence, number of hands |
| `yolo` | `every_n_frames`, `imgsz`, `conf`, `max_age_sec` |
| `chest` | zone shape per view + all anti-false-alarm filters |
| `breathing` | `enabled` (false), window, FFT/score thresholds, flow settings |
| `slump` | drop/lean ramps, stillness, side-fall degrees, hold time |
| `phone` | hand padding, min score |
| `events` | `gap_sec`, `min_duration_sec` per event |

CV1 thresholds are in the dataclasses in `face/eyes.py`, `head_pose.py`, `yawn.py`, `unconscious.py`; alert durations/levels in `default_rules()` in `face/alerts.py`.

### Common tuning

| Problem | Change |
|---|---|
| "Eyes closed" when smiling/squinting | ↑ `EyeConfig.close_on`, ↑ `smile_close_on` |
| Phone-with-eyes-only not caught | ↓ `HeadConfig.eye_down_limit` |
| Hand on chest rejected as `close` | ↑ `chest.max_hand_scale` (check "hand size" on the panel) |
| Hand in front of camera accepted | ↓ `chest.max_hand_scale` |
| Hand on chest rejected as `face` | ↓ `chest.face_radius` |
| Chest box too small/large | `chest.view_forward`, `view_back`, `zone_top`, `zone_bottom` |
| Low FPS | ↑ `yolo.every_n_frames`, ↓ `yolo.imgsz`, or `--no-body` |
| Fatigue clears too slowly | ↓ `EyeConfig.recovery_window_sec` |

---

## 16. Testing

`Vision/tests/` — about 90 tests, all on **synthetic data** (no camera, no models needed):

| File | Covers |
|---|---|
| `test_eyes.py` | EAR, closure, blinks, PERCLOS, learning `ear_closed`, recovery |
| `test_yawn.py` | yawn duration window, gaps, frequent yawning |
| `test_unconscious.py` | score, states, hold when face disappears |
| `test_face_alerts.py` | rule timing, repeat, cooldown, `hides` |
| `test_cv2.py` | camera readers, geometry, chest, phone, tracker, pipeline message |
| `test_chest_filters.py` | each chest filter (face, partial, close, moving…) |
| `test_posture.py` | slump / side / bad posture / out of position |
| `test_merge.py` | two-camera merging, stale cameras, events dedup |
| `test_panel.py` | dashboard panel JSON |

Run: `python -m pytest tests` (install pytest) or `python -m unittest tests.test_cv2 tests.test_chest_filters`. Analysers expose value-based entry points (`update_ear`, `update_angles`, `update_values`) precisely so they can be tested without images.

---

## 17. Key design principles

1. **Each model runs once per frame**; results are shared via `FrameContext`.
2. **Time in seconds (`time.monotonic()`), never in frame counts** — FPS varies from 7 to 30, and thresholds must mean the same thing.
3. **Latest frame only** — no queues, no growing latency; drop rather than lag (camera, video, MQTT).
4. **Scores instead of hard AND rules** — weighted sums of ramps (`_ramp(x, lo, hi)`) give graceful, explainable decisions (`parts` / `chest_reject` are exposed for debugging).
5. **Hysteresis + minimum durations everywhere** — enter/exit thresholds, `enter_evals`, `min_duration_sec`, `gap_sec`: no flickering, no alarms from a single frame.
6. **Personal calibration** — EAR, head neutral pose, body proportions, normal breathing and upright posture are learned per driver; robust statistics (median, percentiles).
7. **Camera-angle independence** — body measurements in "body units" with view-dependent zones.
8. **Graceful degradation** — missing package/model/camera/face → that part is disabled, the rest keeps working, missing values are `null`.
9. **Loose coupling** — CV1 and CV2 don't import each other; Vision ↔ ML only via a JSON contract on MQTT.
10. **Privacy by design** — frames stay in memory; only numbers and a reduced skeleton go on the decision channel.

---

## 18. Concepts glossary

| Term | Meaning |
|---|---|
| **Landmark** | a keypoint (x, y, z, visibility) predicted by a neural network on a face/body/hand |
| **Blendshape** | a 0..1 coefficient describing a facial expression component (e.g. `jawOpen`) |
| **EAR** | Eye Aspect Ratio — eyelid opening / eye width |
| **MAR** | Mouth Aspect Ratio — lip opening / mouth width |
| **PERCLOS** | % of time eyes are closed in a window (60 s); standard drowsiness measure |
| **Microsleep** | a brief involuntary sleep episode, eyes closed ~0.5–several seconds |
| **Yaw / pitch / roll** | head rotation left-right / up-down / tilt to shoulder |
| **Hysteresis** | different thresholds for entering and leaving a state, to avoid flicker |
| **EMA** | Exponential Moving Average — simple low-pass smoothing |
| **One Euro Filter** | adaptive low-pass filter: smooth when slow, responsive when fast |
| **Optical flow (Farneback)** | per-pixel motion field between two frames |
| **FFT** | Fast Fourier Transform — finds the dominant frequency (breathing rate) |
| **PCA** | Principal Component Analysis — finds the main direction of motion |
| **Detrending** | removing a slow linear drift before frequency analysis |
| **YOLO** | "You Only Look Once" — single-pass real-time object detector |
| **COCO** | dataset with 80 object classes YOLOv8 is trained on ("cell phone" = 67) |
| **Body unit** | body size in the image, used to make thresholds distance-independent |
| **Detection vs Event** | detection = this frame; event = sustained detection with duration and confidence |
| **MQTT / topic / QoS** | lightweight publish-subscribe protocol; topic = channel name; QoS 0 = fire-and-forget |
| **Broker (Mosquitto)** | the MQTT server that routes messages between publishers and subscribers |

---

## 19. Known limitations and open points

- **Breathing analysis is disabled** (`breathing.enabled: false`) because of false alarms; it should also not run while a hand covers the chest. The `chest_long` (hand on chest ≥ 6 s) alert currently replaces the "hand on chest + heavy breathing" urgency.
- **Depth filter** for the chest is off: MediaPipe Pose z is too noisy at short distance.
- **Event naming**: CV2 sends `phone_use`, ML1 expects `phone_in_hand` — currently bridged by `ML1/ml1/cv_adapter.py`.
- `Vision/dashboard/` (FastAPI) is only a skeleton; the real dashboard is the React app in `/dashboard`.
- Lighting, sunglasses, strong head turns (profile) reduce face-landmark quality; iris is ignored above 15° yaw for this reason.
- Sounds use `winsound` → full beeps only on Windows.
- The video stream (when enabled) does send images over the network — fine on a private hotspot for the demo, but disable with `--no-video` for strict privacy mode.
