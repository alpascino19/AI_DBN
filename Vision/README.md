# Vigil — Vision (CV1 + CV2)

Pipeline-ul de computer vision: camera → MediaPipe + YOLO → analizori → evenimente → MQTT `vigil/vision`.
Fiecare model rulează **o singură dată pe cadru**. Rezultatele stau în `FrameContext`, din care citesc toți analizorii.

```
camera ──► perception (Face / Pose / Hands / YOLO) ──► FrameContext ──► face/* (CV1) ┐
                                                                    └► body/* (CV2)  ├► events/tracker ─► vigil/vision
                                                                                      ┘
```

## CV 1 — Fața și ochii

| Fișier | Ce face |
|---|---|
| `vigil_vision/perception/face.py` | MediaPipe FaceLandmarker: landmarks, blendshapes (`eyeBlink*`, `jawOpen`), matricea de poziție a capului |
| `vigil_vision/face/eyes.py` | EAR, PERCLOS (fereastră glisantă), rata de clipire: prea rară = somnolență, prea deasă = clipit anormal |
| `vigil_vision/face/head_pose.py` | Pitch / yaw / roll + direcția privirii → `gaze_on_road` |
| `vigil_vision/face/unconscious.py` | Cap căzut + ochi închiși prelungit → `head_drop` (pierderea cunoștinței) |
| `vigil_vision/face/calibration.py` | Calibrarea din primele 30 s: praguri EAR / poziție neutră a capului adaptate șoferului |
| `dashboard/server.py` | FastAPI, abonat la `vigil/#`, trimite datele către browser prin WebSocket |
| `dashboard/static/index.html` | Scheletul feței (privacy mode), scorul de atenție, contribuția semnalelor, evenimentele active, ultima decizie a LLM-ului și motivul ei |

## CV 2 — Corp, mâini și obiecte

| Fișier | Ce face |
|---|---|
| `vigil_vision/camera.py` | Thread de captură: un singur proces deschide camera și păstrează doar ultimul cadru |
| `vigil_vision/pipeline.py` | Bucla principală: cadru → perception → analizori → tracker → publish |
| `vigil_vision/publisher.py` | Publică mesajul pe MQTT `vigil/vision` (sau pe consolă, pentru debug) |
| `vigil_vision/perception/body.py` | MediaPipe Pose + Hands |
| `vigil_vision/perception/objects.py` | YOLO, clasa „cell phone”, rulat la fiecare N cadre |
| `vigil_vision/body/chest.py` | Mâna pe piept: încheietura în zona toracelui câteva secunde → `hand_on_chest` |
| `vigil_vision/body/breathing.py` | Respirație grea: oscilația umerilor/pieptului + gura deschisă (`jawOpen` din CV1) |
| `vigil_vision/body/phone.py` | Telefon: cutia YOLO corelată cu mâna și cu privirea în jos → `phone` |
| `vigil_vision/events/tracker.py` | Detecții per cadru → evenimente cu durată și încredere (cu histerezis), ex. `hand_on_chest`, 4 s, 0.8 |

## Comune (stabilite împreună)

| Fișier | Ce face |
|---|---|
| `vigil_vision/schemas.py` | Contractul mesajului `vigil/vision` (vezi `TASKS.md`) |
| `vigil_vision/context.py` | `FrameContext`: rezultatele modelelor pentru cadrul curent + calibrarea |
| `vigil_vision/config.py`, `config.yaml` | Praguri, fps, cameră, broker MQTT |
| `vigil_vision/fake/fake_vision.py` | Generator de date false pe `vigil/vision` cu scenariile din demo |
| `vigil_vision/__main__.py` | Pornire: `python -m vigil_vision [--fake]` |

## Reguli

- Pragurile sunt în **secunde** (`time.monotonic()`), nu în număr de cadre: FPS-ul variază.
- Mesajul `vigil/vision` se publică la ~5–10 Hz, nu la fiecare cadru.
- Privacy mode: niciun cadru nu iese din pipeline. Pe MQTT ajung doar cifre și un subset de landmarks.
- CV1 și CV2 nu se importă reciproc: schimbă date doar prin `FrameContext`.
