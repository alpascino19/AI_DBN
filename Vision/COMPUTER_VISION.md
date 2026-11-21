# Vigil — Computer Vision (CV1 + CV2)

Partea de computer vision din Vigil urmărește șoferul printr-o cameră web și transformă ce vede în **semnale**, **evenimente** și **avertizări**. Camera e citită de un singur proces. Fața (CV1) și corpul (CV2) sunt analizate pe același cadru, iar rezultatul pleacă pe MQTT (`vigil/vision`) către motorul de decizie LLM (ML1).

```
Cameră ─► CV1: față (ochi, privire, căscat, conștiență) ─┐
       └► CV2: corp (piept, respirație, telefon, postură) ┼─► avertizări locale (monitor)
                                                          └─► vigil/vision (MQTT) ─► ML1 (LLM)
```

Confidențialitate: niciun cadru video nu se salvează și nu se trimite în rețea. Pe MQTT ajung doar cifre și un schelet simplificat al feței.

---

## 1. Pornire rapidă

### Instalare (o singură dată)

Din folderul `Vision`:

```bash
python -m pip install -r requirements.txt
```

Ce trebuie să știi:
- **Python 3.10+.** Proiectul e testat pe Python 3.13 cu `mediapipe` 1.2.
- **Pachetul `ultralytics`**, pentru detecția telefonului cu YOLO, aduce și **PyTorch**, deci câteva sute de MB.
- **Modelele** (`face_landmarker.task`, `pose_landmarker_lite.task`, `hand_landmarker.task`, `yolov8n.pt`) se descarcă **automat** la prima pornire, în `Vision/models/`. Folderul e în `.gitignore`.
- **Pe Windows**, dacă apar erori cu diacriticele în consolă (`UnicodeEncodeError`), activează modul UTF-8: `$env:PYTHONUTF8=1` în PowerShell sau `set PYTHONUTF8=1` în cmd.

### Monitorul live (față + corp + avertizări)

Aplicația principală pentru demo și testare:

```bash
python -m vigil_vision.face.monitor --show --calib 5
```

Opțiuni:

| Opțiune | Ce face |
|---|---|
| `--show` | deschide fereastra cu camera, panoul cu detalii și avertizările |
| `--calib N` | calibrarea feței durează N secunde (implicit 30). La test, 5 e suficient |
| `--no-body` | doar fața, fără Pose/Hands/YOLO (mai rapid) |
| `--publish` | trimite și mesajul `vigil/vision` pe MQTT (pentru ML1 și dashboard) |
| `--mute` | fără sunete |
| `--min-sound N` | sunet doar de la nivelul N în sus (1 = toate, implicit) |
| `--index N` | altă cameră decât cea din `config.yaml` |
| `--config cale` | alt fișier de configurare |

Ieși cu tasta **q** în fereastră sau cu **Ctrl+C** în consolă.

### Pipeline-ul CV2 (fără monitor)

Pipeline-ul colegului de la CV2 rulează și singur. Fața (CV1) se conectează automat prin hook-ul `vigil_vision.face.create(cfg)`:

```bash
python -m vigil_vision --show
```

```bash
python -m vigil_vision --no-mqtt
```

```bash
python -m vigil_vision --fake --scenario demo
```

Ce face fiecare:
- **`--show`**: fereastra de debug a colegului (zona pieptului, telefonul, respirația);
- **`--no-mqtt`**: afișează mesajele doar în consolă;
- **`--fake --scenario demo`**: date false, fără cameră. Scenarii: `demo`, `normal`, `phone`, `drowsy`, `chest`, `unconscious`.

### Teste

Testele CV1 sunt funcții simple, iar cele CV2 folosesc `unittest`:

```bash
python -m unittest tests.test_cv2 tests.test_chest_filters
```

```bash
python -m pytest tests
```

`pytest` trebuie instalat separat (`python -m pip install pytest`). Toate testele rulează pe date sintetice, fără cameră.

---

## 2. Cum arată fereastra monitorului

```
┌────────────────────────────┬──────────────────────┐
│                            │ VIGIL - SOFER  [OK]  │
│                            │ OCHI                 │
│   camera                   │ PRIVIRE              │
│   (ochi, mâini, chenarul   │ CONSTIENTA           │
│    pieptului, telefon)     │ GURA                 │
│                            │ CORP                 │
├────────────────────────────┴──────────────────────┤
│ AVERTIZARI   (banda colorată după nivel)          │
└───────────────────────────────────────────────────┘
```

**Pe cameră:**
- punctele ochilor: verde = deschiși, roșu = închiși;
- cele 21 de puncte ale fiecărei mâini și centrul palmei;
- chenarul pieptului: verde, sau roșu când e o mână pe el;
- cutia telefonului, în mov;
- linia posturii: gri = drept, portocaliu = aplecat, roșu = prăbușit.

**Eticheta din colțul panoului:** `CALIBRARE xx%`, `OK` (verde) sau `NIVEL 1/2/3`. Marginea camerei are aceeași culoare.

**Barele din panou:** verde = normal, portocaliu = aproape de prag, roșu = peste prag.

**Rândul „Mână pe piept”** arată și de ce mâna **nu** a fost acceptată: `la fata`, `langa camera`, `in miscare`, `partial`, `degete sus`, `telefon`. Când e acceptată, arată sursa: `DA (mana)` sau `DA (schelet)`.

### Calibrarea

La pornire, stai **drept, privește înainte, normal, fără să zâmbești și respiră liniștit**. Sistemul învață:
- **CV1:** EAR-ul tău cu ochii deschiși, poziția neutră a capului și a privirii;
- **CV2:** proporțiile corpului, respirația normală și poziția dreaptă (fiecare cu `calib_sec` din `config.yaml`).

### Nivelurile de avertizare

| Nivel | Culoare | Sunet | Sens |
|---|---|---|---|
| 0 | verde | – | totul în regulă |
| 1 | galben | un bip | distragere scurtă |
| 2 | portocaliu | două bipuri | oboseală, distragere repetată, simptome |
| 3 | roșu | trei bipuri | urgență |

---

## 3. CV1 — Fața și ochii

Toate fișierele sunt în `vigil_vision/` (`perception/face.py` și folderul `face/`).

| Fișier | Ce face |
|---|---|
| `perception/face.py` | MediaPipe **FaceLandmarker**: 478 de puncte ale feței, blendshapes (`eyeBlink*`, `jawOpen`, `eyeLookDown*`, `mouthSmile*`…) și poziția capului. Descarcă modelul automat |
| `face/eyes.py` | **Ochii:** EAR și un **scor de închidere** (60% EAR raportat la ochii tăi + 40% `eyeBlink`), cu compensare la zâmbet și la ochi mijiți. Calculează clipirile, **PERCLOS** (ultimele 60 s), rata de clipire și revenirea rapidă din oboseală (ultimele 15 s normale) |
| `face/head_pose.py` | **Privirea:** yaw / pitch / roll din punctele feței, plus poziția irisului și privirea în jos cu ochii (`eyeLookDown`). Dă direcția (la drum / stânga / dreapta / jos / sus), `gaze_on_road`, de cât timp privește în afara drumului și procentul din ultimele 10 s |
| `face/yawn.py` | **Căscatul:** gura larg deschisă (MAR sau `jawOpen`) între 1.5 și 8 s; numără căscatele din ultimele 5 minute |
| `face/unconscious.py` | **Pierderea cunoștinței:** scor 0–1 din cap căzut (în față sau lateral), ochi închiși, nemișcare și cădere bruscă. Stări: normal / suspect / inconștient. Starea se păstrează 10 s dacă fața dispare |
| `face/calibration.py` | Calibrarea feței: EAR deschis, poziția neutră a capului, a irisului și a privirii în jos |
| `face/alerts.py` | **Regulile de avertizare** (față + corp): nivel, durată minimă, repetare, pauză și ce avertizări ascunde fiecare |
| `face/stack.py` | Leagă toată partea CV1 pentru un cadru și produce `FaceOutput` pentru pipeline (vezi secțiunea 5) |
| `face/monitor.py` | Aplicația live: cameră → față + corp → avertizări + fereastră + sunete |

### Avertizările din față

| Nivel | Avertizare | Când |
|---|---|---|
| 3 | Ochi închiși prea mult | ochi închiși ≥ 3 s |
| 3 | Posibilă pierdere a cunoștinței | scor de inconștiență ≥ 0.65 (nu și dacă e telefon în mână) |
| 2 | Ochi închiși | ≥ 1.5 s (nu în timpul căscatului și nu când privește în jos) |
| 2 | Cap căzut + ochi închiși | ochi închiși ≥ 1 s cu capul căzut |
| 2 | Oboseală | PERCLOS ≥ 15% sau sub 8 clipiri/min, timp de 5 s; dispare după ~15 s normale |
| 2 | Clipit excesiv | peste pragul `blink_rate_high` (clipiri/min), timp de 5 s |
| 2 | Căscat des | ≥ 3 căscate în 5 minute |
| 2 | Distras | ≥ 50% din ultimele 10 s în afara drumului |
| 1 | Căscat | un căscat |
| 1 | Privire stânga / dreapta / jos | ≥ 2 s continuu |
| 1 | Fără față | ≥ 3 s |

Pragurile se schimbă în clasele de configurare din fiecare fișier: `EyeConfig`, `HeadConfig`, `YawnConfig`, `UnconsciousConfig`. Duratele și nivelurile sunt în `default_rules()` din `face/alerts.py`.

---

## 4. CV2 — Corp, mâini și obiecte

| Fișier | Ce face |
|---|---|
| `camera.py` | Un singur thread citește camera și păstrează **doar ultimul cadru**. Fiecare consumator (`cv1`, `cv2`) primește fiecare cadru nou o singură dată |
| `perception/body.py` | MediaPipe **Pose** (33 de puncte ale corpului) și **Hands** (21 de puncte pe mână, până la 2 mâini) |
| `perception/objects.py` | **YOLO** (`yolov8n`), clasa „cell phone”, rulat la fiecare N cadre |
| `body/geometry.py` | Geometria corpului, valabilă oricare ar fi unghiul camerei (față / ~45° / lateral): mărimea corpului, direcția, umărul apropiat |
| `body/chest.py` | **Mâna pe piept**, cu filtre împotriva alarmelor false (vezi mai jos) |
| `body/breathing.py` | **Respirația grea:** mișcarea pieptului (optical flow, cu Pose ca rezervă) analizată pe ~10 s. Scor din amplitudine, ritm, regularitate și gura deschisă (`jaw_open` de la CV1) |
| `body/phone.py` | **Telefonul:** cutia YOLO corelată cu mâna, cu fața și cu privirea (`gaze_on_road` de la CV1). Folosire: în mână / se uită / apel |
| `body/slump.py` | **Prăbușirea / capul pe volan:** căderea capului față de umeri, aplecarea în față, nemișcarea |
| `events/tracker.py` | Transformă detecțiile de la fiecare cadru în **evenimente cu durată și încredere**, de exemplu `hand_on_chest`, 4 s, 0.8 |
| `pipeline.py` | Bucla CV2: cadru → percepție → analizori → tracker → mesaj `vigil/vision` |
| `publisher.py` | Trimite pe MQTT. Fără broker, afișează în consolă |
| `debug_view.py` | Fereastra de debug a CV2 (`python -m vigil_vision --show`) |
| `fake/fake_vision.py` | Date false cu scenariile din demo |

### Filtrele pentru mâna pe piept (`body/chest.py`)

O mână e „pe piept” doar dacă trece de **toate** filtrele:

1. **Nu e la față:** degetele sau palma sunt departe de bărbie și gură. Bărbia vine de la CV1; fără ea, e estimată din Pose.
2. **Zona** e calculată din umeri (`zone_top`, `zone_bottom`, `view_forward`, `view_back`).
3. **Toată mâna e în zonă:** cel puțin 60% din cele 21 de puncte, iar degetele nu trec deasupra umerilor.
4. **Mâna stă:** nu doar trece prin zonă (mișcare mică în ultimele 0.5 s).
5. **Adâncimea:** încheietura nu e mult în fața umerilor. **Oprit implicit**, pentru că valoarea din Pose e prea zgomotoasă.
6. **Nu e lângă cameră:** o mână ținută aproape de cameră pare mare. Dacă palma împărțită la mărimea corpului depășește `max_hand_scale`, mâna e respinsă.

Pose (încheietura + arătătorul) e folosit **doar ca rezervă**, când Hands nu vede nicio mână. Punctele lui din afara imaginii sunt ignorate.

### Avertizările din corp

| Nivel | Avertizare | Când |
|---|---|---|
| 3 | Mână pe piept + respirație grea | ≥ 2 s |
| 3 | Prăbușit / capul pe volan | ≥ 2 s |
| 2 | Mână pe piept | ≥ 2 s |
| 2 | Respirație grea | ≥ 3 s |
| 2 | Telefon + privire în afara drumului | ≥ 1 s |
| 1 | Telefon în mână (privirea la drum) | ≥ 1 s |

---

## 5. Cum comunică CV1, CV2 și ML1

**CV1 → CV2:** pipeline-ul CV2 caută funcția `vigil_vision.face.create(cfg)`, care întoarce `FaceStack`. Pentru fiecare cadru, `FaceStack.process(frame)` dă un `FaceOutput` cu:
- **semnale:** `ear`, `perclos`, `blink_rate`, `gaze_on_road`, `head_drop`, `jaw_open`, `chin`, `mouth`, `consciousness` etc.;
- **detecții:** `eyes_closed`, `microsleep`, `gaze_off_road`, `head_drop`, `unconscious`, `yawn`, `hyper_blinking`, `low_blink_rate`.

CV2 folosește trei dintre aceste semnale:
- `jaw_open`, pentru respirație;
- `gaze_on_road`, pentru telefon;
- `chin` și `mouth`, pentru filtrul de mână la față.

**CV1 + CV2 → ML1:** mesajul JSON de pe topicul MQTT **`vigil/vision`** e definit în `schemas.py`, după contractul din `TASKS.md`:

```json
{"ear": 0.27, "perclos": 0.05, "blink_rate": 17, "gaze_on_road": true, "head_drop": false,
 "phone": 0.0, "breathing": {"state": "normal", "rate_bpm": 14},
 "posture": {"state": "upright"},
 "events": [{"event": "hand_on_chest", "duration_sec": 4.0, "confidence": 0.8}],
 "landmarks": [[0.51, 0.42], ...], "camera_on": true, "face_detected": true, "body_detected": true}
```

Mesajul se trimite cu `python -m vigil_vision` sau cu `python -m vigil_vision.face.monitor --publish`, de `publish_hz` ori pe secundă (implicit 10).

Atenție la nume: ML1 caută evenimentul `phone_in_hand`, dar CV2 trimite `phone_use`. Numele trebuie aliniat între cele două părți.

---

## 6. Configurare și reglaj

- **`config.yaml`** conține setările pentru cameră, MQTT, modele, unghiul camerei (`view.mode`: `front` / `three_quarter` / `side`), YOLO, zona pieptului și filtrele ei, respirația, postura și evenimentele. Valorile implicite sunt în `vigil_vision/config.py`, așa că în fișier pui doar ce vrei să schimbi.
- **Pragurile CV1** sunt în clasele de configurare din `face/eyes.py`, `face/head_pose.py`, `face/yawn.py` și `face/unconscious.py`.

### Reglaje frecvente

| Problemă | Ce schimbi |
|---|---|
| „Ochi închiși” când zâmbești sau mijești | `EyeConfig.close_on` ↑, `smile_close_on` ↑ |
| Nu prinde privitul la telefon (doar cu ochii) | `HeadConfig.eye_down_limit` ↓ |
| Mâna pe piept respinsă „lângă cameră” | `chest.max_hand_scale` ↑ (uită-te la „Mărime mână” pe panou) |
| Mâna în fața camerei acceptată | `chest.max_hand_scale` ↓ |
| Mâna pe piept respinsă „la față” | `chest.face_radius` ↓ |
| Chenarul pieptului prea mic sau prea mare | `chest.view_forward`, `view_back`, `zone_top`, `zone_bottom` |
| FPS prea mic | `yolo.every_n_frames` ↑, `yolo.imgsz` ↓ sau `--no-body` |
| Oboseala dispare prea încet | `EyeConfig.recovery_window_sec` ↓ |
| Alt sunet | dicționarul `SOUNDS` din `face/monitor.py` |

---

## 7. Ce lipsește încă

- `dashboard/`: dashboard-ul web (FastAPI) e doar schelet; deocamdată monitorul are rolul de dashboard local.
- Analiza respirației nu ar trebui să ruleze când o mână acoperă pieptul.
- Numele pentru telefon trebuie aliniat între CV2 și ML1 (`phone_use` / `phone_in_hand`).
