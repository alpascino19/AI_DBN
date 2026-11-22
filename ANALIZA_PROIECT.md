# Vigil — analiza proiectului

> Analiză făcută pe codul din `AI_DBN/` (repo `alpascino19/AI_DBN`, ultimul commit: „Voice assistent button finished”)
> și pe documentația existentă (`README.md`, `CLAUDE.md`, `TASKS.md`, `docs/`, `ML1/docs/`, `ML2/README.md`,
> `Vision/COMPUTER_VISION.md`, `START/README.md`, `dashboard/README.md`).
> Scopul: o imagine de ansamblu — ce face sistemul, cum e construit, ce alegeri s-au făcut și de ce, ce lipsește.

---

## 1. Pe scurt

**Vigil** este un sistem de monitorizare a șoferului în timp real. O cameră web urmărește fața și corpul șoferului,
modulul de computer vision transformă imaginea în semnale numerice (EAR, PERCLOS, privire, telefon, mână pe piept,
respirație, postură), un motor de decizie combină semnalele și — cu ajutorul unui **LLM local** (Qwen2.5 1.5B prin
Ollama) — decide o **severitate 0–3** și o listă de **acțiuni**. Acțiunile sunt executate de un modul separat: voce în
română, alarmă, conversație cu șoferul, LED/buzzer pe un ESP32, apel 112 **simulat**. Un dashboard React afișează
totul live, iar un controler fuzzy produce o comandă de viteză/asistență de bandă pentru un „ECU” simulat.

Proiect de hackathon, 4 persoane (2 CV, 2 ML), 24–48 h. Totul rulează local pe laptop, modulele comunică prin
**MQTT** (Mosquitto).

```
Cameră ─► Vision (CV1 față + CV2 corp) ──vigil/vision──► ML1 (decide) ──vigil/decision──► ML2 (execută) ──vigil/alert──► ESP32
                                                          ▲   │                              │
                                                          │   └──vigil/ecu──► ECU (fuzzy)    │
                                                          └──────vigil/driver_response───────┘
                         dashboard React (browser) ◄── toate topicurile, prin WebSocket (port 9001)
```

---

## 2. Structura repo-ului

| Folder | Echipa | Limbaj / tehnologie | Rol |
|---|---|---|---|
| `Vision/` | CV1 + CV2 | Python, MediaPipe, YOLOv8n (ultralytics), OpenCV | camera → semnale → evenimente → `vigil/vision` |
| `ML1/` | ML1 | Python (stdlib + numpy), Ollama | fuziune, reguli de siguranță, LLM, HMM, fuzzy ECU, API HTTP |
| `ML2/` | ML2 | Python, edge-tts/Piper/pyttsx3, faster-whisper, Silero VAD, Ollama | executorul de acțiuni, voce, conversație, asistentul Virgil |
| `ML2/esp32/vigil_alert/` | ML2 | Arduino C++ (ESP32-C3, PubSubClient) | LED/buzzer după nivelul de alertă |
| `ECU_Arduino/` | — | Arduino C++ (ESP32-C3, ArduinoJson) | mașină simulată: frână + avarii prin MQTT |
| `dashboard/` | ML2 | React 19 + Vite 8 + TypeScript + Tailwind 4 + mqtt.js | interfața live, fără backend |
| `START/` | toți | `.bat` + `run_vigil.py` | instalare, pornirea întregului sistem, teste |
| `docs/` | — | Markdown | conceptul inițial, propunerea cu două camere, plan/spec dashboard |

Un singur mediu Python (`.venv` la rădăcină) cu `requirements.txt` care include cerințele celor trei module.

---

## 3. Vision (CV1 + CV2)

### Arhitectură
```
camera (thread, doar ultimul cadru) ─► perception (Face / Pose / Hands / YOLO) ─► FrameContext
      ─► face/* (CV1) ┐
      ─► body/* (CV2) ┴─► events/tracker (durată + încredere, histerezis) ─► publisher ─► vigil/vision
```

**Alegeri de design importante:**
- **Fiecare model rulează o singură dată pe cadru**; rezultatele stau în `FrameContext`, din care citesc toți
  analizorii. CV1 și CV2 nu se importă reciproc — schimbă date doar prin context (decuplare între cele două echipe).
- **Un singur proces deschide camera** (`camera.py`), un thread care păstrează doar ultimul cadru → fără întârziere
  acumulată când procesarea e mai lentă decât camera.
- **Pragurile sunt în secunde** (`time.monotonic()`), nu în cadre, pentru că FPS-ul variază.
- **YOLO rulează la fiecare 5 cadre** (`imgsz=320`), cu memorie de 1 s pentru ultima cutie — compromis viteză/acuratețe.
- **Privacy**: niciun cadru nu iese din pipeline; pe MQTT pleacă doar cifre și un subset de landmark-uri.
- Mesajele se publică la ~10 Hz, nu la fiecare cadru.

### CV1 — fața
| Fișier | Ce calculează |
|---|---|
| `perception/face.py` | MediaPipe FaceLandmarker: 478 puncte, blendshapes (`eyeBlink*`, `jawOpen`, `eyeLookDown*`), poziția capului |
| `face/eyes.py` | EAR + scor de închidere (60% EAR raportat la ochii șoferului + 40% `eyeBlink`), compensare la zâmbet; clipiri, **PERCLOS pe 60 s**, rata de clipire, revenire pe 15 s |
| `face/head_pose.py` | yaw/pitch/roll, iris, privire în jos → `gaze_on_road`, direcția, % din ultimele 10 s în afara drumului |
| `face/yawn.py` | căscat (MAR / `jawOpen` 1.5–8 s), număr de căscate în 5 min |
| `face/unconscious.py` | scor 0–1 din cap căzut + ochi închiși + nemișcare + cădere bruscă → normal / suspect / inconștient |
| `face/calibration.py` | calibrare în primele N s (implicit 30): EAR deschis, poziția neutră a capului și a privirii |
| `face/alerts.py`, `face/monitor.py` | avertizări locale pe 3 niveluri (cu sunete) și aplicația live cu fereastră |

### CV2 — corpul
| Fișier | Ce calculează |
|---|---|
| `perception/body.py` | MediaPipe Pose (33 puncte) + Hands (21 puncte / mână) |
| `body/geometry.py` | geometrie independentă de unghiul camerei (față / ~45° / lateral) |
| `body/chest.py` | **mâna pe piept**, cu 6 filtre anti-alarmă-falsă (nu e la față, zonă din umeri, ≥60% din mână în zonă, mâna stă, adâncime — oprită, nu e lângă cameră) |
| `body/breathing.py` | respirație grea: optical flow pe piept (Pose ca rezervă), ~10 s, amplitudine + ritm + regularitate + gura deschisă |
| `body/phone.py` | telefon: cutia YOLO corelată cu mâna, fața și privirea |
| `body/slump.py` | prăbușire / capul pe volan |
| `smoothing.py` | filtru One Euro pe punctele Pose |

### Două camere
`run_face.py` → `vigil/vision/face`, `run_body.py` → `vigil/vision/body`, iar `merge.py` le combină într-un singur
`vigil/vision` la 10 Hz (de la față: ochi/privire; de la corp: telefon/respirație/postură; evenimentele amândurora).
O cameră tăcută > 1 s → câmpurile ei lipsesc, ML1 scade încrederea. A rezolvat două probleme reale: `client_id` MQTT
identic (laptopurile se deconectau reciproc) și fluxuri amestecate pe același topic (scorul oscila).

---

## 4. ML1 — motorul de decizie

### Fluxul pe cadru
```
vigil/vision ─┐
              ├─► 1 normalizare (cv_adapter) ─► 2 calibrare (30 s, mediană/IQR) ─► 3 fuziune (scor a, încredere)
driver_resp ──┘        ─► 4 reguli (calea de siguranță, sev 0–3) ─┬─► vigil/decision
                                                                   └─► 5 Qwen2.5 1.5B (doar la schimbare) ─► 6 gărzi ─┘
        reguli + a ─► 7 matricea de probabilitate (HMM) ─► 8 fuzzy Mamdani + supervizor ─► vigil/ecu (5 Hz)
```

| Pas | Fișier | Ce face |
|---|---|---|
| 1 | `cv_adapter.py` | aduce mesajul CV la forma internă (`phone_use`→`phone_in_hand`, `breathing.state`, `posture.state`), decupează la intervale fizice, valori ne-finite → `None`, aruncă landmark-urile |
| 2 | `calibration.py` | baseline personal din primele 30 s, doar cadre „curate”; nu acceptă un „normal” cu PERCLOS > 0.20 |
| 3 | `fusion.py` | sub-scoruri per semnal → medie ponderată renormalizată peste semnalele disponibile (ochi 0.30, privire/cap/telefon 0.15 etc.), **plafoane** (ochi închiși ⇒ a ≤ 0.35), **EMA asimetric** (scade în 0.7 s, revine în 4 s); încredere după surse (cameră 0.6, voce 0.2, condus 0.2) |
| 4 | `rules.py`, `profile.py` | scara de severitate deterministă (ex. ochi închiși ≥ 1.5 s → 2, ≥ 3 s continuu → 3; mână pe piept ≥ 3 s + respirație grea → 3) |
| 5 | `prompt.py`, `llm_client.py`, `actions.py` | LLM local alege severitatea și **ID-uri de acțiuni** dintr-un meniu fix |
| 6 | `validate.py` | gărzi: LLM-ul nu coboară sub reguli (max o treaptă, niciodată sub 3), **nu urcă peste reguli**, 112 doar după „Sunteți bine?” fără răspuns; timeout 2.5 s / JSON invalid → reguli |
| — | `engine.py` | orchestrare: LLM apelat doar la schimbarea situației (max 1/s), **cale rapidă la severitate 3** (nu așteaptă LLM-ul), histerezis la coborâre, decizii LLM întârziate aruncate, tăcere ≥ 8 s = „fără răspuns” |
| 7 | `probability.py` | filtru HMM (4 stări ascunse, 12 simboluri de observație), tranziții `T(dt)=exp(Q·dt)`, decizie Bayes cu cost asimetric (a rata o urgență = 30× o alarmă falsă); ~3 µs/cadru |
| 8 | `ecu_fuzzy.py` | Mamdani (min/max/centroid), 8 reguli; intrări `a`, risc, încredere → fracție de viteză, autoritate LKA, distanță; supervizor determinist + **MRM** (oprire pe banda de urgență); export în tabel C `uint8` 15×15×9 (6 KB) |
| — | `service.py`, `mqtt_bridge.py`, `http_api.py` | transporturi: MQTT și REST + SSE (port 8090) pentru un server Spring Boot |
| — | `reports.py` | raportul 112 și raportul post-cursă din **șablon**, nu din LLM |

### Cum e folosit LLM-ul (cea mai interesantă alegere)
LLM-ul este **consilier, nu decident final**. Optimizări pentru un model mic pe hardware auto:
- **Nu scrie text liber** — alege ID-uri (`remind_phone`, `alarm`, `ask_ok`, `call_112`…). Frazele în română sunt fixe
  și validate → zero halucinații în ce aude șoferul, ~25 tokeni de ieșire.
- **Structured output** (schemă JSON în `format=` la Ollama) → modelul nu poate ieși din meniu.
- **Prefix fix** (prompt de sistem + 5 exemple, identice la fiecare apel) → KV-cache refolosit, ~80 tokeni noi per apel.
- Intrarea e **o singură linie compactă de cifre**, nu JSON brut; instrucțiunile în engleză (modelele mici le urmează
  mai bine), motivul în română.
- `temperature=0`, `num_predict=48`, `keep_alive`, warmup.
- Măsurat pe RTX 4050: p50 364 ms, p95 448 ms, 100% JSON valid pe contextele de bază.

**De ce Qwen2.5 1.5B și nu Llama 3.2 1B:** licență Apache 2.0 (fără condiții), suport multilingv mai larg (româna nu e
oficială la Llama 3.2), măsurători proprii. Llama 3.2 a rămas rezervă configurabilă (`VIGIL_MODEL`), dar nu a fost testat.

**Greșeli reale ale modelului prinse de gărzi** (fiecare cu test de regresie): a inventat „telefon în mână”, a urcat
telefonul (1) la oboseală (2), a cerut „Doriți o pauză?” la o urgență, a omis întrebarea vocală la oboseală.

### Siguranță (ISO 26262 / SOTIF)
Separarea clară între **calea de siguranță deterministă** (reguli + gărzi, testabile) și **componenta ML fără rating
de siguranță** (LLM, QM). Ideile din SOTIF apar în calibrarea personală, persistența în timp (fața lipsă ≥ 1 s,
`a < 0.75` ≥ 4 s), încrederea scăzută fără cameră și evaluarea separată pe domeniul „shift”.

### Evaluare (`ML1/docs/EVALUATION_RESULTS.md`)
Șoferi simulați (20 train / 8 val / 10 test / 8 shift), împărțire **pe șofer**, IC 95% bootstrap, cost asimetric.

| Variantă (test) | macro-F1 | recall sev. 3 | alarme false / oră | 112 corecte · false |
|---|---|---|---|---|
| reguli, praguri vechi | 0.75 | 1.00 | 18.2 | 6/6 · 0 |
| reguli + calibrare | 0.83 | 1.00 | 13.6 | 6/6 · 0 |
| reguli + LLM (din `MODEL_CHOICE.md`) | 0.89 | — | 17 | 6/6 · 0 |
| HMM (cost minim) | 0.94 | 1.00 | — | — |

Pe domeniul „shift” (ochi mici, zgomot), regulile fără calibrare cad la macro-F1 0.58 și 62 alarme false/oră —
calibrarea personală e justificată de date. Modelele învățate (softmax, arbore) sunt ținute doar ca „model umbră”.

**Limita, recunoscută chiar în documentație:** toate cifrele vin de pe un simulator care generează și etichetele.
Compară variante între ele, nu prezic comportamentul pe șoferi reali.

Documentația leagă fiecare decizie de cursurile de ML (C1–C6: preprocesare, regresie, clasificare, evaluare,
ML clasic), iar `TECHNICAL_DEEP_DIVE.pdf` (18 pagini, LaTeX) urmărește un cadru prin tot sistemul.

---

## 5. ML2 — executorul de acțiuni

```
vigil/decision ─► validare decizie ─► pentru fiecare acțiune:
                    ├─ unealtă necunoscută / argumente greșite ─► aruncată + log
                    ├─ set_esp32 ─► publicată IMEDIAT pe vigil/alert
                    ├─ debounce (aceeași unealtă, aceeași severitate, 10 s) ─► aruncată
                    └─ PriorityQueue după severitate ─► UN SINGUR worker
```

- **Unelte**: `speak`, `play_alarm`, `ask_driver`, `set_esp32`, `call_112`, `suggest_break`.
- **Un singur worker** → vocea și alarma nu se suprapun. **Severitatea 3 întrerupe** acțiunea curentă în ~50 ms
  (`threading.Event`, inclusiv în timpul generării vocii) și golește coada de acțiunile mai puțin importante.
- **Validare strictă la intrare** — a doua barieră împotriva unui LLM care inventează unelte sau argumente
  (`True` nu trece drept număr, texte goale respinse).
- **TTS în cascadă**: edge-tts `ro-RO-EmilNeural` (online) → Piper `ro_RO-mihai-medium` (offline) → `pyttsx3`.
- **STT**: faster-whisper `small`, `language="ro"`, prompt cu vocabular, Silero VAD, filtru de volum (respinge vocile
  din fundal) și filtru de text „nesigur” — motivat explicit: un „a răspuns” fals ar ascunde un șofer inconștient.
- **`ask_driver`**: microfonul pornește după ce TTS-ul termină, 4 s ca șoferul să înceapă, max 3 replici; replicile
  le generează LLM-ul local; fiecare răspuns pleacă imediat pe `vigil/driver_response`.
- **`call_112` e doar simulat**: raport în terminal + fișier `reports/112_*.txt` + anunț vocal.
- **Asistentul Virgil**: „Hei Virgil” (wake word aproximativ), plus `info_assistant.py` / butoanele din dashboard
  pentru oră, vreme (Open-Meteo), trafic (Waze prin OpenWebNinja, TomTom), căutări (Tavily prin OmniRoute). Are
  prioritatea cea mai mică și tace la orice alertă.
- **ESP32 (`vigil_alert.ino`)**: abonat la `vigil/alert` (cu retain): 0 stins, 1 aprins, 2 clipește, 3 clipește rapid
  + buzzer; reconectare automată; putere de emisie redusă la 8,5 dBm (problemă cunoscută la ESP32-C3 SuperMini).

---

## 6. Dashboard (React)

- React 19 + Vite + TypeScript + Tailwind, **conectat direct la Mosquitto prin WebSocket** (port 9001) cu mqtt.js —
  fără backend, deci un proces mai puțin de pornit.
- `state.ts` transformă mesajele MQTT în starea paginii și ignoră mesajele stricate; `useVigil.ts` se reconectează la 2 s.
- Panouri: scorul de atenție, ferestrele „FAȚĂ” și „CORP” desenate din landmark-uri (fără imagine — privacy),
  evenimente active, decizia LLM + istoric, cardul ECU cu P(atent/distras/obosit/urgență), asistentul Virgil,
  răspunsurile șoferului, **fereastra roșie de urgență** cu raportul 112, buton „Reset alerte”.
- Teste cu vitest (`state`, `skeleton`, `panels`).

---

## 7. ECU / controlul vehiculului

Două piese, deocamdată **neconectate între ele**:
1. **`ML1/ml1/ecu_fuzzy.py`** publică pe `vigil/ecu` (5 Hz): mod `NORMAL | CAUTION | ASSIST | MRM`, viteză de
   referință, autoritate LKA, distanță, avarii, probabilitățile stărilor. Supervizorul pornește MRM după 112 sau după
   10 s de severitate 3 **confirmată de HMM** (decizie luată după ce severitatea publicată singură a produs un MRM fals
   în evaluare). Rezultate în simulare: 6/6 MRM în urgențe, 0 false.
2. **`ECU_Arduino/ECU_Arduino.ino`** (ESP32-C3): mașină simulată care ascultă `vigil/car/cmd`
   (`emergency`, `slow_down`, `hazard`, `resume`, `reset`), frânează lin (decelerare limitată 0.5–4 m/s²), aprinde
   avariile și publică `vigil/car/state` / `vigil/car/event`. Are și un mod MOCK pe serial.

Nu există încă buclă închisă (`highway-env`) și nici MPC — marcate opțional în `TASKS.md`.

---

## 8. Infrastructură și pornire

- **MQTT/Mosquitto** ca bus comun: decuplează echipele (fiecare modul se testează singur pe date false publicate pe
  topicul lui — `fake_vision.py`, `fake_llm.py`), și oricum era necesar pentru ESP32.
- **`START/run_vigil.py`** pornește totul în ordinea corectă (ML2 încarcă vocea → ML1 warmup → Vision → dashboard),
  cu loguri colorate per modul și `logs/<modul>.log`. Moduri: `--fake`, `--cameras two|remote`, `--input http`, `--check`.
- Fișiere `.bat` numerotate pentru utilizare cu dublu-clic (instalare, broker ca admin cu regulă de firewall, demo,
  cameră, 2 camere, PC central + 2 laptopuri cu camere, „cine e conectat”, teste).
- Topologia demo: 3 laptopuri pe un hotspot propriu (central: broker + ML1 + ML2 + dashboard + ESP32; două laptopuri
  doar cu camere).

### Contractele MQTT
| Topic | De la → la | Conținut |
|---|---|---|
| `vigil/vision` (+ `/face`, `/body`) | Vision → ML1, dashboard | semnale + evenimente + landmark-uri + panouri |
| `vigil/decision` | ML1 → ML2, dashboard | `{a, severity, actions[{tool,args}], reason, source, ts, …}` |
| `vigil/driver_response` | ML2 → ML1 | `{answered, latency_sec, text}` |
| `vigil/ml1/status` | ML1 → dashboard | scor, încredere, calibrare, statistici (1 Hz) |
| `vigil/ecu` | ML1 → ECU, dashboard | comanda fuzzy (5 Hz) |
| `vigil/alert` | ML2 → ESP32 | `0`–`3`, retain |
| `vigil/assistant`, `/reply` | dashboard ↔ ML2 | butoanele asistentului |
| `vigil/car/cmd`, `/state`, `/event` | ? → ECU_Arduino → | mașina simulată (vezi observațiile) |

### Teste
ML1 ~137 teste (`run_ml1.py test`, fără Ollama și fără broker; include compilarea header-ului C cu `gcc -Werror`),
Vision (unittest/pytest pe date sintetice), ML2 (`test_executor.py` 13, `test_voice.py` 4, `test_info*.py`),
dashboard (vitest). Toate pornesc din `START\6_teste.bat`.

---

## 9. Alegeri de arhitectură — rezumat

| Alegere | Alternativa | De ce |
|---|---|---|
| MQTT între module | apeluri directe / HTTP | decuplare între echipe, date false per topic, ESP32 vorbește nativ MQTT, dashboard prin WebSocket |
| Reguli deterministe ca cale de siguranță, LLM ca consilier | doar LLM / doar reguli | testabilitate și explicabilitate (ISO 26262) + nuanță contextuală |
| LLM mic local (1.5B) | model mare în cloud | latență sub 0.5 s, fără internet, fără date despre șofer în afara mașinii, ținta = controller de cockpit |
| Meniu de acțiuni cu fraze fixe | text generat liber | zero halucinații vocale, ieșire scurtă |
| LLM apelat doar la schimbare + cale rapidă la sev. 3 | apel la fiecare cadru | latență, cost, siguranța nu așteaptă modelul |
| Calibrare personală (mediană/IQR) | praguri globale | alarme false pe „shift”: 21 → 0/oră în evaluare |
| HMM + decizie Bayes pentru ECU | severitatea publicată direct | severitatea are histerezis; a frâna mașina cere o probabilitate, nu o decizie |
| Fuzzy Mamdani compilat în tabel C | MPC / LLM pe ECU | rulează pe orice microcontroler, timp mărginit, reguli citibile |
| Un singur worker + PriorityQueue în ML2 | acțiuni în paralel | vocea și alarma nu se suprapun, urgența întrerupe |
| Dashboard fără backend | FastAPI/Streamlit (planul inițial din TASKS) | un proces mai puțin, mqtt.js direct în browser |
| Thread de cameră cu „ultimul cadru” | coadă de cadre | fără latență acumulată |

---

## 10. Observații și probleme găsite

**De rezolvat înainte de a face repo-ul vizibil altora:**
1. **Parola WiFi e în git.** `ECU_Arduino/ECU_Arduino.ino` are SSID-ul și parola hotspotului scrise direct în cod
   (`WIFI_SSID`, `WIFI_PASSWORD`), iar fișierul e urmărit de git. Repo-ul e public (comentariul din `.gitignore` o
   spune). ML2 a rezolvat deja asta cu `wifi_secrets.h` ignorat — același model ar trebui folosit și aici, iar
   parola hotspotului schimbată (rămâne în istoricul git).

**Integrare:**
2. **`ECU_Arduino` nu primește comenzi de la nimeni.** Ascultă `vigil/car/cmd`, dar niciun modul Python și nici
   dashboard-ul nu publică pe acest topic; ML1 publică comanda ECU pe `vigil/ecu`, cu alt format. Lipsește o punte
   (`vigil/ecu` mode `MRM` → `{"cmd":"emergency"}`), iar `vigil/car/*` nu e în tabelul de contracte din `CLAUDE.md`.
3. **Două sketch-uri ESP32 pe același topic și același pin.** `vigil_alert.ino` și `ECU_Arduino.ino` ascultă amândouă
   `vigil/alert` și folosesc LED-ul de pe GPIO 8. Trebuie clarificat care rulează la demo (sau dacă sunt două plăci).
4. **IP-ul brokerului e scris în mai multe locuri** (`Vision/config.yaml`, `ECU_Arduino.ino`, `vigil_alert.ino`,
   `START/3_LAPTOPURI.md`): la un alt hotspot trebuie schimbat peste tot.

**Documentație / curățenie:**
5. **`TASKS.md` nu reflectă starea CV.** Toate punctele CV1/CV2 sunt nebifate, deși codul din `Vision/` le
   implementează.
6. **`Vision/dashboard/` e un rest.** `server.py` (63 B) și `index.html` (65 B) sunt goale, iar `Vision/README.md`
   încă le descrie ca dashboard FastAPI — înlocuit de dashboard-ul React.
7. Fișierele `.idea/` (PyCharm) din `START/` și `Vision/` sunt în git.

**Afirmații de nuanțat la pitch:**
8. **„Totul local” are excepții:** edge-tts trimite textul rostit la Microsoft, iar asistentul de informare folosește
   servicii cloud (Gemini prin OmniRoute, Tavily, Waze). Partea de siguranță e locală; asistentul nu.
9. **Rezultatele sunt pe date simulate** — documentația o spune clar, dar cifrele (F1 0.89, 0 urgențe ratate) nu
   trebuie prezentate ca performanță pe șoferi reali. Câștigul LLM față de reguli are intervale de încredere care se
   suprapun.

**Ce nu e făcut (opțional în plan):** `highway-env` în buclă închisă, MPC, butoanele Calibrare / Privacy Mode în
dashboard, imaginile camerelor în dashboard, date reale etichetate (KSS + adnotări) pentru re-învățarea pragurilor
și a matricelor HMM.

---

## 11. Unde citești mai departe

| Subiect | Document |
|---|---|
| Pornire, probleme frecvente | `START/README.md`, `START/3_LAPTOPURI.md` |
| Viziunea inițială, cadrul legal | `docs/CONCEPT.md` |
| CV în detaliu | `Vision/COMPUTER_VISION.md` |
| ML1: arhitectură, siguranță, latență | `ML1/docs/ARCHITECTURE.md` |
| ML1: înțelegere tehnică completă | `ML1/docs/TECHNICAL_DEEP_DIVE.pdf` |
| Metode ML ↔ cursuri | `ML1/docs/ML_METHODS.md` |
| Evaluare | `ML1/docs/EVALUATION.md`, `EVALUATION_RESULTS.md` |
| HMM + fuzzy ECU | `ML1/docs/PROBABILITY_ECU.md` |
| Alegerea modelului | `ML1/docs/MODEL_CHOICE.md` |
| API Spring Boot | `ML1/docs/API.md` |
| Executor, voce, ESP32 | `ML2/README.md` |
| Dashboard | `dashboard/README.md` |
| Pitch | `ML1/PITCH.md` |
