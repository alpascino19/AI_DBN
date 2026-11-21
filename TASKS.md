# Vigil — Împărțirea taskurilor (4 persoane: 2 CV, 2 ML)

**Focus:** camera detectează → LLM-ul decide și **ia acțiuni** pe baza informațiilor din cameră.
Controlul vehiculului (MPC, `highway-env`) e **opțional**, doar dacă rămâne timp.

```
Cameră → CV (față + corp) → fuziune / scor de atenție → LLM (decide) → acțiuni (voce, alarmă, ESP32, 112)
```

## CV 1 — Fața și ochii
- [ ] MediaPipe Face Mesh: EAR, PERCLOS, rata de clipire (prea rară = somnolență, prea deasă = clipit anormal)
- [ ] Poziția capului (pitch, yaw, roll) și direcția privirii → `gaze_on_road`
- [ ] Căderea capului + ochi închiși prelungit (pierderea cunoștinței)
- [ ] Calibrarea din primele 30 s (praguri adaptate șoferului)
- [ ] **Dashboard** (FastAPI + web / Streamlit): scheletul feței, scorul de atenție, contribuția semnalelor, evenimentele active, ultima decizie a LLM-ului și motivul ei

## CV 2 — Corp, mâini și obiecte
- [ ] **Bucla camerei**: un singur proces deschide camera și dă cadrul ambelor module
- [ ] MediaPipe Pose + Hands: mâna pe piept (încheietura în zona toracelui, câteva secunde)
- [ ] Respirație grea: oscilația umerilor/pieptului + gura deschisă
- [ ] YOLO clasa „cell phone” + corelare cu mâna și privirea
- [ ] Transformă detecțiile în **evenimente cu durată și încredere** (ex. `hand_on_chest`, 4 s, 0.8) și le publică pe `vigil/vision`

## ML 1 — Motorul de decizie LLM
- [x] Fuziune: semnale → scor `a` ∈ [0, 1] (medie ponderată, netezită); fără cameră → încredere mai mică — `ml1/fusion.py`, date de test în `tests/data/`
- [x] LLM local pentru controllerul din mașină: `qwen2.5:1.5b` (Apache 2.0, ~1 GB), rezervă `llama3.2:1b`, structured output cu meniu de acțiuni — `ml1/llm_client.py`, `ml1/actions.py` (vezi `ml1/README.md`)
  - [x] Ollama instalat, `qwen2.5:1.5b` rulat pe laptop (RTX 4050, 100% GPU, 1.1 GB): p50 364 ms, p95 448 ms, 100% JSON valid, 100% severitate corectă pe cele 5 contexte (`python -m ml1.bench`)
  - Scenariile complete cu modelul real au scos 3 greșeli (alertă inventată, reminder greșit, fără întrebare vocală) → gardurile din `validate.apply_safety`, cu teste de regresie
- [x] Prompt de sistem cu regulile de severitate și 5 exemple (prefix fix → KV-cache) — `ml1/prompt.py`
- [x] Raționament contextual: durate + combinații + răspunsul șoferului; engine apelează LLM-ul doar la schimbare, cale rapidă din reguli la severitate 3 — `ml1/engine.py`
- [x] Validarea răspunsului LLM: acțiune necunoscută sau JSON invalid → reguli fixe de rezervă (plan B și pentru latență) — `ml1/validate.py`, `ml1/rules.py`, `ml1/tools.py`
- [x] Textul raportului 112 (semne observate, durata, locație simulată) și al raportului post-cursă — `ml1/reports.py` (șablon, nu LLM)
- [x] Integrare cu datele reale CV (`Vision/`): `ml1/cv_adapter.py` (phone_use, head_on_wheel, breathing/posture, fața lipsă), `ml1/mqtt_bridge.py` (vigil/vision → vigil/decision)
- [x] Calibrare personală (30 s, mediană/IQR) — `ml1/calibration.py`; evaluare după protocolul din C5 — `ml1/evaluate.py`, `docs/EVALUATION_RESULTS.md`
- [x] Documentație — `docs/` (arhitectură, metode ML legate de C1–C6, evaluare, alegerea modelului, runbook)
- [x] Aliniere la codul real CV1/CV2 (PERCLOS pe 60 s, ochi închiși 1.5 / 3 s, artefacte căscat / privit în jos, tracker cu pauze) — `ml1/profile.py`, `docs/CV_INTEGRATION.md`; simulatorul de evaluare rescris în aceeași semantică
- [x] Pregătire pentru serverul Spring Boot + camera live: `ml1/service.py`, `ml1/http_api.py` (REST + SSE), camelCase / ms / ISO / plicuri / duplicate / mesaje vechi — `docs/API.md`
- [ ] De la CV (opțional): publicat `perclos_recent` / `recovered`, `closed_sec`, `off_road_ratio`, `direction`; alertele locale CV pe mut în demo
- [ ] De la ML2: `vigil/driver_response` după fiecare întrebare (altfel 8 s de tăcere = fără răspuns)
- [x] **Owner pitch** — `PITCH.md` (script de 3 min + întrebări probabile; de repetat cu echipa)

## ML 2 — Executorul de acțiuni, vocea și hardware-ul
- [ ] Uneltele pe care le apelează LLM-ul: `speak(text)`, `play_alarm(level)`, `ask_driver(question)`, `set_esp32(level)`, `call_112(report)`, `suggest_break()`
- [ ] TTS pentru mesaje (`pyttsx3`) și STT pentru răspunsul șoferului (Whisper)
- [ ] Verificarea vocală: întreabă „Sunteți bine?”, măsoară timpul și coerența răspunsului, trimite rezultatul înapoi la LLM
- [ ] ESP32 + MQTT: LED/buzzer după nivelul de alertă
- [ ] Apel 112 **simulat**: afișează raportul generat pe dashboard + avarii simulate
- [ ] *Opțional:* controlul vehiculului (vezi mai jos)

## Severitate → acțiuni (decise de LLM)
| Severitate | Exemple din cameră | Acțiuni |
|---|---|---|
| 1. Redusă | privire în afara drumului, telefon câteva secunde | memento vocal blând („Vă rugăm să păstrați privirea la drum”) |
| 2. Medie | PERCLOS mare, clipit rar, căscat, micro-somn | alarmă mai puternică, sugerează pauză, scurtă conversație vocală |
| 3. Ridicată | mână pe piept + respirație grea, cap căzut, ochi închiși + fără răspuns vocal | alarmă urgentă, buzzer ESP32, apel 112 simulat cu raport, avarii |

## Contracte (stabilite în prima oră)
Comunicare prin brokerul MQTT local (oricum necesar pentru ESP32).

| Topic | De la → la | Conținut |
|---|---|---|
| `vigil/vision` | CV 1+2 → ML 1, dashboard | `{"ear":0.21,"perclos":0.3,"blink_rate":28,"gaze_on_road":false,"head_drop":false,"phone":0.1,"events":[{"event":"hand_on_chest","duration_sec":4,"confidence":0.8,"breathing":"heavy"}],"landmarks":[...]}` |
| `vigil/decision` | ML 1 → ML 2, dashboard | `{"a":0.2,"severity":3,"actions":[{"tool":"play_alarm","args":{"level":3}},{"tool":"ask_driver","args":{"question":"Sunteți bine?"}}],"reason":"mână pe piept 4 s + respirație grea"}` |
| `vigil/driver_response` | ML 2 → ML 1 | `{"answered":false,"latency_sec":null,"text":null}` |
| `vigil/alert` | ML 2 → ESP32 | `3` |

## Etape
1. **Prima oră:** contractele de mai sus + fiecare scrie un generator de date false pentru topicul lui.
2. **Prima jumătate:** totul merge cap-coadă pe date false (cameră → LLM → acțiuni → dashboard).
3. **A doua jumătate:** date reale în locul celor false, reglarea pragurilor la lumina din sală, ajustarea promptului.
4. **Ultimele ore:** fără funcții noi, repetăm demo-ul de 3 minute.

## Scenariul de demo și owneri
| Pas | Ce se vede | Owner |
|---|---|---|
| Calibrare (30 s) | scorul se stabilizează | CV 1 |
| Distragere / telefon | memento vocal blând | CV 2 → ML 1 → ML 2 |
| Somnolență | alarmă + „Faceți o pauză?” + conversație scurtă | CV 1 → ML 1 → ML 2 |
| Mână pe piept + respirație grea | LLM-ul întreabă „Sunteți bine?” | CV 2 → ML 1 → ML 2 |
| Cap căzut, fără răspuns | alarmă urgentă, buzzer, apel 112 simulat cu raport | toți |
| Privacy mode (camera oprită) | doar scheletul pe dashboard, încredere scăzută | CV 1 + ML 1 |

---

## Opțional — Controlul vehiculului (doar dacă rămâne timp, ML 2)
- [ ] `highway-env` + controler P simplu
- [ ] MPC `cvxpy` + OSQP cu scorul `a` în funcția de cost
- [ ] Manevra de risc minim (bandă de urgență, viteză → 0) declanșată de severitatea 3
