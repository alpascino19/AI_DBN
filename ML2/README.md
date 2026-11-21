# ML 2 — Executorul de acțiuni Vigil

ML 2 e „mâinile și vocea” proiectului. ML 1 (LLM-ul care decide) publică pe MQTT **ce** trebuie făcut, iar ML 2 **face**: vorbește cu șoferul, sună alarma, aprinde LED-ul de pe ESP32, poartă o conversație scurtă și simulează apelul la 112.

```
CV ──vigil/vision──► ML 1 (decide) ──vigil/decision──► ML 2 (execută) ──vigil/alert──► ESP32 (LED / buzzer)
                         ▲                                  │
                         └───────vigil/driver_response──────┘   (ce a răspuns șoferul)
```

Totul rulează pe laptop. Singurele excepții: vocea Emil (edge-tts trimite textul rostit la Microsoft) și descărcarea modelelor, o singură dată.

---

## Cuprins
1. [Structura folderului](#1-structura-folderului)
2. [Instalare](#2-instalare)
3. [Pornire rapidă](#3-pornire-rapidă)
4. [Fiecare program în detaliu](#4-fiecare-program-în-detaliu)
5. [Contractele MQTT](#5-contractele-mqtt)
6. [Reglaje](#6-reglaje)
7. [Testare înainte de demo](#7-testare-înainte-de-demo)
8. [Probleme cunoscute și depanare](#8-probleme-cunoscute-și-depanare)
9. [Ce nu e făcut](#9-ce-nu-e-făcut)

---

## 1. Structura folderului

| Fișier | Ce este | Se rulează direct? |
|---|---|---|
| `executor.py` | **Programul principal**: ascultă deciziile lui ML 1 și le execută | da |
| `actions.py` | Uneltele: voce, alarmă, conversație, 112 simulat, asistentul Virgil | nu (îl folosesc ceilalți) |
| `fake_llm.py` | Trimite decizii false, ca să testezi fără ML 1 | da |
| `voice_test.py` | Asistentul vocal **Virgil** la microfon („Hei Virgil”), fără ML 1 | da |
| `esp32_test.py` | Test cap-coadă: laptop → broker → ESP32 → confirmare pe serial | da |
| `test_executor.py` | Teste automate pentru executor și conversație (fără hardware) | da |
| `test_voice.py` | Teste automate pentru recunoașterea vocală (fără microfon) | da |
| `esp32/vigil_alert/vigil_alert.ino` | Programul plăcii ESP32 | se flashuiește |
| `esp32/vigil_alert/wifi_secrets.h.example` | Model pentru datele WiFi ale plăcii | se copiază |
| `mosquitto.conf` | Configul brokerului MQTT (ascultă în rețea) | se copiază |
| `requirements.txt` | Pachetele Python | `pip install -r` |

Fișiere create la rulare, **care nu se urcă pe git și nu se dau mai departe**:

| Fișier | Conține |
|---|---|
| `esp32/vigil_alert/wifi_secrets.h` | **parola WiFi** |
| `recordings/*.wav` | înregistrări ale vocii celui care a testat |
| `voice_live.json` | transcrierile ultimului test vocal |
| `reports/112_*.txt` | rapoartele apelurilor 112 simulate |
| `voices/` | vocea Piper descărcată (63 MB) |

---

## 2. Instalare

Testat pe **Windows 11** cu **Python 3.14.6**. Comenzile se dau în PowerShell.

### 2.1 Python
Din rădăcina repo-ului (`AI_DBN-main`):
```powershell
python -m venv .venv
.venv\Scripts\pip install -r ML2\requirements.txt
```

### 2.2 Programe externe

| Program | Instalare | Pentru ce |
|---|---|---|
| **Mosquitto** (broker MQTT) | `winget install -e --id EclipseFoundation.Mosquitto` | comunicarea dintre module |
| **Ollama** (LLM local) | `winget install -e --id Ollama.Ollama`, apoi `ollama pull gemma3:4b` (3,4 GB) | replicile conversației |
| **ffmpeg** | `winget install -e --id Gyan.FFmpeg` | decodează vocea edge-tts |
| **arduino-cli** | `winget install -e --id ArduinoSA.CLI` | compilează și flashuiește ESP32 |

Vocea Piper, de rezervă, fără internet:
```powershell
.venv\Scripts\python -m piper.download_voices ro_RO-mihai-medium --download-dir ML2\voices
```
Modelul Whisper (`small`, ~500 MB) se descarcă singur la prima rulare.

### 2.3 Brokerul Mosquitto să asculte în rețea
Implicit, Mosquitto 2.x acceptă conexiuni doar de pe laptop, deci ESP32 nu se poate conecta. Installerul pornește și un **serviciu Windows** cu configul implicit. Dacă pornești încă un broker, ai **două brokere**: executorul vorbește cu unul, ESP32 cu celălalt, și nu se aud.

Soluția: serviciul folosește configul nostru. Rulează **PowerShell ca administrator**:
```powershell
$d = 'C:\Program Files\mosquitto'
Copy-Item "$d\mosquitto.conf" "$d\mosquitto.conf.orig"          # păstrează originalul
Copy-Item 'CALEA\ML2\mosquitto.conf' "$d\mosquitto.conf" -Force
Restart-Service mosquitto
```
Verificare: `netstat -ano | findstr :1883` trebuie să arate **o singură** linie `0.0.0.0:1883 ... LISTENING`. Brokerul pornește apoi singur la fiecare boot.

### 2.4 Placa ESP32
Placa folosită e un **ESP32-C3** (de exemplu SuperMini). Pașii:
1. Instalează core-ul ESP32 și librăria MQTT:
   ```powershell
   arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
   arduino-cli core update-index
   arduino-cli core install esp32:esp32
   arduino-cli lib install PubSubClient
   ```
2. Copiază `wifi_secrets.h.example` în `wifi_secrets.h` și completează rețeaua și parola. **ESP32 merge doar pe WiFi 2.4 GHz**: la hotspotul de pe telefon, setează banda pe 2.4 GHz, apoi oprește și repornește hotspotul.
3. În `vigil_alert.ino`, pune la `BROKER_IP` IP-ul laptopului: `ipconfig`, la adaptorul Wi-Fi.
4. Flash (portul îl vezi cu `arduino-cli board list`):
   ```powershell
   arduino-cli compile --fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc -u -p COM13 ML2\esp32\vigil_alert
   ```
   Pentru un ESP32 clasic, nu C3: `--fqbn esp32:esp32:esp32`, iar în sketch `LED_PIN 2` și `LED_ON HIGH`.

---

## 3. Pornire rapidă

Toate comenzile se dau din `ML2`:
```powershell
cd ML2
```

**Demo-ul complet, fără ML 1.** Terminalul 1:
```powershell
..\.venv\Scripts\python executor.py
```
Așteaptă mesajul `modelele de voce sunt încărcate` (10–50 s), apoi rulează în terminalul 2:
```powershell
..\.venv\Scripts\python fake_llm.py distragere oboseala medical fara_raspuns reset -d 15
```

**Cu ML 1 real:** pornești doar `executor.py`. ML 1 publică pe `vigil/decision`.

**Asistentul vocal Virgil:**
```powershell
..\.venv\Scripts\python voice_test.py
```

---

## 4. Fiecare program în detaliu

### 4.1 `executor.py` — programul principal

**Rol:** primește deciziile lui ML 1 de pe `vigil/decision` și execută acțiunile din ele, în ordinea importanței, fără să le amestece.

**Ce se întâmplă la pornire:**
1. Se conectează la brokerul MQTT. Adresa e `localhost`, sau ce e setat în variabila de mediu `VIGIL_BROKER`.
2. Se abonează la `vigil/decision`.
3. Trimite `0` pe `vigil/alert`, ca ESP32 să nu rămână pe o stare veche. Brokerul ține minte ultimul nivel trimis.
4. Încarcă în fundal modelele de voce (Piper, Whisper, LLM-ul). Altfel, prima întrebare către șofer ar întârzia ~15–50 s.
5. Intră în bucla worker-ului, care rulează câte o acțiune pe rând.

**Ce se întâmplă când vine o decizie:**
```
mesaj JSON ──► validare decizie ──► pentru fiecare acțiune:
                                      ├─ validare acțiune (unealtă cunoscută? argumente corecte?) ── nu ──► aruncată + log
                                      ├─ set_esp32 ──► publicată IMEDIAT pe vigil/alert (nu stă la coadă)
                                      ├─ debounce: aceeași unealtă la aceeași severitate în ultimele 10 s? ──► aruncată
                                      └─ pusă în coadă, după severitate
```
- **Validarea decizie:** JSON corect, `severity` întreg 1/2/3, `actions` listă. Orice altceva e respins și logat, fără ca programul să cadă.
- **Validarea acțiunii:**
  - doar cele 6 unelte cunoscute;
  - exact argumentele cerute, cu tipul corect (`True` nu trece drept număr);
  - textele nu pot fi goale;
  - nivelul alarmei e 1–3, iar cel pentru ESP32 0–3.

  Protecția e la intrarea din LLM: un LLM poate „inventa” unelte sau argumente.
- **Debounce:** aceeași unealtă la aceeași severitate rulează cel mult o dată la 10 s, ca un LLM care repetă decizia să nu repete alarma. Escaladarea trece: alarma de nivel 3 nu e blocată de una de nivel 2.
- **Coada** (`PriorityQueue`) ordonează după severitate (3 înainte de 1), iar în aceeași severitate, în ordinea sosirii.
- **Un singur worker** rulează câte o acțiune, ca vocea și alarma să nu se suprapună.

**Urgențele (severitate 3):**
1. **Întrerup acțiunea în curs** dacă e mai puțin importantă: vorbirea sau alarma se opresc în ~50 ms, inclusiv în timp ce vocea încă se generează.
2. **Anulează acțiunile mai puțin importante rămase în coadă.** După apelul 112, „Vă simțiți obosit?” nu mai are sens.
3. Nu întrerup o altă urgență.

**Log:** fiecare decizie, acțiune respinsă, debounce, întrerupere și acțiune rulată apare în terminal. Exemple: `rulez play_alarm {'level': 3} (sev 3)`, `urgență: anulat din coadă ['ask_driver']`.

### 4.2 `actions.py` — uneltele

`executor.py` cheamă `actions.run(tool, args, interrupt, publish)`, care trimite fiecare unealtă la funcția ei.

#### `speak(text)` — vorbește
1. Generează vocea în această ordine:
   1. **edge-tts**, vocea `ro-RO-EmilNeural`, puțin mai calmă și mai gravă. E cea mai naturală, dar cere internet.
   2. Dacă edge-tts nu răspunde în 5 s, trece pe **Piper** `ro_RO-mihai-medium`, local, timp de 60 s, apoi reîncearcă edge-tts.
   3. Dacă și Piper cade: `pyttsx3`, vocea Windows.
2. Generarea rulează pe un fir separat, ca o urgență să o poată abandona imediat.
3. Redă sunetul, iar redarea se oprește la urgență.

#### `play_alarm(level)` — alarmă sonoră generată în cod
- nivelul 1: un bip scurt, încet;
- nivelul 2: trei bipuri;
- nivelul 3: sirenă pe două tonuri, 4 s, tare.

#### `ask_driver(question)` — conversație cu șoferul
1. Rostește întrebarea primită de la ML 1.
2. **Pornește microfonul doar după ce a terminat de vorbit**, ca să nu se audă pe sine.
3. **Ascultă.** Șoferul are **4 s să înceapă** să răspundă. Odată ce a început, se ascultă până termină fraza (~1 s de liniște), cel mult 15 s.
4. Transcrie cu Whisper (detalii mai jos).
5. **Publică imediat** pe `vigil/driver_response`: `{"answered": true/false, "latency_sec": ..., "text": ...}`.
6. Dacă șoferul **tace**: conversația se oprește aici, iar ML 1 decide escaladarea.
7. Dacă a răspuns: LLM-ul local (Ollama, `gemma3:4b`) generează replica următoare, de exemplu „De când conduceți fără pauză?”. Se repetă de la pasul 1, cu **cel mult 3 răspunsuri** ale șoferului.
8. Conversația se încheie când LLM-ul spune că și-a atins scopul (șoferul e bine sau trebuie să oprească) sau când replica nu conține nicio întrebare.

Fiecare răspuns pleacă pe MQTT pe loc, nu la finalul conversației. Dacă șoferul spune „mă doare pieptul”, ML 1 află imediat.

#### Recunoașterea vocală (`listen` → `record` → `transcribe`)
- **`record`**: citește microfonul în bucăți de 250 ms. După fiecare bucată, `utterance_state` folosește **Silero VAD** ca să decidă: `wait` (mai ascultăm), `silence` (n-a început să vorbească în 4 s) sau `done` (a terminat fraza).
- **`transcribe`**: Whisper `small` pe CPU, cu limba română, un prompt cu contextul și VAD strict. Apoi **filtrează** fiecare bucată de text:
  - `nesigur`: Whisper inventează text pe zgomot. Un „a răspuns” fals ar ascunde un șofer inconștient.
  - `departe`: volumul e sub `NEAR_DBFS`, deci e o voce din fundal, nu a șoferului.
- **Avertizare de saturare:** dacă vârfurile ating maximul, microfonul e prea tare și vocea se distorsionează.

#### `call_112(report)` — **doar simulat**
Nu sună și nu trimite nimic în afara laptopului. Ce face:
- afișează raportul în terminal, cu „APEL 112 SIMULAT” și „AVARII PORNITE (simulat)”;
- îl salvează în `reports/112_DATA_ORA.txt`;
- anunță vocal: „Am pornit avariile și am trimis alerta de urgență, simulat...”.

Dashboard-ul poate afișa raportul din `vigil/decision`, care conține deja acțiunea `call_112` cu textul.

#### `suggest_break()` — propune o pauză
Rostește una din 3 fraze, aleasă la întâmplare, ca să nu sune repetitiv.

#### `set_esp32(level)`
Nu e în `actions.py`: executorul o publică direct pe `vigil/alert` (vezi 4.1).

#### Asistentul Virgil (`assistant`, `converse`, `heard_wake_word`)
Folosit de `voice_test.py`, vezi 4.4. `converse` e logica de conversație comună cu `ask_driver`.

### 4.3 `fake_llm.py` — decizii false
Publică pe `vigil/decision` decizii scrise de mână, ca ML 2 să poată fi testat fără ML 1. Afișează și ce răspunde șoferul (`vigil/driver_response`).

| Scenariu | Severitate | Ce trimite |
|---|---|---|
| `distragere` | 1 | memento vocal + ESP32 nivel 1 |
| `oboseala` | 2 | alarmă 2 + ESP32 2 + propunere de pauză + întrebare |
| `medical` | 3 | sirenă + ESP32 3 + „Sunteți bine?” |
| `fara_raspuns` | 3 | sirenă + ESP32 3 + apel 112 simulat |
| `invalid` | 2 | unelte greșite: trebuie respinse |
| `reset` | 1 | ESP32 0 |

```powershell
..\.venv\Scripts\python fake_llm.py                          # toate, la 15 s
..\.venv\Scripts\python fake_llm.py oboseala medical -d 2    # urgența întrerupe oboseala
```

### 4.4 `voice_test.py` — asistentul vocal Virgil
Merge fără ML 1 și fără executor. Are două stări:
- **Idle:** ascultă continuu, dar răspunde **doar la „Hei Virgil”**. Recunoașterea e aproximativă, pentru că Whisper scrie numele și „Vigil” sau „Virgilă”.
- **Conversație:** Virgil răspunde la ce ai spus. Dacă te întreabă ceva și **nu începi să vorbești în 4 s**, pornește **alarma sonoră + ESP32 nivel 2** și ascultă din nou, până răspunzi. Când spui că ești bine sau închei („mulțumesc”, „gata”), revine în idle și stinge ESP32.

Ce scrie live:
- `voice_live.json`: starea curentă, fiecare replică, volumul vocii (`speech_dbfs`), ce a fost respins și de ce, procentul de saturare;
- `recordings/ORA.wav`: fiecare frază auzită, pentru compararea modelelor Whisper pe voci reale.

Dacă brokerul nu răspunde, merge mai departe fără ESP32.

### 4.5 `esp32_test.py` — test cap-coadă al plăcii
```powershell
..\.venv\Scripts\python esp32_test.py COM13
```
1. Resetează placa prin portul serial și așteaptă mesajul `mqtt: ok`, cel mult 30 s. Afișează ce raportează placa: WiFi, IP, erori.
2. Aruncă ultima stare reținută de broker, ca să nu treacă drept răspuns.
3. Trimite pe rând nivelurile 1, 2, 3, 0 și verifică pe serial că placa a confirmat fiecare (`alert N`), cu timpul măsurat.

**Serial Monitor-ul din Arduino IDE trebuie să fie închis**, altfel portul e ocupat.

### 4.6 `test_executor.py` și `test_voice.py` — teste automate
```powershell
..\.venv\Scripts\python test_executor.py   # 13 teste, ~5 s, fără hardware
..\.venv\Scripts\python test_voice.py      # 4 teste, ~30 s, fără microfon (folosește voce sintetizată Piper)
```
| Ce verifică | Teste |
|---|---|
| validarea acțiunilor, mesaje stricate ignorate | `test_validate`, `test_bad_payloads_ignored` |
| prioritate, ESP32 direct, debounce, escaladare | `test_priority_and_esp32_direct`, `test_debounce_allows_escalation` |
| urgența întrerupe și golește coada, inclusiv în timpul generării vocii | `test_severity3_interrupts`, `test_emergency_drops_stale_actions`, `test_speak_interrupted_during_slow_synthesis` |
| conversația: tăcere, continuare, limită, LLM căzut, replici fără întrebare | `test_conversation_*`, `test_chat_reply_without_question_ends` |
| asistentul: „Hei Virgil”, alarmă la tăcere, revenire în idle | `test_wake_word`, `test_assistant_idle_wake_alert_and_back_to_idle` |
| recunoașterea: voce apropiată acceptată, voce de departe și liniște respinse, sfârșitul frazei | `test_voice.py` |

Liniile `decizie respinsă...` din timpul testelor sunt normale: testele trimit intenționat date greșite.

### 4.7 `esp32/vigil_alert/vigil_alert.ino` — programul plăcii
- Se conectează la WiFi și la broker, se abonează la `vigil/alert` și **se reconectează singur** dacă pică oricare dintre ele.
- Comandă LED-ul (și buzzerul, dacă e montat) după nivelul primit:

| Nivel | LED (GPIO 8) | Buzzer (GPIO 4, opțional) |
|---|---|---|
| 0 | stins | — |
| 1 | aprins continuu | — |
| 2 | clipește (2 ori/s) | — |
| 3 | clipește rapid (~8 ori/s) | sună |

- La pornire scanează rețelele și scrie pe serial (115200) dacă o vede pe a ta, pe ce canal și cu ce semnal. La eșec scrie motivul: `status 1` înseamnă rețea negăsită, `4` parolă greșită, iar `mqtt: eroare -2` înseamnă că brokerul nu e accesibil.
- Puterea de emisie e redusă la 8,5 dBm, pentru că multe plăci ESP32-C3 SuperMini nu se conectează la putere maximă.

---

## 5. Contractele MQTT

Contractele nu se schimbă fără acordul echipei (vezi `CLAUDE.md`).

| Topic | Direcție | Conținut |
|---|---|---|
| `vigil/decision` | ML 1 → ML 2 | `{"a":0.2,"severity":3,"actions":[{"tool":"play_alarm","args":{"level":3}}],"reason":"..."}` |
| `vigil/driver_response` | ML 2 → ML 1 | `{"answered":false,"latency_sec":null,"text":null}` |
| `vigil/alert` | ML 2 → ESP32 | `0`–`3`, trimis cu *retain* |

Uneltele acceptate și argumentele lor:

| Unealtă | Argumente | Observații |
|---|---|---|
| `speak` | `{"text": "..."}` | |
| `play_alarm` | `{"level": 1-3}` | |
| `ask_driver` | `{"question": "..."}` | pornește o conversație de cel mult 3 răspunsuri |
| `set_esp32` | `{"level": 0-3}` | executată imediat, fără coadă |
| `call_112` | `{"report": "..."}` | doar simulat |
| `suggest_break` | `{}` | |

**De știut pentru ML 1:**
- **Pe `vigil/driver_response` pot veni mai multe mesaje pentru o singură întrebare:** unul pentru fiecare răspuns din conversație.
- **Puneți întrebări deschise** („Cum vă simțiți?”), nu da/nu („Sunteți bine?”). Un „da” scurt e recunoscut prost de Whisper.
- **Fereastra de răspuns e de 4 s** până când șoferul trebuie să înceapă să vorbească.
- **O decizie de severitate 3 anulează acțiunile mai puțin importante** încă neexecutate.

---

## 6. Reglaje

Toate se află în capul fișierului `actions.py`.

| Constantă | Valoare | Ce face |
|---|---|---|
| `LISTEN_SEC` | 4 | secunde în care șoferul trebuie să înceapă să răspundă |
| `END_SILENCE_SEC` | 1.0 | liniștea după care se consideră că a terminat fraza |
| `MAX_UTTERANCE_SEC` | 15 | durata maximă a unei fraze |
| `NEAR_DBFS` | -40 | volumul sub care vocea e „din fundal”. **De calibrat** pe microfonul de la demo: ~10 dB sub `speech_dbfs` din `voice_live.json` |
| `WHISPER_MODEL` | `small` | `large-v3-turbo` recunoaște mai bine frazele libere, dar e ~3× mai lent (~6 s/frază) |
| `PROMPT` | context + „Hei Virgil” | ghidează Whisper. **Nu pune răspunsuri-exemplu**: pe audio scurt Whisper copiază promptul |
| `EDGE_VOICE` | `ro-RO-EmilNeural` | sau `ro-RO-AlinaNeural` |
| `LLM_MODEL` | `gemma3:4b` | `llama3.2:3b` e mai rapid, dar inventează și nu respectă regulile |
| `MAX_TURNS` | 3 | răspunsuri ale șoferului într-un `ask_driver` |
| `PERSONA` | — | personalitatea și regulile lui Virgil. **Fără ghilimele românești „...” în text**: modelul le copiază și strică JSON-ul |

În `executor.py`: `DEBOUNCE_SEC = 10`, plus variabila de mediu `VIGIL_BROKER`, cu adresa brokerului.

---

## 7. Testare înainte de demo

| # | Comandă | Rezultat așteptat |
|---|---|---|
| 1 | `..\.venv\Scripts\python test_executor.py` | 13 × `ok` |
| 2 | `..\.venv\Scripts\python test_voice.py` | 4 × `ok` |
| 3 | `..\.venv\Scripts\python esp32_test.py COM13` | 4 × `nivel N: OK`; LED aprins → clipind → clipind rapid → stins |
| 4 | `executor.py` + `fake_llm.py distragere oboseala medical fara_raspuns reset -d 15` | vocea și LED-ul potrivite fiecărui scenariu; raport în `reports/` |
| 5 | `executor.py` + `fake_llm.py oboseala medical -d 2` | sirena taie imediat mesajul de oboseală |
| 6 | `..\.venv\Scripts\python voice_test.py` | „Hei Virgil” → conversație → alarmă la tăcere → idle după „Sunt bine” |

Înainte de testele 3–6, verifică:
- hotspot pe 2.4 GHz, cu internet;
- ESP32 în USB;
- Ollama pornit (iconița din tray);
- Serial Monitor-ul din Arduino IDE închis.

---

## 8. Probleme cunoscute și depanare

| Simptom | Cauză / soluție |
|---|---|
| **Recunoaștere vocală slabă** | Microfonul laptopului prinde oamenii din jur aproape la fel de tare ca pe șofer (diferență de ~4 dB). **Folosiți căști cu microfon.** Scădeți volumul microfonului la 60–70% dacă apare avertismentul de saturare. |
| Conversația are pauze de ~7 s | Atât durează `gemma3:4b` pe un GTX 1650 de 4 GB (modelul încape doar pe jumătate în placa video). Un laptop cu placă video mai puternică le reduce. Dacă ML 1 rulează alt LLM pe același laptop, se încetinesc reciproc. |
| `edge-tts indisponibil`, vocea devine robotică | Fără internet sau internet lent: trece automat pe Piper. |
| Prima întrebare vine târziu | Prima încărcare a lui `gemma3:4b` durează ~50 s. Așteptați mesajul `modelele de voce sunt încărcate`. |
| ESP32: `NU văd <rețea>` | Rețeaua e pe 5 GHz. Setați 2.4 GHz și reporniți hotspotul. |
| ESP32: `mqtt: eroare -2` | Brokerul nu e accesibil: IP greșit în `BROKER_IP` (s-a schimbat IP-ul laptopului?) sau broker oprit (`Get-Service mosquitto`). |
| ESP32 nu primește nimic, deși scrie `mqtt: ok` | Rulează două brokere (vezi 2.3). `netstat -ano \| findstr :1883` trebuie să arate o singură linie LISTENING. |
| `could not open port 'COM13': Access is denied` | Portul e ocupat, de obicei de Serial Monitor-ul din Arduino IDE. |
| Placa nu mai răspunde după reset | La ESP32-C3, resetul poate lăsa placa în bootloader. Scoateți și puneți la loc cablul USB. |
| Buzzerul sau LED-ul pornesc singure la pornire | Ultimul nivel e reținut în broker. Executorul trimite `0` la pornire. Manual: `mosquitto_pub -h localhost -t vigil/alert -m 0 -r`. |
| Asistentul nu se trezește la „Hei Virgil” | Spuneți numele clar. Verificați în `voice_live.json` ce a auzit Whisper. |

---

## 9. Ce nu e făcut

- **Controlul vehiculului cu fuzzy logic** (opțional, `scikit-fuzzy` + `highway-env`): încă neimplementat. Pachetele sunt comentate în `requirements.txt`.
- **„Hei Virgil” în executor:** asistentul cu idle există doar în `voice_test.py`. În executor, conversația pornește numai când ML 1 trimite `ask_driver`.
- **Recunoașterea unui anumit vorbitor:** Virgil ascultă vocea cea mai apropiată, nu doar vocea șoferului.
- **Calibrarea `NEAR_DBFS`** pe microfonul de la demo.
