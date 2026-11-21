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
| **Ollama** (LLM local) | `winget install -e --id Ollama.Ollama`, apoi `ollama pull qwen2.5:1.5b` (1 GB; același model ca ML 1). Opțional, pentru conversații mai bune: `ollama pull gemma3:4b` (3,4 GB) | replicile conversației |
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
Configul deschide și portul **9001 (WebSocket)**, pentru dashboard-ul React (`dashboard/`).

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
- **Validarea decizie:** JSON corect, `severity` întreg 0–3, `actions` listă. Orice altceva e respins și logat, fără ca programul să cadă. `severity: 0` e „revenire la normal” de la ML 1 și stinge ESP32.
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
7. Dacă a răspuns: LLM-ul local (Ollama, modelul din `VIGIL_MODEL`, implicit `qwen2.5:1.5b`) generează replica următoare, de exemplu „De când conduceți fără pauză?”. Se repetă de la pasul 1, cu **cel mult 3 răspunsuri** ale șoferului.
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

### 4.4b `info_assistant.py` — Virgil, asistentul de informare
Program separat de executor, doar pentru informare: vreme, știri, trafic, întrebări generale.
```powershell
$env:OMNIROUTE_API_KEY  # trebuie setată (vezi mai jos)
..\.venv\Scripts\python info_assistant.py
```
- **„Hei Virgil, <întrebare>”**, sau doar „Hei Virgil”, iar el răspunde „Da, vă ascult”.
- **După fiecare răspuns** se aude un bip și Virgil ascultă încă o întrebare, fără „Hei Virgil”. Tăcerea sau „mulțumesc”/„gata” închid conversația. Terminalul arată la fiecare pas ce să faceți (`>>> GATA...`, `>>> VORBIȚI ACUM...`, `>>> Mai aveți o întrebare?...`).
- **Date live, alese după cuvintele din întrebare:**
  - vreme: Open-Meteo, fără cheie;
  - știri: RSS HotNews / Digi24;
  - trafic: **alertele live Waze** pe o rază de 10 km (accidente, ambuteiaje, pericole, poliție), prin OpenWebNinja (`OPENWEBNINJA_API_KEY`; fiecare întrebare de trafic consumă o cerere din credit), plus căutarea Tavily „trafic Cluj-Napoca”;
  - durata până la o destinație („Cât fac până la gară?”, „Cât durează drumul spre aeroport?”): **Google Maps Routes API** (`GOOGLE_MAPS_API_KEY`, cu Routes API activat în Google Cloud), cu traficul de acum și întârzierea față de drumul liber; plecarea e „de la ...” din întrebare, altfel poziția curentă din serviciul de localizare Windows (după Wi-Fi, ~100 m; Setări → Confidențialitate → Locație trebuie să fie pornită), altfel centrul Clujului;
  - orice altceva (sport, evenimente, prețuri, întrebări generale): **căutare Tavily prin OmniRoute** (cheia Tavily se pune în dashboard-ul OmniRoute; ~1.000 de căutări gratuite pe lună). La întrebările cu „azi/acum/diseară/mâine” se adaugă data zilei în căutare. Dacă Tavily nu răspunde, Virgil spune că nu are informația. Variante abandonate: DuckDuckGo prin OmniRoute (nu mai întorcea rezultate) și Gemini cu Google Search (limita gratuită se umplea repede).
- **Răspunsul îl formulează un model Groq prin OmniRoute** (`http://localhost:20128`, providerul Groq conectat în OmniRoute), ~2–3 s pe răspuns. Se încearcă pe rând `groq/qwen/qwen3.8-27b` → `groq/openai/gpt-oss-120b` → `groq/openai/gpt-oss-20b` (următorul doar dacă primul dă eroare). Primul model se schimbă cu `$env:VIRGIL_MODEL='...'`. Gemini gratuit nu mai e folosit: dădea des 429 și fiecare întrebare aștepta eroarea.
- **Nu se amestecă în siguranță:** ascultă `vigil/decision`, iar la orice alertă (severitate ≥ 1) se oprește din vorbit imediat și tace 60 s.
- **Locația e fixă: Cluj-Napoca** (`CITY`, `LAT`, `LON` în capul fișierului).
- **Cheile se pun ca variabile de mediu,** apoi se deschide un terminal nou. Cheia OpenWebNinja e opțională: fără ea, traficul vine doar din știri.
  ```powershell
  [Environment]::SetEnvironmentVariable('OMNIROUTE_API_KEY', 'CHEIA', 'User')
  [Environment]::SetEnvironmentVariable('OPENWEBNINJA_API_KEY', 'CHEIA', 'User')   # app.openwebninja.com, API Waze
  [Environment]::SetEnvironmentVariable('GOOGLE_MAPS_API_KEY', 'CHEIA', 'User')    # opțional: durata până la destinație
  ```
- **Test:** `..\.venv\Scripts\python test_info_assistant.py`.

### 4.4c Butoanele asistentului din dashboard (`executor.py` + `info.py`)
Cardul **Asistent Virgil** din dashboard publică pe `vigil/assistant`; răspunde executorul, deci nu trebuie
pornit nimic în plus.
- **„Întreabă-l pe Virgil”**: Virgil spune „Vă ascult” și ascultă o întrebare (4 s ca să începeți). Tăcerea
  nu pornește alarma: Virgil spune doar că nu a auzit.
- **„Ora”, „Vremea”, „Traficul”**: răspuns direct, fără microfon (sigur și într-o sală zgomotoasă).
- **De unde vin răspunsurile** (`actions.assistant_lookup`):
  - ora: ceasul laptopului, mereu;
  - cu `OMNIROUTE_API_KEY` setată: asistentul de informare de la 4.4b (`info_assistant.answer`: Waze, Tavily,
    știri, model prin OmniRoute), deci aceleași răspunsuri ca la „Hei Virgil”;
  - fără cheie sau dacă OmniRoute nu răspunde în 20 s: `info.py`, adică Open-Meteo pentru vreme (fără cheie),
    TomTom pentru trafic dacă există `TOMTOM_KEY`, altfel o **estimare după oră spusă ca estimare**
    („Nu am date de trafic în timp real; ...”), iar întrebările libere merg la LLM-ul local, doar cu datele reale.
- **Siguranța are prioritate:** cererile asistentului au prioritatea cea mai mică în coadă, orice decizie ML 1
  de severitate ≥ 1 le întrerupe, iar severitatea 3 le anulează. Căutările rulează pe un fir separat:
  o urgență nu așteaptă după o cerere HTTP lentă.
- **Locația:** Cluj-Napoca, ca la 4.4b (`VIGIL_LAT`, `VIGIL_LON`, `VIGIL_PLACE` pentru alta). Spre Open-Meteo
  și TomTom pleacă doar coordonatele acestei locații.
- **Test:** `..\.venv\Scripts\python test_info.py` și `test_executor.py` (fără internet, microfon sau difuzoare).

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
| `vigil/assistant` *(nou)* | dashboard → ML 2 | `{"request": "listen" \| "time" \| "weather" \| "traffic"}`: butoanele asistentului |
| `vigil/assistant/reply` *(nou)* | ML 2 → dashboard | `{"request", "status": "listening" \| "working" \| "speaking" \| "done" \| "interrupted", "heard", "text", "source"}` |

| `vigil/ecu` | ML 1 → ML 2 | comanda fuzzy continuă (5 Hz): `mode`, `target_kmh`, `hazards`... |
| `vigil/car/cmd` | ML 2 → ECU_Arduino | doar schimbările (`CarLink` în `executor.py`): `{"cmd":"emergency","decel_mps2":2.5,"id":"ml2-3"}`, `slow_down`, `hazard`, `cancel_emergency`, `reset` |
| `vigil/car/state` / `vigil/car/event` | ECU_Arduino → ML 1, dashboard | viteza reală, frânare, avarii, oprit; evenimentele poartă `id`-ul comenzii ML 2 |
| `vigil/video/face` / `vigil/video/body` | Vision (`--video`) → dashboard | JPEG binar, 480 px, 5 cadre/s, QoS 0, fără retain |

Topicurile asistentului, ale mașinii și ale imaginii sunt noi și nu schimbă contractele existente. O cerere
din dashboard nu publică niciodată pe `vigil/driver_response`.

**Lanțul ECU** (combinat din ML1_Pasca / ML2_Pasca): ML 1 calculează comanda fuzzy (`vigil/ecu`) → ML 2 o
traduce în comenzi discrete pentru placa ECU (`vigil/car/cmd`) → placa simulează mașina și raportează
(`vigil/car/state`, `vigil/car/event`) → ML 1 pune starea reală în status și în raportul 112, iar dashboard-ul
desenează graficul vitezei: ce cere ML 1 vs ce face ECU-ul. La pornire ML 2 trimite `reset` (croazieră, fără avarii).

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
| `LLM_MODEL` | variabila de mediu `VIGIL_MODEL`, implicit `qwen2.5:1.5b` | aceeași variabilă ca ML 1. `qwen2.5:1.5b`: ~2,9 s/replică, dar româna liberă e slabă (merg bine doar situațiile din exemplele din prompt). `gemma3:4b`: ~4,4 s, româna corectă. Pentru ML 2 pe gemma: `$env:VIGIL_MODEL='gemma3:4b'` în terminalul executorului. Ambele încap împreună pe un GTX 1650 de 4 GB (testat). `llama3.2:3b` inventează și nu respectă regulile |
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
| Conversația are pauze lungi | Replica LLM durează ~2,9 s cu `qwen2.5:1.5b` și ~4,4 s cu `gemma3:4b` pe un GTX 1650 (după prima încărcare). Dacă ML 1 și ML 2 cer un răspuns în același timp, se încetinesc reciproc. |
| Replici fără sens sau bucăți din instrucțiuni („pui o întrebare scurtă...”) | `qwen2.5:1.5b` e prea mic pentru conversație liberă în română. Treceți ML 2 pe `gemma3:4b` (vezi `LLM_MODEL` în secțiunea 6). |
| `edge-tts indisponibil`, vocea devine robotică | Fără internet sau internet lent: trece automat pe Piper. |
| Prima întrebare vine târziu | Prima încărcare a modelului de pe disc durează 15–50 s. Așteptați mesajul `modelele de voce sunt încărcate`. |
| ESP32: `NU văd <rețea>` | Rețeaua e pe 5 GHz. Setați 2.4 GHz și reporniți hotspotul. |
| ESP32: `mqtt: eroare -2` | Brokerul nu e accesibil: IP greșit în `BROKER_IP` (s-a schimbat IP-ul laptopului?) sau broker oprit (`Get-Service mosquitto`). |
| ESP32 nu primește nimic, deși scrie `mqtt: ok` | Rulează două brokere (vezi 2.3). `netstat -ano \| findstr :1883` trebuie să arate o singură linie LISTENING. |
| `could not open port 'COM13': Access is denied` | Portul e ocupat, de obicei de Serial Monitor-ul din Arduino IDE. |
| Placa nu mai răspunde după reset | La ESP32-C3, resetul poate lăsa placa în bootloader. Scoateți și puneți la loc cablul USB. |
| Buzzerul sau LED-ul pornesc singure la pornire | Ultimul nivel e reținut în broker. Executorul trimite `0` la pornire. Manual: `mosquitto_pub -h localhost -t vigil/alert -m 0 -r`. |
| Asistentul nu se trezește la „Hei Virgil” | Spuneți numele clar. Verificați în `voice_live.json` ce a auzit Whisper. |

---

## 9. Ce nu e făcut

- **Bucla închisă cu vehiculul:** comanda fuzzy pentru ECU există în ML 1 (`ML1/ml1/ecu_fuzzy.py`, topicul `vigil/ecu`, vezi [ML1/docs/PROBABILITY_ECU.md](../ML1/docs/PROBABILITY_ECU.md)), dar nu e legată de o mașină sau de `highway-env`.
- **„Hei Virgil” în executor:** asistentul cu idle există doar în `voice_test.py`. În executor, conversația pornește numai când ML 1 trimite `ask_driver`.
- **Recunoașterea unui anumit vorbitor:** Virgil ascultă vocea cea mai apropiată, nu doar vocea șoferului.
- **Calibrarea `NEAR_DBFS`** pe microfonul de la demo.
