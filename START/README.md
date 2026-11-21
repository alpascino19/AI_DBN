# START — cum pornești Vigil

Tot sistemul pornește de aici: **camera (Vision) → ML1 (decide) → ML2 (voce, alarmă, ESP32, 112 simulat) →
dashboard-ul React** în browser. Fișierele `.bat` se pornesc cu **dublu-clic** (sau din terminal, cu opțiuni
la final). Toate lucrează din rădăcina repo-ului și folosesc mediul Python comun `.venv`.

```
Vision ──vigil/vision──► ML1 ──vigil/decision──► ML2 ──vigil/alert──► ESP32
 camera / date false       ▲                       │
                           └──vigil/driver_response┘
            (toate trec prin brokerul Mosquitto; dashboard-ul le citește din browser)
```

## Prima dată (o singură dată pe calculator)

| Pas | Fișier | Ce face | Durată |
|---|---|---|---|
| 1 | **`1_instalare.bat`** | instalează Mosquitto, Ollama, ffmpeg, Node.js, mediul `.venv` cu toate pachetele Python (Vision + ML1 + ML2), modelul `qwen2.5:1.5b`, vocea offline Piper, pachetele dashboard-ului; la final verifică tot | 10–20 min |
| 2 | **`0_broker_admin.bat`** — clic dreapta → *Run as administrator* | deschide brokerul pe portul **9001** (dashboard-ul din browser) și pe **1883 în rețea** (placa ESP32). Configul vechi rămâne salvat ca `mosquitto.conf.original` | 5 s |

Ai nevoie de: Windows 10/11, Python 3.11+ (testat pe 3.14), internet la instalare, cameră web pentru modul real.
Fără placă video merge și pe procesor, doar mai încet.

> Pasul 2 face brokerul accesibil fără parolă din rețeaua locală. Folosiți propriul hotspot la demo, nu un Wi-Fi public.

## De fiecare dată

| Fișier | Ce pornește | Când |
|---|---|---|
| **`2_verificare.bat`** | doar verificări: broker, Ollama, modele, pachete, dashboard | dacă ceva nu merge |
| **`3_pornire_demo.bat`** | tot sistemul **fără cameră**: Vision trimite scenariul demo al echipei CV (telefon, somnolență, mână pe piept, prăbușire) | repetiții, test rapid |
| **`4_pornire_camera.bat`** | tot sistemul **cu camera reală** | demo-ul adevărat |
| **`5_pornire_spring.bat`** | ML1 primește cadrele prin **HTTP** de la serverul Spring Boot (port 8090), restul la fel | integrarea cu serverul |
| **`6_teste.bat`** | toate testele automate: ML1, Vision, ML2, dashboard | după orice modificare |

Dashboard-ul se deschide singur la **http://localhost:5173** (de pe tabletă / alt laptop din aceeași rețea:
`http://IP-LAPTOP:5173`). **Ctrl+C** în fereastra neagră oprește tot. Logurile complete: `logs/ml1.log`,
`logs/ml2.log`, `logs/cv.log`, `logs/web.log`.

Ce vezi la `3_pornire_demo.bat`, în ordine: ML2 încarcă vocea (10–50 s) → ML1 face warmup la Qwen →
Vision începe scenariul → la telefon un memento vocal blând, la somnolență alarmă + „Faceți o pauză?”, la
prăbușire alarmă urgentă + „Sunteți bine?”, iar dacă nimeni nu răspunde, raportul 112 simulat (fereastra roșie
din dashboard).

## Opțiuni (se adaugă după numele fișierului, în terminal)

```powershell
START\3_pornire_demo.bat --scenario chest        # doar un scenariu: demo | normal | phone | drowsy | chest | unconscious
START\4_pornire_camera.bat --show                # + fereastra de debug a camerei
START\4_pornire_camera.bat --camera 1            # altă cameră
START\4_pornire_camera.bat --voice-llm gemma3:4b # conversație mai bună în română (după: ollama pull gemma3:4b)
START\3_pornire_demo.bat --no-llm                # ML1 doar cu reguli (fără Ollama)
START\3_pornire_demo.bat --no-ml2                # fără voce / alarmă
START\3_pornire_demo.bat --no-dashboard          # fără dashboard
START\3_pornire_demo.bat --duration 120          # oprire automată după 2 minute
```

Toate opțiunile: `.venv\Scripts\python START\run_vigil.py --help` (scriptul pe care îl pornesc fișierele `.bat`).

**Modelul conversației (ML2).** Implicit, ML2 folosește același model ca ML1 (`qwen2.5:1.5b`): un singur model în
placa video, dar româna liberă e slabă. Pe aceleași replici:

| `--voice-llm` | Replici de siguranță (oboseală, piept, „sunt bine”) | Conversație liberă | Memorie GPU în plus |
|---|---|---|---|
| *(implicit)* `qwen2.5:1.5b` | corecte | incoerentă la întrebări deschise | 0 |
| `qwen2.5:3b` | corecte | inteligibilă, uneori stângace (licență necomercială) | ~2 GB |
| `gemma3:4b` | corecte | cea mai bună | ~3.3 GB |

## Probleme frecvente

| Simptom | Ce faci |
|---|---|
| `brokerul MQTT nu răspunde pe localhost:1883` | `Start-Service mosquitto` (sau rulează din nou `1_instalare.bat`) |
| dashboard-ul arată „deconectat” / fără date | rulează `0_broker_admin.bat` ca administrator (portul 9001) |
| `Ollama nu răspunde` | pornește aplicația Ollama din Start; scriptul încearcă și singur |
| `model ... LIPSEȘTE` | `ollama pull qwen2.5:1.5b` |
| vocea e robotică | fără internet edge-tts nu merge; trece automat pe vocea offline Piper |
| camera nu pornește | închide alte aplicații care o folosesc (Teams, Zoom) sau încearcă `--camera 1` |
| fereastra se închide imediat | rulează fișierul din terminal (`cmd`), ca să vezi eroarea |

## Unde e restul

| Folder | Ce conține | Documentație |
|---|---|---|
| [`../Vision/`](../Vision/) | CV1 + CV2: MediaPipe + YOLO → EAR, PERCLOS, privire, mână pe piept, respirație, telefon | [Vision/README.md](../Vision/README.md), [COMPUTER_VISION.md](../Vision/COMPUTER_VISION.md) |
| [`../ML1/`](../ML1/) | motorul de decizie (fuziune, reguli, Qwen, API Spring Boot), cu propria comandă `run_ml1.py` | [ML1/README.md](../ML1/README.md) |
| [`../ML2/`](../ML2/) | executorul: voce, conversație, alarmă, ESP32, 112 simulat | [ML2/README.md](../ML2/README.md) |
| [`../dashboard/`](../dashboard/) | dashboard-ul React (MQTT prin WebSocket) | [dashboard/README.md](../dashboard/README.md) |
