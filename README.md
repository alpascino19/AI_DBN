# Vigil — monitorizarea șoferului (CV + LLM + acțiuni)

Camera vede șoferul, ML1 decide cât de grav e, ML2 acționează (voce, alarmă, LED/buzzer ESP32, apel 112 simulat).

```
Vision (CV1 + CV2) ──vigil/vision──► ML1 (decide) ──vigil/decision──► ML2 (execută) ──vigil/alert──► ESP32
   camera / date false                   ▲                                │
                                         └──────vigil/driver_response─────┘
```

| Folder | Echipa | Ce face | Documentație |
|---|---|---|---|
| [`Vision/`](Vision/) | CV1 + CV2 | MediaPipe + YOLO → EAR, PERCLOS, privire, mână pe piept, respirație, telefon, prăbușire | [Vision/README.md](Vision/README.md) |
| [`ml1/`](ml1/) | ML1 | fuziune → scor de atenție, reguli de siguranță, LLM local (Qwen2.5 1.5B), gărzi, API pentru Spring Boot | [docs/README.md](docs/README.md) |
| [`ML2/`](ML2/) | ML2 | executor: voce (edge-tts / Piper), conversație (Whisper + LLM), alarmă, ESP32, 112 simulat | [ML2/README.md](ML2/README.md) |
| [`docs/`](docs/) | ML1 | arhitectură, metode ML, evaluare, API, deep dive (PDF) | [docs/README.md](docs/README.md) |

## Instalare (o singură dată, Windows 11, Python 3.14)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt                          # Vision + ML1 + ML2
winget install -e --id EclipseFoundation.Mosquitto                     # broker MQTT (pornește ca serviciu)
winget install -e --id Ollama.Ollama                                   # LLM local
winget install -e --id Gyan.FFmpeg                                     # vocea edge-tts a ML2
ollama pull qwen2.5:1.5b                                               # ML1: decizia (~1 GB)
ollama pull qwen2.5:3b                                                 # ML2: conversația (~2 GB)
.venv\Scripts\python -m piper.download_voices ro_RO-mihai-medium --download-dir ML2\voices   # voce offline
```

Pentru ESP32 (brokerul ascultă în rețea, Wi-Fi 2.4 GHz, flash): pașii din [ML2/README.md](ML2/README.md#24-placa-esp32).

## Pornire

```powershell
.venv\Scripts\python run_vigil.py --check          # verifică broker, modele, pachete
.venv\Scripts\python run_vigil.py --fake           # tot sistemul, fără cameră (scenariul demo al echipei CV)
.venv\Scripts\python run_vigil.py                  # cameră reală
.venv\Scripts\python run_vigil.py --show           # + fereastra de debug a camerei
.venv\Scripts\python run_vigil.py --input http     # ML1 primește cadre de la serverul Spring Boot (docs/API.md)
```

Opțiuni utile: `--no-llm` (ML1 doar cu reguli), `--no-ml2` (fără voce), `--voice-llm gemma3:4b`
(alt model pentru conversație), `--duration 120` (oprire automată). Logurile complete: `logs/`.

**Modelul conversației (ML2).** `run_vigil.py` folosește implicit `qwen2.5:3b`; `ML2/executor.py` pornit
singur folosește în continuare `gemma3:4b` (sau ce e în variabila `VIGIL_VOICE_LLM`). Comparația pe aceleași
replici:

| Model | Replici de siguranță (oboseală, piept, „sunt bine”) | Conversație liberă | Memorie GPU |
|---|---|---|---|
| `gemma3:4b` | corecte | cea mai bună (dar a inventat o oră de sosire) | ~3.3 GB |
| `qwen2.5:3b` | corecte | inteligibilă, uneori stângace | ~2 GB |
| `qwen2.5:1.5b` | corecte | incoerentă la întrebări deschise | 0 în plus (același model ca ML1) |

## Teste

```powershell
.venv\Scripts\python -m unittest                                  # ML1 (114 teste)
cd Vision; ..\.venv\Scripts\python -m unittest discover -s tests   # Vision
cd ML2; ..\.venv\Scripts\python test_executor.py; ..\.venv\Scripts\python test_voice.py
```
