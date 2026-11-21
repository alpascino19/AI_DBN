# Vigil — monitorizarea șoferului (CV + LLM + acțiuni)

Camera vede șoferul, ML1 decide cât de grav e, ML2 acționează (voce, alarmă, LED/buzzer ESP32, apel 112
simulat), iar dashboard-ul arată totul live.

```
Vision (CV1 + CV2) ──vigil/vision──► ML1 (decide) ──vigil/decision──► ML2 (execută) ──vigil/alert──► ESP32
   camera / date false                   ▲  │                             │
                                         │  └──vigil/ecu──► ECU (viteză, asistență de bandă, oprire pe banda de urgență)
                                         └──────vigil/driver_response─────┘
                         dashboard React (browser) ◄── toate topicurile, prin WebSocket
```

## Pornire: folderul [`START/`](START/README.md)

| | |
|---|---|
| prima dată | `START\1_instalare.bat`, apoi `START\0_broker_admin.bat` ca administrator |
| demo fără cameră | `START\3_pornire_demo.bat` |
| cu camera reală | `START\4_pornire_camera.bat` |
| cu serverul Spring Boot | `START\5_pornire_spring.bat` |
| teste | `START\6_teste.bat` |

Toate instrucțiunile, opțiunile și problemele frecvente: **[START/README.md](START/README.md)**.

## Structura

| Folder | Echipa | Ce face | Documentație |
|---|---|---|---|
| [`START/`](START/) | toți | instalare, pornirea întregului sistem (`run_vigil.py` + fișiere `.bat`), teste | [START/README.md](START/README.md) |
| [`Vision/`](Vision/) | CV1 + CV2 | MediaPipe + YOLO → EAR, PERCLOS, privire, mână pe piept, respirație, telefon, prăbușire | [Vision/README.md](Vision/README.md) |
| [`ML1/`](ML1/) | ML1 | fuziune → scor de atenție, reguli de siguranță, LLM local (Qwen2.5 1.5B), gărzi, API pentru Spring Boot, matricea de probabilitate a stărilor + control fuzzy pentru ECU; cod, teste, documentație, pitch | [ML1/README.md](ML1/README.md) |
| [`ML2/`](ML2/) | ML2 | executor: voce (edge-tts / Piper), conversație (Whisper + LLM), alarmă, ESP32, 112 simulat | [ML2/README.md](ML2/README.md) |
| [`dashboard/`](dashboard/) | ML2 | interfața live (React + mqtt.js) | [dashboard/README.md](dashboard/README.md) |

Dependențele Python ale tuturor modulelor: [`requirements.txt`](requirements.txt) (un singur `.venv` la rădăcină).
Împărțirea taskurilor: [TASKS.md](TASKS.md). Conceptul inițial (viziune, MPC, cadru legal):
[docs/CONCEPT.md](docs/CONCEPT.md). Rularea pe 3 laptopuri: [START/3_LAPTOPURI.md](START/3_LAPTOPURI.md).
