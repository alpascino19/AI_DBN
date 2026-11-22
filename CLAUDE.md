# Vigil — context pentru Claude Code

Proiect de hackathon (4 persoane, 24–48 h): sistem care monitorizează șoferul prin cameră și lasă un **LLM local să ia acțiuni** pe baza a ce vede camera. Detalii: `vigil_real_time_health_driver_safety_monitoring_system.md`. Împărțirea taskurilor: `TASKS.md`.

## Flux
```
Cameră → CV (față + corp) → fuziune / scor de atenție a ∈ [0,1] → LLM (decide) → acțiuni (voce, alarmă, ESP32, 112 simulat)
```
Totul rulează local pe laptop. Modulele comunică prin **MQTT** (Mosquitto local).

## Echipa
- **CV 1** — față și ochi (EAR, PERCLOS, clipit, poziția capului, privire, cap căzut, calibrare 30 s)
- **Dashboard** — `dashboard/`: React + Vite, conectat direct la Mosquitto prin WebSocket (port 9001), fără backend. Detalii în `dashboard/README.md`.
- **CV 2** — bucla camerei, mâna pe piept, respirație grea, telefon (YOLO)
- **ML 1** — fuziune + motorul de decizie LLM (Ollama, function calling), raport 112, pitch
- **ML 2** — executorul de acțiuni, vocea, ESP32; opțional controlul vehiculului cu fuzzy logic

## Contracte MQTT (nu le schimba fără acordul echipei)
| Topic | De la → la | Conținut |
|---|---|---|
| `vigil/vision` | CV → ML 1, dashboard | `{"ear":0.21,"perclos":0.3,"blink_rate":28,"gaze_on_road":false,"head_drop":false,"phone":0.1,"events":[{"event":"hand_on_chest","duration_sec":4,"confidence":0.8,"breathing":"heavy"}],"landmarks":[...]}` |
| `vigil/decision` | ML 1 → ML 2, dashboard | `{"a":0.2,"severity":3,"actions":[{"tool":"play_alarm","args":{"level":3}}],"reason":"..."}` |
| `vigil/driver_response` | ML 2 → ML 1 | `{"answered":false,"latency_sec":null,"text":null}` |
| `vigil/alert` | ML 2 → ESP32 | `0`–`3` |

Severitate: 1 = distragere (memento vocal), 2 = oboseală (alarmă, pauză, conversație), 3 = urgență medicală / fără reacție (alarmă urgentă, buzzer, 112 simulat).

## ML 2 — executorul de acțiuni (cod în `ML2/`)
- `paho-mqtt` 2.x primește `vigil/decision` → **validare** (doar uneltele de mai jos, argumente corecte; restul se aruncă și se loghează) → `queue.PriorityQueue` după severitate → **un singur worker**.
- Unelte: `speak(text)`, `play_alarm(level)`, `ask_driver(question)`, `set_esp32(level)`, `call_112(report)`, `suggest_break()`.
- Severitatea 3 întrerupe acțiunea în curs (`threading.Event`). `set_esp32` se publică imediat, pe lângă coadă.
- Debounce: aceeași acțiune cel mult o dată la ~10 s.
- `ask_driver`: conversație scurtă (max 3 răspunsuri) în stil Jarvis — replicile le generează un LLM local (Ollama, modelul din `VIGIL_MODEL`, implicit `qwen2.5:1.5b`, ca ML 1; `gemma3:4b` dă o română mult mai bună, la ~4,4 s/replică). Microfonul pornește **după** ce TTS-ul termină și ascultă fix **4 s**. Fiecare răspuns sau tăcere pleacă imediat pe `vigil/driver_response`.
- Asistentul vocal **Virgil** (`voice_test.py`, `actions.assistant`): în idle răspunde doar la „Hei Virgil”; în conversație, tăcere 4 s → alarmă + ESP32 nivel 2; când șoferul spune că e bine (LLM `end=true`) → înapoi în idle. ML 1 ar trebui să trimită întrebări deschise („Cum vă simțiți?”), nu da/nu: un „da” scurt e recunoscut prost.
- TTS: `edge-tts`, voce `ro-RO-EmilNeural` (cere internet; pleacă doar textul rostit), rezervă automată Piper `ro_RO-mihai-medium`, apoi `pyttsx3`. STT: `faster-whisper` `small`, `language="ro"`, prompt cu vocabular, VAD strict, filtru de volum pentru vocea apropiată.
- `call_112` este **doar simulat** în executor (afișează raportul pe dashboard + avarii simulate). Nu suna niciodată la 112.
- Excepție opțională: `emergency_notify.py` (proces separat) anunță la `call_112` **doar** persoana de contact din `ML2/.env` (mesaj Telegram prin CallMeBot, apel de pe Android prin `adb`). `.env` nu se urcă pe git.
- Fișier de test `fake_llm.py` care publică decizii false pe `vigil/decision`.

## ESP32
Placa e **ESP32-C3** (`--fqbn esp32:esp32:esp32c3:CDCOnBoot=cdc`, COM13). Arduino + `PubSubClient`, ascultă `vigil/alert`: 0 stins, 1 LED aprins, 2 LED clipește, 3 LED clipește rapid (+ buzzer pe GPIO 4, dacă e montat). WiFi în `wifi_secrets.h` (nu se urcă pe git); test: `python esp32_test.py COM13`. Reconectare automată la WiFi și broker. IP-ul brokerului într-o singură variabilă. Mosquitto 2.x are nevoie de `listener 1883 0.0.0.0` + `allow_anonymous true`. La demo: hotspot propriu.

## Opțional — control cu fuzzy logic (doar după ce executorul și ESP32 merg)
Nu MPC. `scikit-fuzzy`, Mamdani, defuzzificare centroid. Intrări: scorul `a` și distanța în timp față de mașina din față (s). Ieșire: reducerea vitezei față de limită. 6–9 reguli după treptele: atent → limita drumului; obosit/distras → limită − 20 km/h, 3 s distanță; urgență → viteză spre 0 (manevra de risc minim). Simulare în `highway-env`.

## Convenții
- Python 3.10+, librăria standard unde se poate (`threading`, `queue`), fără framework-uri în plus.
- Fiecare modul se poate testa singur pe date false publicate pe MQTT.
- Lucrează local și fă `git pull` înainte de `git push`; nu urca fișiere prin interfața web GitHub.
