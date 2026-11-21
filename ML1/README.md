# ML1 — Motorul de decizie Vigil

Primește ce vede camera (`vigil/vision`) și răspunsul șoferului (`vigil/driver_response`), decide cât de grav
e (severitate 0–3) și ce acțiuni urmează, și trimite decizia la ML2 (`vigil/decision`).

```
vigil/vision ─┐
              ├─► normalizare + calibrare ─► fuziune (scor a, încredere) ─► reguli (calea de siguranță) ─┬─► vigil/decision
driver_resp ──┘                                                          │                              │
                                                                         └─► Qwen2.5 1.5B (doar la schimbare) ─► gărzi ─┘
                                 reguli + a ─► matricea de probabilitate (HMM) ─► fuzzy + supervizor ─► vigil/ecu
```

Pe lângă decizie, ML1 publică și comanda pentru ECU (`vigil/ecu`, 5 Hz): viteză, asistență de bandă, distanță,
manevra de oprire pe banda de urgență. Vezi [docs/PROBABILITY_ECU.md](docs/PROBABILITY_ECU.md).

Tot ce ține de ML1 e în acest folder: codul (`ml1/`), testele (`tests/`), documentația (`docs/`), pitch-ul și o
singură comandă, `run_ml1.py`. Sistemul complet (cameră + ML1 + ML2 + dashboard) se pornește din
[`../START/`](../START/README.md).

## Pornire rapidă

Din folderul `ML1/` (sau oricare altul, cu calea `ML1\run_ml1.py`); folosește automat `.venv` de la rădăcină:

```powershell
python run_ml1.py check                    # Ollama, modelul qwen2.5:1.5b, pachete, broker
python run_ml1.py test                     # 137 de teste, ~1 min, fără Ollama și fără broker
python run_ml1.py dashboard --autostart demo --speed 2 --llm   # corp simulat + grafice, se deschide în browser
```

Instalare doar pentru ML1 (dacă nu ai rulat deja `START\1_instalare.bat`):
`pip install -r requirements.txt` și `ollama pull qwen2.5:1.5b`.

## Toate comenzile

| Comandă | Ce face |
|---|---|
| `python run_ml1.py` | lista comenzilor |
| `python run_ml1.py check` | verifică Ollama, modelul, pachetele Python, brokerul MQTT |
| `python run_ml1.py test` | toate testele (`test -v` pentru detalii) |
| `python run_ml1.py dashboard` | dashboard-ul ML1: schelet simulat, scorul `a`, contribuții, decizii, raport 112 (`--llm`, `--autostart demo\|cv_demo\|phone\|drowsy\|chest\|collapse\|privacy\|noisy`, `--speed 2`) |
| `python run_ml1.py scenario collapse` | un scenariu din `tests/data/`, în consolă, cu raportul 112 (`--llm` = cu Qwen) |
| `python run_ml1.py bridge --llm` | ML1 pe MQTT, legat de Vision și ML2 (`--host IP` pentru alt broker) |
| `python run_ml1.py api --llm` | ML1 pe HTTP + SSE pentru serverul Spring Boot, port 8090 (`--host 0.0.0.0`, `--mqtt-out`) — [docs/API.md](docs/API.md) |
| `python run_ml1.py evaluate` | evaluarea pe șoferi nevăzuți → [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) + `docs/figures/` (`--llm`, `--quick`) |
| `python run_ml1.py bench` | latența și validitatea LLM-ului pe calculatorul curent (`--models qwen2.5:1.5b --runs 5`) |
| `python run_ml1.py prob` | matricea de probabilitate a stărilor (emisie, tranziție, P(stare reală \| alarmă)); `--fit` o re-învață |
| `python run_ml1.py ecu` | controlerul fuzzy pentru ECU pe situații tipice (`--bench`, `--export-c` → `ecu/vigil_fuzzy_lut.h`) |
| `python run_ml1.py data` | regenerează `tests/data/*.jsonl` (seed fix) |
| `python run_ml1.py pdf` | recompilează [docs/TECHNICAL_DEEP_DIVE.pdf](docs/TECHNICAL_DEEP_DIVE.pdf) din LaTeX |

Alt model: `set VIGIL_MODEL=llama3.2:1b` înainte de comandă. Detalii și probleme frecvente: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Ce citești, în ordine

| Document | Pentru ce |
|---|---|
| [PITCH.md](PITCH.md) | scriptul de 3 minute + întrebările probabile ale juriului |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | fluxul de date, contractele MQTT, severitate → acțiuni, siguranță, latență |
| [docs/TECHNICAL_DEEP_DIVE.pdf](docs/TECHNICAL_DEEP_DIVE.pdf) | **înțelegerea tehnică**: un cadru urmărit prin tot sistemul, formule, linii de cod, limite (18 pagini; aceeași materie în [.md](docs/TECHNICAL_DEEP_DIVE.md), sursa în [docs/latex/](docs/latex/deep_dive.tex)) |
| [docs/CV_INTEGRATION.md](docs/CV_INTEGRATION.md) | semantica reală a semnalelor CV și ce s-a aliniat în ML1 |
| [docs/ML_METHODS.md](docs/ML_METHODS.md) | metodele ML folosite, legate de cursurile C1–C6 |
| [docs/EVALUATION.md](docs/EVALUATION.md) / [EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) | protocolul de evaluare și rezultatele curente |
| [docs/PROBABILITY_ECU.md](docs/PROBABILITY_ECU.md) | matricea de probabilitate (HMM, decizie Bayes) și controlul fuzzy pentru ECU, cu tabelul C |
| [docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md) | de ce Qwen2.5 1.5B (vs Llama 3.2 1B) și ce s-a măsurat |
| [docs/API.md](docs/API.md) | API-ul HTTP pentru Spring Boot, cu exemple Java |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | toate comenzile pe larg + probleme frecvente |

## Pe scurt

* **Modelul:** `qwen2.5:1.5b` (Q4_K_M, ~1 GB, Apache 2.0), rezervă `llama3.2:1b`. Ținta: controllerul de
  cockpit / DMS din mașină (Snapdragon Cockpit, DRIVE Orin), nu un ECU clasic. Măsurat pe RTX 4050: p50 364 ms,
  100% JSON valid.
* **Siguranța (ISO 26262 / SOTIF):** regulile deterministe din `ml1/rules.py` sunt calea de siguranță. LLM-ul e
  consilier: nuanțează și explică, dar nu poate coborî un caz clar de severitate 3, nu poate urca peste reguli și
  nu poate suna la 112 fără „Sunteți bine?” rămas fără răspuns.
* **Rezultate (simulare în semantica reală CV, șoferi nevăzuți):** 0 urgențe ratate, 0 apeluri 112 false;
  macro-F1 0.75 (praguri vechi) → 0.83 (aliniat la CV + calibrare) → 0.89 (cu LLM).
* **Limita principală:** măsurat pe date simulate în forma reală a mesajelor CV; urmează înregistrări reale
  etichetate ([docs/EVALUATION.md](docs/EVALUATION.md#6-date-reale-pasul-următor)).

## Optimizări pentru mașină

| Optimizare | Unde | Efect |
|---|---|---|
| LLM-ul alege ID-uri de acțiuni, nu scrie text | `ml1/actions.py` | ~25 tokeni ieșire; fraze RO validate, zero halucinații în voce |
| Structured output (`format=` schemă JSON) | `ml1/llm_client.py` | modelul nu poate ieși din meniu |
| Prefix fix (system + exemple) | `ml1/prompt.py` | KV-cache refolosit; per apel doar ~80 tokeni noi (`num_ctx` 2048) |
| `temperature=0`, `num_predict=48`, `keep_alive`, warmup | `ml1/llm_client.py` | determinist, fără reîncărcări |
| Apel doar la schimbarea situației, max 1/s | `ml1/engine.py` | ~11 apeluri / 150 cadre la somnolență, 0 la condus normal |
| Cale rapidă din reguli la severitate 3 | `ml1/engine.py` | siguranța nu așteaptă LLM-ul |
| Histerezis la coborâre, răspunsuri întârziate aruncate | `ml1/engine.py` | fără pâlpâire, fără decizii vechi |
| Raport 112 din șablon | `ml1/reports.py` | instant, doar fapte observate |
| Timeout 2.5 s → reguli | `ml1/validate.py` | plan B garantat |
| Filtru HMM în Python pur, tranziții în cache | `ml1/probability.py` | ~3 µs / cadru; kappa 0.78 → 0.89 față de reguli pe cadru |
| Fuzzy compilat în tabel C (6 KB, uint8) | `ecu/vigil_fuzzy_lut.h` | rulează pe orice ECU, fără LLM și fără numpy |

## Structura

```
ML1/
├── run_ml1.py          ← singura comandă (vezi tabelul de mai sus)
├── requirements.txt    numpy, matplotlib, paho-mqtt
├── PITCH.md
├── ml1/                codul
│   ├── cv_adapter.py   mesaje CV / Spring Boot → forma internă (camelCase, ms, plicuri)
│   ├── calibration.py  baseline personal în primele 30 s (mediană / IQR)
│   ├── fusion.py       scorul de atenție a ∈ [0, 1] + încredere
│   ├── rules.py        scara de severitate deterministă (calea de siguranță)
│   ├── profile.py      pragurile aliniate la codul CV
│   ├── prompt.py, llm_client.py, actions.py, validate.py   LLM-ul și gărzile lui
│   ├── engine.py       DecisionEngine: le leagă pe toate
│   ├── probability.py  matricea de probabilitate a stărilor (HMM) + statistici; prob_matrix.json = matricele învățate
│   ├── ecu_fuzzy.py    controlerul fuzzy + supervizorul pentru ECU (vigil/ecu), exportul în C
│   ├── service.py, mqtt_bridge.py, http_api.py              intrare / ieșire (MQTT, REST + SSE)
│   ├── reports.py      raportul 112 și raportul post-cursă
│   ├── dashboard.py, web/, mock_pose.py                     dashboard-ul ML1
│   └── evaluate.py, bench.py, fake_vision.py, replay.py    evaluare, benchmark, date de test
├── ecu/                vigil_fuzzy_lut.h (generat) + test_lut.c
├── tests/              137 de teste + tests/data (scenarii .jsonl)
└── docs/               documentația, figurile, PDF-ul și sursa LaTeX
```
