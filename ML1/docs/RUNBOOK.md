# Cum rulezi ML1

> Pentru: oricine din echipă care vrea să pornească ML1, să-l testeze sau să-l lege de CV / ML2.
> Comenzile se rulează din folderul `ML1/`. `run_ml1.py` folosește automat `.venv` de la rădăcina repo-ului.
> Sistemul complet (cameră + ML1 + ML2 + dashboard React) se pornește din [`START/`](../../START/README.md).

## 1. O singură dată

Totul deodată (Vision + ML1 + ML2 + programe): `START\1_instalare.bat`. Doar ML1:

```powershell
pip install -r requirements.txt       # numpy, matplotlib, paho-mqtt (nucleul e doar biblioteca standard)
winget install Ollama.Ollama          # LLM local; pe Linux/macOS: ollama.com
ollama pull qwen2.5:1.5b              # ~1 GB
python run_ml1.py check               # verifică tot
```

## 2. Teste

```powershell
python run_ml1.py test                # 114 teste, ~1 min, fără Ollama și fără broker
python run_ml1.py test -v             # cu numele fiecărui test
```

## 3. Dashboard ML1 (corp simulat + grafice)

```powershell
python run_ml1.py dashboard                                  # http://127.0.0.1:8765
python run_ml1.py dashboard --autostart cv_demo --speed 2 --llm
```

Scenarii: `demo` (povestea completă din TASKS.md), `cv_demo` (demo-ul echipei CV, mesaje exact ca de
la cameră, fără răspunsuri vocale), plus cele scurte (`phone`, `drowsy`, `chest`, `collapse`, `privacy`,
`noisy`). Treci cu mouse-ul peste grafice ca să derulezi în timp.

Acesta e dashboard-ul de dezvoltare al ML1 (simulare, fără broker). Dashboard-ul live al întregului sistem
e cel React din [`dashboard/`](../../dashboard/README.md), pornit automat de `START\3_pornire_demo.bat`.

## 4. Un scenariu în consolă

```powershell
python run_ml1.py scenario collapse          # doar reguli, cu raportul 112 complet
python run_ml1.py scenario collapse --llm    # cu Qwen
python run_ml1.py scenario C:\cale\inregistrare.jsonl
```

## 5. Legătura cu CV și ML2 (MQTT)

Cel mai simplu: `START\3_pornire_demo.bat` (date false) sau `START\4_pornire_camera.bat` (camera reală).
De mână, în terminale separate:

```powershell
cd Vision; ..\.venv\Scripts\python -m vigil_vision --fake     # CV (fără --fake: camera reală)
cd ML1;    python run_ml1.py bridge --llm                     # ML1 (fără --llm: doar reguli)
cd ML2;    ..\.venv\Scripts\python executor.py                # ML2 (voce, alarmă, ESP32)
```

ML1 ascultă `vigil/vision` și `vigil/driver_response`, publică `vigil/decision` și `vigil/ml1/status`.

### Prin HTTP (serverul Spring Boot + camera live)

```powershell
python run_ml1.py api --host 0.0.0.0 --port 8090 --llm
curl http://127.0.0.1:8090/api/v1/status
```
Contractul și exemplele Java: [API.md](API.md). Cu `--mqtt-out`, deciziile merg și la ML2 prin broker
(așa pornește `START\5_pornire_spring.bat`). Fără ML2 conectat, tăcerea de 8 s după o întrebare e
tratată ca „fără răspuns”.

## 6. LLM

```powershell
python run_ml1.py bench --models qwen2.5:1.5b --runs 5     # latență + validitate pe hardware-ul curent
set VIGIL_MODEL=llama3.2:1b                                 # alt model (PowerShell: $env:VIGIL_MODEL="llama3.2:1b")
```

## 7. Evaluare și documentație

```powershell
python run_ml1.py evaluate          # scrie docs/EVALUATION_RESULTS.md + docs/figures/*.png
python run_ml1.py evaluate --llm    # + varianta cu LLM
python run_ml1.py pdf               # recompilează docs/TECHNICAL_DEEP_DIVE.pdf (MiKTeX / TeX Live)
python run_ml1.py data              # regenerează tests/data/*.jsonl (seed fix)
```

## Probleme frecvente

| Simptom | Cauză / rezolvare |
|---|---|
| `Ollama indisponibil: rulez doar cu reguli fixe` | Ollama nu rulează sau modelul nu e descărcat: `ollama pull qwen2.5:1.5b` |
| primul apel LLM durează mult | încărcarea modelului după repornire; apoi ~0.35 s |
| `Lipsește paho-mqtt` | `pip install -r requirements.txt` |
| nicio decizie deși CV trimite date | verifică topicul (`vigil/vision`) și că mesajele au `ts`; `vigil/ml1/status` arată câte mesaje au sosit |
| scenariul `cv_demo` lipsește din dashboard | folderul `Vision/` trebuie să fie lângă `ML1/` (în rădăcina repo-ului) |
| `UnicodeEncodeError` în consolă | rulează prin `run_ml1.py` (setează UTF-8) sau `set PYTHONUTF8=1` |
