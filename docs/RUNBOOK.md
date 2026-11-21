# Cum rulezi ML1

> Pentru: oricine din echipă care vrea să pornească ML1, să-l testeze sau să-l lege de CV / ML2.
> Toate comenzile se rulează din rădăcina repo-ului (`CodeLap/`). Python 3.11+.

## 1. O singură dată

```
pip install numpy matplotlib          # evaluarea și graficele (restul ML1 e doar biblioteca standard)
pip install paho-mqtt                 # doar pentru puntea MQTT
winget install Ollama.Ollama          # LLM local (Windows); pe Linux/macOS: ollama.com
ollama pull qwen2.5:1.5b              # ~1 GB
```

## 2. Teste

```
python -m unittest -v                 # ~110 teste, ~1 min, fără Ollama și fără broker
```

## 3. Dashboard (corp simulat + grafice)

```
python -m ml1.dashboard               # http://127.0.0.1:8765
python -m ml1.dashboard --autostart cv_demo --speed 2 --llm
```

Scenarii: `demo` (povestea completă din TASKS.md), `cv_demo` (demo-ul echipei CV, mesaje exact ca de
la cameră, fără răspunsuri vocale), plus cele scurte (`phone`, `drowsy`, `chest`, `collapse`, `privacy`,
`noisy`). Treci cu mouse-ul peste grafice ca să derulezi în timp.

## 4. Legătura cu CV și ML2 (MQTT)

```
# terminal 1: broker (ex. Mosquitto)
mosquitto -v
# terminal 2: CV (cameră reală sau date false)
cd Vision && python -m vigil_vision            # sau: python -m vigil_vision --fake
# terminal 3: ML1
python -m ml1.mqtt_bridge --llm                # fără --llm: doar reguli
```

ML1 ascultă `vigil/vision` și `vigil/driver_response`, publică `vigil/decision` și `vigil/ml1/status`.

### Prin HTTP (serverul Spring Boot + camera live)

```
python -m ml1.http_api --host 0.0.0.0 --port 8090 --llm
curl http://127.0.0.1:8090/api/v1/status
```
Contractul și exemplele Java: [API.md](API.md).
Fără ML2 conectat, tăcerea de 8 s după o întrebare e tratată ca „fără răspuns”.

## 5. LLM

```
python -m ml1.bench --models qwen2.5:1.5b --runs 5        # latență + validitate pe hardware-ul curent
python -m ml1.engine tests/data/collapse.jsonl --ollama --report
set VIGIL_MODEL=llama3.2:1b                                # alt model (Windows; Linux: export)
```

## 6. Evaluare

```
python -m ml1.evaluate          # scrie docs/EVALUATION_RESULTS.md + docs/figures/*.png
python -m ml1.evaluate --llm    # + varianta cu LLM
```

## 7. Date de test

```
python -m ml1.fake_vision       # regenerează tests/data/*.jsonl (seed fix)
```

## Probleme frecvente

| Simptom | Cauză / rezolvare |
|---|---|
| `Ollama indisponibil: rulez doar cu reguli fixe` | Ollama nu rulează sau modelul nu e descărcat: `ollama pull qwen2.5:1.5b` |
| primul apel LLM durează ~90 s | încărcarea modelului după repornire; apoi ~0.35 s |
| `Lipsește paho-mqtt` | `pip install paho-mqtt` |
| nicio decizie deși CV trimite date | verifică topicul (`vigil/vision`) și că mesajele au `ts`; `vigil/ml1/status` arată câte mesaje au sosit |
| scenariul `cv_demo` lipsește din dashboard | folderul `Vision/` nu e lângă `ml1/` |
