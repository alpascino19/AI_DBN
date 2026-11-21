# API-ul HTTP al ML1 (pentru serverul Spring Boot)

> Pentru: cine construiește serverul Spring Boot care va primi cadrele de la camera live și le va trimite
> la ML1. Descrie rutele, formatul mesajelor, ce tolerează ML1 și cum se consumă deciziile din Java.

```
python -m ml1.http_api --host 0.0.0.0 --port 8090 --llm      # pornire (fără --llm: doar reguli)
set ML1_API_TOKEN=secret                                      # opțional (Windows; Linux: export)
```

Baza: `http://<ip-ml1>:8090/api/v1`. Corpul e JSON UTF-8. Cu `ML1_API_TOKEN` setat, toate rutele în afară
de `/health` cer `Authorization: Bearer <token>`.

## Rute

| Metodă | Rută | Corp | Răspuns |
|---|---|---|---|
| POST | `/vision` | un cadru `vigil/vision` **sau** o listă de cadre | `{"accepted", "ts", "decisions": [...]}` (listă: `{"accepted": n, "received": m, "decisions": [...]}`) |
| POST | `/driver-response` | `{"answered": bool, "latencySec"?: num, "text"?: str, "ts"?: ...}` | `{"accepted": true}` sau 400 |
| GET | `/decisions?since=N` | — | `{"decisions": [...]}` cu `seq > N` (max 100) |
| GET | `/decisions/latest` | — | `{"decision": {...} \| null}` |
| GET | `/decisions/stream` | — | Server-Sent Events: `id: <seq>`, `event: decision`, `data: <json>`; comentariu `keepalive` la 15 s |
| GET | `/status` | — | scor `a`, încredere, severitatea curentă, calibrare, contoare |
| POST | `/reset` | — | călătorie nouă / alt șofer: calibrarea pornește de la zero |
| GET | `/health` | — | `{"ok": true}` (fără token) |

Coduri: 200 ok · 400 JSON invalid / câmp lipsă · 401 token greșit · 404 rută necunoscută.
Un cadru respins (duplicat, prea vechi) întoarce tot 200, cu `"accepted": false` și motivul.

## Cadrul de intrare

Forma de referință e `Vision/vigil_vision/schemas.py` (`VisionMessage.to_dict()`). Câmpurile folosite de ML1:

| Câmp | Tip | Observații |
|---|---|---|
| `ts` | număr sau șir | secunde epoch, **milisecunde** (`System.currentTimeMillis()`), microsecunde sau ISO-8601 (`Instant.toString()`); lipsă → ora ML1 |
| `ear`, `perclos`, `blinkRate` | număr / null | `perclos` în semantica CV: fracția din ultimele 60 s cu ochii aproape complet închiși |
| `gazeOnRoad`, `headDrop` | bool / null | |
| `phone` | 0..1 | |
| `breathing` | `{"state": "heavy"\|"normal"\|...}` | |
| `posture` | `{"state": "upright"\|"leaning"\|"slumped", "stillSec": num}` | |
| `events` | listă `{"event", "durationSec", "confidence", ...}` | nume CV: `phone_use`, `hand_on_chest`, `heavy_breathing`, `head_on_wheel`, `eyes_closed`, `microsleep`, `gaze_off_road` (+`direction`), `yawn`, `low_blink_rate`, `hyper_blinking`, `head_drop` |
| `cameraOn`, `faceDetected` | bool | camera oprită = privacy mode (încredere mică, nu „neatenție”) |
| opțional: `perclosRecent`, `recovered`, `closedSec` | | dacă CV le adaugă, ML1 le folosește direct (vezi [CV_INTEGRATION.md](CV_INTEGRATION.md)) |

**Ce tolerează ML1** (ca serverul să nu fie nevoit să transforme nimic):
* nume în **camelCase** (Jackson implicit) sau snake_case, la orice nivel (inclusiv în `events`);
* plicuri `{"type": "...", "payload": {...}}` (sau `data`, `vision`, `message`, `body`);
* câmpuri necunoscute (ignorate), `landmarks` / `bodyLandmarks` (eliminate imediat — **nu le trimite** dacă
  nu e nevoie: un cadru fără ele are ~1 KB în loc de ~30 KB);
* valori lipsă / `NaN` / în afara intervalului (curățate);
* cadre duplicate (același `ts`) și cadre întârziate cu peste 0.5 s față de ultimul (ignorate, numărate în `/status`);
* pauze de peste 30 s în flux (episodul curent se închide; calibrarea rămâne).

## Decizia (ieșire)

```json
{"seq": 7, "ts": 1763720043.6, "a": 0.57, "severity": 3,
 "actions": [{"tool": "play_alarm", "args": {"level": 3}},
             {"tool": "set_esp32", "args": {"level": 3}},
             {"tool": "call_112", "args": {"report": "APEL DE URGENȚĂ AUTOMAT – Vigil (SIMULAT)\n..."}}],
 "reason": "mână pe piept + respirație grea", "source": "rules",
 "safety_notes": [], "fallback_reason": "cale rapidă (reguli, severitate 3)"}
```

`tool` ∈ `speak`, `play_alarm`, `ask_driver`, `set_esp32`, `call_112`, `suggest_break` (contractul cu ML2).
`source` = `llm` sau `rules`. `seq` crește strict: folosește-l pentru `?since=` și pentru reconectarea SSE
(`Last-Event-ID`).

## Recomandări pentru server

* **Ritm:** CV publică ~10 Hz. Trimite fie fiecare cadru, fie loturi de 5–10 cadre (o listă JSON) la
  0.5–1 s; ML1 procesează în ordinea din listă.
* **Deciziile:** abonează-te o singură dată la `/decisions/stream` (push, latență minimă). Ca plasă de
  siguranță, la reconectare citește `/decisions?since=<ultimul seq>`.
* **Călătorie nouă / alt șofer:** `POST /reset` (altfel ML1 folosește calibrarea șoferului anterior).
* **Răspunsul șoferului:** trimite-l la `/driver-response` după fiecare `ask_driver`. Fără el, ML1 tratează
  8 s de tăcere ca „fără răspuns” (corect pentru siguranță, dar poate escalada la 112 un șofer care a
  răspuns, însă răspunsul nu a ajuns).

## Exemple Java (Spring Boot 3)

Trimiterea cadrelor (`RestClient`, Spring 6.1+):

```java
RestClient ml1 = RestClient.builder()
    .baseUrl("http://ml1-host:8090/api/v1")
    .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + token)
    .build();

// VisionFrame = record cu câmpurile din tabel (camelCase e acceptat)
Ml1IngestResult res = ml1.post().uri("/vision")
    .contentType(MediaType.APPLICATION_JSON)
    .body(List.of(frame1, frame2, frame3))        // sau un singur cadru
    .retrieve()
    .body(Ml1IngestResult.class);
```

Primirea deciziilor în timp real (`WebClient` + SSE):

```java
WebClient web = WebClient.builder().baseUrl("http://ml1-host:8090/api/v1")
    .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + token).build();

web.get().uri("/decisions/stream")
    .retrieve()
    .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<Decision>>() {})
    .filter(e -> "decision".equals(e.event()))
    .map(ServerSentEvent::data)
    .retryWhen(Retry.backoff(Long.MAX_VALUE, Duration.ofSeconds(1)))
    .subscribe(decision -> actionService.execute(decision));   // ML2 / dashboard
```

Răspunsul șoferului:

```java
ml1.post().uri("/driver-response").contentType(MediaType.APPLICATION_JSON)
   .body(Map.of("answered", true, "latencySec", 1.4, "text", "da, sunt bine",
                "ts", Instant.now().toString()))
   .retrieve().toBodilessEntity();
```

## Test rapid cu curl

```
curl http://127.0.0.1:8090/api/v1/health
curl -X POST http://127.0.0.1:8090/api/v1/vision -H "Content-Type: application/json" ^
     -d "{\"ts\": 1763720000000, \"ear\": 0.3, \"perclos\": 0.03, \"gazeOnRoad\": true, \"events\": []}"
curl -N http://127.0.0.1:8090/api/v1/decisions/stream
```

Alternativă la HTTP: aceleași mesaje pe MQTT (`python -m ml1.mqtt_bridge`), dacă serverul preferă un broker
(Spring Integration MQTT). Ambele transporturi folosesc același serviciu (`ml1/service.py`), deci aceleași reguli.
