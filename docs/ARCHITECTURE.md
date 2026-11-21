# Arhitectura ML1 — motorul de decizie Vigil

> Pentru: echipa Vigil (CV1, CV2, ML1, ML2) și juriu. Ce face ML1, cum se leagă de celelalte module,
> unde sunt deciziile de siguranță.

## 1. Fluxul de date

```
            vigil/vision (10 Hz)                          vigil/decision
 CV1 + CV2 ───────────────────► ┌──────────────────────────────────────┐ ───────────────► ML2 (voce, alarmă,
 (cameră)                       │ ML1                                  │                   ESP32, 112 simulat)
                                │  1. normalize_vision  (cv_adapter)   │ vigil/ml1/status
            vigil/driver_response│  2. DriverBaseline    (calibration) │ ───────────────► dashboard
 ML2 ──────────────────────────►│  3. AttentionFusion   (fusion)       │
 (răspunsul șoferului)          │  4. rule_severity     (rules)  ◄── calea de siguranță
                                │  5. LLM Qwen2.5 1.5B  (prompt, llm_client) ◄── doar la schimbare
                                │  6. decide/apply_safety (validate)   │
                                │  7. DecisionEngine    (engine)       │
                                └──────────────────────────────────────┘
```

| Pas | Modul | Ce face | De ce |
|---|---|---|---|
| 1 | [`cv_adapter.py`](../ml1/cv_adapter.py) | aduce mesajul CV la forma ML1: `phone_use`→`phone_in_hand`, `breathing`/`posture` → `breathing_state`/`posture_state`, decupare la intervale fizice, valori ne-finite → `None`, elimină landmark-urile | echipa CV folosește alte nume; datele murdare nu trebuie să ajungă la reguli/LLM |
| 2 | [`calibration.py`](../ml1/calibration.py) | învață din primele 30 s valorile normale ale șoferului (mediană/IQR pentru EAR, clipit, PERCLOS) | un șofer cu ochi mici nu trebuie considerat permanent obosit |
| 3 | [`fusion.py`](../ml1/fusion.py) | semnale → scor `a ∈ [0,1]` (medie ponderată + plafoane + netezire asimetrică) + încredere | un singur număr pentru dashboard, MPC și LLM |
| 4 | [`rules.py`](../ml1/rules.py) | severitate 0–3 deterministă | **calea de siguranță**: testabilă, explicabilă, nu depinde de LLM |
| 5 | [`prompt.py`](../ml1/prompt.py), [`llm_client.py`](../ml1/llm_client.py) | LLM local alege severitatea și acțiunile dintr-un meniu fix | raționament contextual (durate, combinații, răspunsul șoferului) |
| 6 | [`validate.py`](../ml1/validate.py) | validează răspunsul LLM; gărzi: nu coboară sub reguli (decât o treaptă, niciodată sub 3), **nu urcă peste reguli**, 112 doar cu dovezi, acțiuni coerente | LLM-ul e consilier, nu decident final |
| 7 | [`engine.py`](../ml1/engine.py) | orchestrare: apel LLM doar la schimbare, cale rapidă la severitate 3, histerezis, fără întrebări/112 repetate; estimează închiderea ochilor pe 15 s | latență, stabilitate, fără spam vocal |
| — | [`profile.py`](../ml1/profile.py) | pragurile care depind de semantica CV (PERCLOS pe 60 s, ochi închiși 1.5 / 3 s), comparate în evaluare | o singură sursă când CV schimbă definiția |
| — | [`service.py`](../ml1/service.py) | serviciul independent de transport: ordine, duplicate, timestamp-uri, flux de decizii | MQTT și HTTP folosesc aceleași reguli |
| — | [`http_api.py`](../ml1/http_api.py), [`mqtt_bridge.py`](../ml1/mqtt_bridge.py) | transporturi: REST + SSE pentru serverul Spring Boot, MQTT pentru broker | vezi [API.md](API.md) |

## 2. Contracte MQTT

| Topic | De la → la | Conținut | Implementat în |
|---|---|---|---|
| `vigil/vision` | CV → ML1 | `schemas.VisionMessage.to_dict()` | `Vision/vigil_vision/schemas.py` |
| `vigil/driver_response` | ML2 → ML1 | `{"answered": bool, "latency_sec": float\|null, "text": str\|null, "ts"?: float}` | ML2 |
| `vigil/decision` | ML1 → ML2, dashboard | `{"a", "severity", "actions": [{"tool","args"}], "reason", "source", "ts", "latency_ms"?, "safety_notes"?}` | [`mqtt_bridge.py`](../ml1/mqtt_bridge.py) |
| `vigil/ml1/status` | ML1 → dashboard | `{"a", "confidence", "critical", "calibration", "stats", ...}` la 1 Hz | [`mqtt_bridge.py`](../ml1/mqtt_bridge.py) |

Aceleași mesaje pot veni și prin HTTP de la serverul Spring Boot (camera live): `POST /api/v1/vision`, decizii pe
`GET /api/v1/decisions/stream` (SSE). Contractul complet: [API.md](API.md).

### Cum citește ML1 mesajul CV

| Câmp CV | Folosit de ML1 ca | Observații |
|---|---|---|
| `ear`, `perclos`, `blink_rate` | ochi / clipit în fuziune; PERCLOS corectat cu calibrarea în reguli | `None` când fața nu se vede |
| `gaze_on_road`, `head_drop` | privire, cap | |
| `face_detected: false` (cameră pornită) | după **≥ 1 s** continuu: „nu privește drumul” | lipsa feței e informație; un cadru pierdut nu |
| `phone` (0..1) | scor telefon | |
| `events[].event` | `phone_use`→`phone_in_hand`; `hand_on_chest`, `heavy_breathing`, `head_on_wheel`, `eyes_closed`, `head_drop`, `yawn` | durata și încrederea contează |
| `breathing.state == "heavy"` | respirație grea (pentru „mână pe piept + respirație grea”) | completează `breathing` pe `hand_on_chest` |
| `posture.state == "slumped"` + `still_sec ≥ 2` | prăbușit → severitate 3 | |
| `landmarks`, `body_landmarks` | ignorate (nu ajung la LLM) | |

## 3. Severitate și acțiuni

| Severitate | Când (reguli) | Acțiuni standard |
|---|---|---|
| 0 | atent; semn medical scurt (< 3 s) doar observat | — |
| 1 | telefon; privire în afara drumului ≥ 2 s; priviri dese (`off_road_ratio` ≥ 0.5, dacă CV îl trimite); `a < 0.75` **de cel puțin 4 s** | `speak` (frază fixă) |
| 2 | PERCLOS ≥ **0.15** (semantica CV; pe 15 s când există, altfel pe 60 s, corectat cu calibrarea), micro-somn, **ochi închiși ≥ 1.5 s** (nu la căscat / privit în jos), căscat des, clipit rar ≥ 5 s, căscat + PERCLOS ≥ 0.12, `a < 0.5`; mână pe piept dar răspuns coerent | alarmă 2, ESP32 2, pauză, „Doriți o pauză?” |
| 3 | cap căzut + ochi închiși, `head_on_wheel` ≥ 2 s, prăbușit nemișcat; **ochi închiși ≥ 3 s continuu**; mână pe piept ≥ 3 s + respirație grea; ochi închiși + fără răspuns; fără răspuns + `a < 0.5` | alarmă 3, ESP32 3, „Sunteți bine?”; **112 doar dacă nu răspunde / spune că nu se simte bine** |

## 4. Arhitectura de siguranță

Vigil e un sistem cu componentă ML într-o mașină, deci urmează două idei din standardele auto:

* **ISO 26262 (siguranță funcțională):** partea care poate face rău (alarmă urgentă, 112) e decisă de
  cod determinist și testat (`rules.py` + gărzile din `validate.py`). LLM-ul nu are rating de siguranță
  (QM): poate nuanța, dar nu poate coborî un caz clar de severitate 3, nu poate ridica o alertă fără semn
  în date și nu poate suna la 112 fără dovezi.
* **ISO 21448 (SOTIF, limitele de performanță ale senzorilor/ML):** ne interesează situațiile în care
  sistemul e „corect” dar senzorii mint (fața pierdută, iluminare, ochi mici). De aici: calibrarea
  personală, persistența în timp (fața lipsă ≥ 1 s, `a` scăzut ≥ 4 s), încrederea scăzută fără cameră și
  evaluarea separată pe un domeniu „shift” ([EVALUATION.md](EVALUATION.md)).

Garanții verificate de teste:

| Garanție | Unde |
|---|---|
| severitatea 3 nu așteaptă LLM-ul (cale rapidă din reguli) | `engine.py`, `test_engine.py` |
| 112 cel mult o dată pe episod, doar după „Sunteți bine?” fără răspuns / cu răspuns de suferință | `engine.py`, `rules.wants_112` |
| tăcerea ≥ 8 s după o întrebare = „fără răspuns” (funcționează și fără ML2/STT) | `engine._context` |
| un semn grav nou în același episod (oboseală → prăbușire) pornește din nou verificarea | `engine.on_vision` |
| LLM lent (> 2.5 s), căzut sau cu JSON invalid → reguli | `validate.decide` |
| decizii LLM întârziate (situația s-a schimbat) sunt aruncate | `engine._llm_job` |

## 5. Bugetul de latență (măsurat pe laptopul de demo, RTX 4050)

| Etapă | Timp |
|---|---|
| normalizare + calibrare + fuziune + reguli | < 1 ms / cadru |
| LLM Qwen2.5 1.5B (doar la schimbare) | mediană 0.25–0.36 s, p95 0.38–0.45 s (rulări diferite) |
| timeout LLM → reguli | 2.5 s |
| detecția urgenței după semn (simulare) | 0.2 s (cap căzut vizibil), 2 s (fața pierdută → `head_on_wheel`), 3 s (mână pe piept: pragul ML1 pe durata evenimentului; CV îl raportează de la 2 s) |
| 112 după debutul urgenței (simulare) | ~8–11 s (8 s așteptare răspuns la „Sunteți bine?”) |

## 6. Rulare

Vezi [RUNBOOK.md](RUNBOOK.md).
