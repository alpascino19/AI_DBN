# ML 1 — Motorul de decizie

> Documentația completă: [docs/README.md](../docs/README.md) (arhitectură, metode ML, evaluare, alegerea modelului, comenzi).

```
vigil/vision ─┐
              ├─► fusion (a, încredere) ─► rules (podea de siguranță) ─┬─► vigil/decision
driver_resp ──┘                                   │                    │
                                                  └─► LLM (doar la schimbare) ─► validate ─┘
```

## Modelul: `qwen2.5:1.5b` (Q4_K_M, ~1 GB) — rezervă `llama3.2:1b`

Ținta e controllerul de cockpit / DMS din mașină (Snapdragon Cockpit, DRIVE Orin, R-Car),
nu un ECU clasic (microcontroller, fără loc pentru un LLM). Bugetul realist: ~1–1.5 GB, sute de ms.

| | Qwen2.5 1.5B | Llama 3.2 1B | Qwen2.5 3B | Phi-3.5 mini |
|---|---|---|---|---|
| Memorie Q4 | ~1 GB | ~0.8 GB | ~2 GB | ~2.3 GB |
| Licență produs auto | Apache 2.0 | Llama Community | **non-comercială** | MIT |
| JSON structurat | bun | ok | foarte bun | bun |

Alegerea finală se face cu numere, pe hardware-ul țintă: `python -m ml1.bench --models ...`.

**Siguranță (ISO 26262):** regulile (`rules.py`) sunt calea de siguranță — deterministe, testate.
LLM-ul e consilier fără rating de siguranță (QM): nuanțează, explică, dar nu poate coborî un caz
clar de severitate 3 și nu poate suna la 112 fără dovezi.

## Optimizări

| Optimizare | Unde | Efect |
|---|---|---|
| LLM-ul alege ID-uri, nu scrie text | `actions.py` | ~25 tokeni ieșire în loc de ~80; fraze RO validate, zero halucinații în voce |
| Structured output (`format=` schema) | `llm_client.py` | modelul nu poate ieși din meniu |
| Prefix fix (system + exemple) | `prompt.py` | KV-cache refolosit; per apel doar ~80 tokeni noi |
| Context = o linie de cifre, fără landmarks | `prompt.py` | prompt ~760 tokeni, `num_ctx=1024` |
| `temperature=0`, `num_predict=48`, `keep_alive`, warmup | `llm_client.py` | determinist, fără încărcări repetate |
| Apel doar la schimbarea situației, max 1/s | `engine.py` | ~11 apeluri / 150 cadre la somnolență, 0 la condus normal |
| Cale rapidă din reguli la severitate 3 | `engine.py` | siguranța nu așteaptă LLM-ul |
| Răspunsuri întârziate aruncate, histerezis la coborâre | `engine.py` | fără pâlpâire, fără decizii vechi |
| 112 o dată, fără întrebări repetate, tăcere 8 s = fără răspuns | `engine.py` | funcționează și dacă STT-ul cade |
| Raport 112 din șablon | `reports.py` | instant, doar fapte observate |
| Timeout 2.5 s → reguli | `validate.py` | plan B garantat |

## Dashboard (corp mock + grafice în timp real)

```
python -m ml1.dashboard                      # deschide http://127.0.0.1:8765
python -m ml1.dashboard --autostart demo --speed 2 --llm     # pornește direct demo-ul cu LLM
```
În pagină alegi scenariul, viteza (0.5×–8×) și dacă folosești LLM-ul local (Ollama trebuie să ruleze),
apoi „Pornește”. Scenariul `demo` (~2.3 min) parcurge toată povestea din TASKS.md.

Ce vezi: scheletul șoferului (reacționează la telefon, mână pe piept, cap căzut, respirație grea, ochi
închiși, căscat), scorul `a` cu banda de severitate și apelurile 112, evenimentele, PERCLOS/EAR/telefon,
contribuția fiecărui semnal, jurnalul deciziilor (sursă LLM/reguli, latență, gardurile care au intervenit,
raportul 112) și raportul post-cursă la final. **Treci cu mouse-ul peste orice grafic** ca să derulezi în
timp: corpul, contribuțiile și valorile arată momentul ales.

Corpul vine din `ml1/mock_pose.py` (puncte normalizate 0..1). Când CV 2 are MediaPipe Pose, același format
de puncte îl înlocuiește.

## Rulare

```
python -m unittest -v                                 # toate testele (fără Ollama)
python -m ml1.engine tests/data/collapse.jsonl --report   # doar reguli
ollama pull qwen2.5:1.5b
python -m ml1.engine tests/data/collapse.jsonl --ollama --report
python -m ml1.bench --models qwen2.5:1.5b llama3.2:1b --runs 5
```
