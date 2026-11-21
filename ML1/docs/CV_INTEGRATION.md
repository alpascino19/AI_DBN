# ML1 ↔ CV: ce am aliniat la codul real și ce cerem echipei CV

> Pentru: echipa CV (CV1, CV2) și ML1. Ce a descoperit ML1 citind și rulând `Vision/`, ce s-a schimbat în ML1
> ca urmare, și patru câmpuri mici pe care, dacă le publică CV, ML1 devine mai rapid și mai precis.

## 1. Semantica semnalelor, așa cum le produce CV (și cum le folosește acum ML1)

| Semnal | Cum îl calculează CV | Ce presupunea ML1 înainte | Ce face ML1 acum |
|---|---|---|---|
| `perclos` | fracția de timp cu ochii **aproape complet** închiși (scor de închidere > 0.72) pe **60 s**; `null` în primele 10 s; prag de închidere calibrat per șofer (`eyes.py`) | normal 0.06, oboseală ≥ 0.35 (din date false proprii) | normal ≤ 0.05, oboseală ≥ **0.15** (ca `alerts.py`), scor ochi 1 → 0 între 0.05 și 0.30 |
| revenirea din oboseală | CV folosește intern ultimele **15 s** (`recovered`), dar nu le publică | — | estimează singur închiderea pe 15 s din EAR, folosită doar dacă e consistentă cu `perclos` de la CV |
| `eyes_closed` (eveniment) | activ cât ochii sunt închiși; tracker-ul **unește** activările separate de < 1 s → durata = episodul, nu închiderea continuă | durata = închidere continuă | ≥ 1.5 s → micro-somn (2); ≥ 3 s **și continuu în EAR** → 3 |
| ochi închiși la căscat / privit în jos | `alerts.py` le exclude din alerte | erau micro-somn | excluse (și episodul de ochi închiși care a început în timpul lor) |
| `head_drop` | față (pitch / privit în jos cu ochii închiși) **SAU** corp prăbușit (`pipeline._head_drop`) | cerea și PERCLOS ≥ 0.6 | cap căzut + ochi închiși / `head_on_wheel` / `posture: slumped` nemișcat — fără PERCLOS |
| `low_blink_rate`, `hyper_blinking`, `frequent_yawning` | detecții CV1 | necunoscute | penalizări mici în scor; clipit rar ≥ 5 s sau căscat des → oboseală |
| `gaze_off_road` | eveniment cu `direction` (left/right/down) | fără direcție | `direction: down` = posibil telefon → nu e micro-somn |

Efect măsurat (simulatorul ML1 a fost rescris în semantica de mai sus, inclusiv tracker-ul cu pauze de 1 s):
pe șoferi de test nevăzuți, macro-F1 **0.75 → 0.83**, costul erorilor aproape înjumătățit, toate apelurile
112 corecte (detalii în [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)).

## 2. Două surse de alarmă: de coordonat

`face/alerts.py` + `face/monitor.py` (CV) bipăie local la nivelurile 1–3, independent de ML1/ML2. În mașină
șoferul ar auzi **două sisteme** pentru același eveniment, cu mesaje diferite. Propunere:
* în demo, `monitor.py` rulează cu `--mute` (sau `--min-sound 3`) cât ML1 + ML2 sunt pornite;
* alertele CV rămân vizibile pe dashboard și sunt „plan B” doar dacă ML1 nu răspunde (ex. `vigil/ml1/status`
  lipsește de > 3 s).

## 3. Ce cerem CV (opțional, mic, dar util)

Toate se pot pune în `VisionMessage` ca extra (schema permite); ML1 le folosește automat dacă apar.

| Câmp | De unde există deja în CV | De ce ajută ML1 |
|---|---|---|
| `perclos_recent` și/sau `recovered` | `EyeState.perclos_recent`, `EyeState.recovered` | revenire rapidă din oboseală fără estimarea proprie ML1 |
| `closed_sec` | `EyeState.closed_sec` (în `FaceOutput.signals`) | durata **continuă** a închiderii: elimină ambiguitatea episoadelor unite de tracker |
| `off_road_ratio` | `HeadState.off_road_ratio` | distragere repetată (priviri dese) chiar fără un episod lung |
| `direction` | `HeadState.direction` | telefon / privit în jos, chiar fără evenimentul `gaze_off_road` |

## 4. Mici observații de format

* `ts` = `time.time()` (epoch secunde): perfect. Dacă vreodată vine prin serverul Spring în ms sau ISO,
  ML1 le acceptă.
* Pe camera pierdută CV trimite `camera_on: false` cu evenimentele încă active: ML1 tratează corect (privacy).
* `landmarks` (468 de puncte) nu sunt folosite de ML1; dacă banda contează, pot lipsi din mesajul către ML1.
