# Matricea de probabilitate și controlul fuzzy pentru ECU

Două straturi noi după motorul de decizie, ambele fără LLM și ieftine (µs pe cadru):

```
reguli (sev 0–3) + scor a ─► matricea de probabilitate (HMM) ─► P(atent, distras, obosit, urgență)
                                                                  │
            a, încredere, severitatea publicată, 112 ────────────┴─► fuzzy Mamdani + supervizor ─► vigil/ecu
```

Cifrele de mai jos vin din `python run_ml1.py evaluate` (38 de șoferi simulați, seed 0); tabelele complete
sunt în [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (secțiunile „Matricea de probabilitate” și „Controlul fuzzy”).

## 1. Matricea de probabilitate — [`ml1/probability.py`](../ml1/probability.py)

**Problema.** Severitatea publicată e o decizie, nu o probabilitate. Pe test, un „2” publicat înseamnă oboseală
reală doar în 58% din cadre: în restul, șoferul e deja atent, iar alarma rămâne activă prin histerezis.
Pentru o alarmă sonoră e acceptabil; pentru a frâna mașina, nu.

**Modelul.** Stările ascunse sunt severitatea reală 0–3. Observația pe cadru are 12 simboluri:
(severitatea din reguli) × (a ≥ 0.75 | 0.5–0.75 | < 0.5).

| Matrice | Ce conține | De unde |
|---|---|---|
| emisie `E[s, o] = P(o \| s)` | cât de des „spun” regulile `o` când starea reală e `s` (matricea de confuzie normalizată pe rânduri) | numărată pe train, netezire Laplace |
| tranziție `T(dt) = exp(Q·dt)` | probabilitatea de a trece dintr-o stare în alta în `dt` secunde (lanț Markov) | generatorul `Q` estimat pe train; durate medii: atent 112 s, distras 8 s, obosit 30 s, urgență 251 s |

**Filtrul** (pe fiecare cadru, Python pur, ~3 µs):
`b ← normalizare( E[:, o] ⊙ (T(dt)ᵀ · b) )`. Când camera e oprită (încredere < 0.3), se face doar
predicția: lipsa observației nu e dovadă că șoferul e atent.

**Condiția de decizie (Bayes cu cost asimetric):** `sev* = argmin_p Σ_s b[s] · COST[s, p]`, cu același cost ca în
evaluare (a rata o urgență costă de 30 de ori mai mult decât o alarmă falsă). Exemplu: `b = [0.8, 0, 0, 0.2]`
are cea mai probabilă stare 0, dar decizia Bayes e 3.

**Eficiența** (pe cadru, test / shift):

| | kappa | MCC | acuratețe echilibrată | cost / cadru |
|---|---|---|---|---|
| reguli + calibrare | 0.776 / 0.784 | 0.788 / 0.796 | 0.943 / 0.944 | 0.090 / 0.101 |
| HMM (cost minim) | **0.888 / 0.881** | **0.892 / 0.884** | **0.983 / 0.969** | **0.057 / 0.088** |

Recall-ul pentru urgențe rămâne 1.00 în ambele variante. Statisticile pe clasă (sensibilitate, specificitate,
VPP, VPN, LR+, LR−) și `P(starea reală | severitatea publicată)` sunt generate automat în raport.

**Ce NU se schimbă:** deciziile de pe `vigil/decision` (alarme, întrebări, 112) vin în continuare din reguli +
LLM. Matricea alimentează ECU-ul și dashboard-ul, nu înlocuiește calea de siguranță.

Matricele învățate stau în [`ml1/prob_matrix.json`](../ml1/prob_matrix.json) (le rescrie evaluarea completă;
fără fișier se folosesc valori implicite prudente). `python run_ml1.py prob` le afișează.

## 2. Controlul fuzzy pentru ECU — [`ml1/ecu_fuzzy.py`](../ml1/ecu_fuzzy.py)

**Intrări:** scorul `a`, riscul `0.5·P(obosit) + P(urgență)` din matrice și încrederea măsurării.
**Ieșiri:** fracția din viteza setată, autoritatea asistenței de bandă (0 = șoferul, 1 = sistemul) și distanța
de siguranță în secunde. Mamdani: ȘI = min, agregare = max, defuzzificare prin centroid. Opt reguli citibile
acoperă orice combinație de intrări (verificat în teste).

| Situație | Viteză | Asistență | Distanță | Mod |
|---|---|---|---|---|
| atent, cameră OK | 100% | 0.00 | 1.8 s | NORMAL |
| telefon în mână | 73% | 0.50 | 2.5 s | CAUTION |
| somnolent | 60% | 1.00 | 3.3 s | ASSIST |
| urgență medicală | 30% → MRM | 1.00 | 3.3 s | ASSIST + avarii → MRM |
| cameră oprită | 82% | 0.50 | 2.5 s | CAUTION |

**Supervizorul determinist** (deasupra fuzzy, aceeași filozofie ca `rules.py`):
* severitatea 3 publicată: viteză ≤ 50%, avarii și asistență de bandă 100%, imediat;
* **MRM** (bandă de urgență, viteză → 0, 2.5 m/s²) după apelul 112 sau după 10 s de severitate 3
  **confirmată de matrice** (P(urgență) ≥ 0.5). Severitatea publicată singură nu ajunge: are histerezis și
  rămânea 3 încă 14 s după un micro-somn de 3 s, ceea ce a produs un MRM fals în evaluare;
* la severitatea 2, viteza o decide fuzzy-ul (riscul include deja regulile). Un plafon fix la 85% încetinea
  mașina în ~10% din condusul normal, din cauza alarmelor false;
* rate limitate: 0.5 m/s² la încetinire ușoară (o privire de 1 s scade viteza cu < 2 km/h), 2 m/s² la risc mare,
  1 m/s² la revenire.

**Eficiența** (test / shift, viteză setată 100 km/h):

| Metrică | Test | Shift |
|---|---|---|
| viteză medie: atent · distras · obosit · urgență | 97% · 87% · 64% · 20% | 97% · 86% · 63% · 19% |
| condus normal încetinit sub 90%, la ≥ 15 s după un episod | 2.9% | 2.1% |
| MRM atins în urgențe | 6/6, mediană 8.6 s | 7/7, mediană 8.2 s |
| MRM-uri false | 0 | 0 |
| cost pe pas | ~60 µs (Python) | |

## 3. Pe ECU: tabelul C — [`ecu/vigil_fuzzy_lut.h`](../ecu/vigil_fuzzy_lut.h)

`python run_ml1.py ecu --export-c` compilează controlerul fuzzy într-un tabel `uint8` de 15×15×9 puncte
(6 KB în flash) cu interpolare triliniară în C99: fără alocare, fără recursivitate, timp mărginit. Nodurile
includ colțurile funcțiilor de apartenență, unde ieșirea are „genunchi”. Eroarea față de inferența exactă:
medie < 0.7%, maximă 0.07–0.14 în puncte izolate (pentru un maxim < 0.04 ar fi nevoie de ~70 KB).

Testul `tests/test_probability_ecu.py` compilează header-ul cu `gcc -Wall -Wextra -Werror` (dacă gcc există) și
compară ieșirea C cu Python. Supervizorul rămâne în stratul de siguranță al ECU (`EcuController`); header-ul
conține doar partea fuzzy.

## 4. Contractul `vigil/ecu` (ML1 → ECU, dashboard; 5 Hz)

```json
{"ts": 1013.2, "mode": "MRM", "speed_factor": 0.0, "target_kmh": 0.0, "speed_ref_kmh": 71.0, "decel_mps2": 2.5,
 "lka_authority": 1.0, "headway_s": 3.5, "hazards": true, "lane": "shoulder",
 "inputs": {"a": 0.08, "risk": 0.99, "confidence": 0.6, "severity": 3},
 "fuzzy": {"speed": 0.3, "authority": 1.0, "headway": 3.3},
 "state": {"p_state": [0.0, 0.0, 0.01, 0.99], "p_risk": 1.0, "expected_severity": 2.99, "map_state": 3,
           "bayes_severity": 3, "observed": true}}
```

`mode`: `NORMAL` | `CAUTION` (viteză < 95% sau distanță mărită) | `ASSIST` (asistența ≥ 60%) | `MRM`.

## Limite

* Matricele sunt învățate pe același simulator care produce datele de test: rezultatele compară variante între
  ele, nu prezic comportamentul pe drum. Pe date reale se re-învață emisia (care depinde de detectorul CV), nu
  codul.
* Riscul ponderat (`0.5·P(obosit)`) și pragurile fuzzy sunt alese de mână, pentru autostradă. Pentru oraș,
  viteza setată și regulile trebuie revăzute.
* Nu există încă o legătură cu o mașină reală sau cu `highway-env`: `vigil/ecu` e comanda, nu bucla închisă.
