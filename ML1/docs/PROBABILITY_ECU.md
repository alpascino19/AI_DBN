# Matricea de probabilitate și controlul fuzzy pentru ECU

Două straturi noi după motorul de decizie, ambele fără LLM și ieftine (µs pe cadru):

```
reguli (sev 0–3) + scor a ─► matricea de probabilitate (HMM) ─► P(atent, distras, obosit, urgență)
                                                                  │
            a, încredere, severitatea publicată, 112 ────────────┴─► fuzzy Mamdani + supervizor ─► vigil/ecu
```

Cifrele de mai jos vin din `python run_ml1.py evaluate` (46 de șoferi simulați + 50 de urgențe dedicate, seed 0);
tabelele complete sunt în [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (secțiunile „Matricea de probabilitate”,
„Situațiile noi din Vision” și „Controlul fuzzy”).

## 1. Matricea de probabilitate — [`ml1/probability.py`](../ml1/probability.py)

**Problema.** Severitatea publicată e o decizie, nu o probabilitate. Pe test, un „2” publicat înseamnă oboseală
reală doar în 43% din cadre: în restul, șoferul e deja atent, iar alarma rămâne activă prin histerezis.
Pentru o alarmă sonoră e acceptabil; pentru a frâna mașina, nu.

**Modelul.** Stările ascunse sunt severitatea reală 0–3. Observația pe cadru are 15 simboluri:
(treapta regulilor: 0, 1, 2, 3 sau **0+**) × (a ≥ 0.75 | 0.5–0.75 | < 0.5). **0+** = regulile spun încă 0, dar
văd un semn medical sub prag (mână pe piept). Vision are analiza respirației oprită implicit, deci primele 6 s ale
unei urgențe cardiace arată pentru reguli ca o privire în altă parte; fără 0+, matricea învăța că orice privire
poate fi urgență (cadre „atent” prezise 3, cost mai mare decât al regulilor).

| Matrice | Ce conține | De unde |
|---|---|---|
| emisie `E[s, o] = P(o \| s)` | cât de des „spun” regulile `o` când starea reală e `s` (matricea de confuzie normalizată pe rânduri) | numărată pe train, netezire Laplace, fiecare cadru cântărit cu acoperirea camerei |
| tranziție `T(dt) = exp(Q·dt)` | probabilitatea de a trece dintr-o stare în alta în `dt` secunde (lanț Markov) | generatorul `Q` estimat pe train; durate medii: atent 106 s, distras 6 s, obosit 30 s, urgență 208 s |

**Filtrul** (pe fiecare cadru, Python pur, ~4–6 µs), cu evidența numărată pe **secundă**, nu pe cadru:
`b ← normalizare( E[:, o]^w ⊙ (T(dt)ᵀ · b) )`, `w = min(1, dt · evidence_hz) · min(1, acoperire / 0.7)`.
* **rata**: datele vin la 10 Hz de la `merge.py` (două laptopuri), 15–30 fps de la o cameră locală, iar matricele
  sunt învățate la 5 Hz. Filtrul vechi număra fiecare cadru întreg: aceleași date la 10 Hz îi schimbau credința
  cu până la 0.88; acum cu 0.02.
* **`evidence_hz` = 1 observație independentă pe secundă**, ales pe validare după log-loss (0.181, față de 0.293
  cu fiecare cadru numărat): cadrele vecine ale aceluiași gest nu sunt independente, iar la 5 observații/s o mână
  pe piept de o secundă devenea „urgență sigură” și ECU-ul frâna la 48 km/h pentru o centură ajustată.
* **acoperirea camerei** (`FusionResult.camera_cov`): ambele laptopuri ≈ 1, doar fața ≈ 0.8, doar corpul ≈ 0.45.
  Un cadru cu o singură cameră mută credința mai puțin, dar o mută. Vechiul prag (încredere ≥ 0.3) ignora
  complet cadrele doar cu corpul, deci o prăbușire văzută doar de laptopul 3 nu ridica P(urgență).
* camera oprită (acoperire 0): doar predicția; lipsa observației nu e dovadă că șoferul e atent.

**Condiția de decizie (Bayes cu cost asimetric):** `sev* = argmin_p Σ_s b[s] · COST[s, p]`, cu același cost ca în
evaluare (a rata o urgență costă de 30 de ori mai mult decât o alarmă falsă). Exemplu: `b = [0.8, 0, 0, 0.2]`
are cea mai probabilă stare 0, dar decizia Bayes e 3.

**Eficiența** (pe cadru, test / shift):

| | kappa | MCC | acuratețe echilibrată | cost / cadru |
|---|---|---|---|---|
| reguli + calibrare | 0.789 / 0.707 | 0.803 / 0.729 | 0.960 / 0.931 | 0.099 / 0.122 |
| HMM vechi (pe cadru, prag pe încredere, fără 0+), aceleași date | | | | 0.272 / 0.224 |
| HMM (cost minim) | **0.922 / 0.910** | **0.924 / 0.911** | **0.980 / 0.976** | **0.037 / 0.049** |

Recall-ul pentru urgențe: HMM 1.00 / 1.00, reguli 0.96 / 0.98 (la mâna pe piept fără respirație măsurată
regulile așteaptă 6 s; matricea trece de P(urgență) ≥ 0.5 după ~4.8 s). Statisticile pe clasă (sensibilitate,
specificitate, VPP, VPN, LR+, LR−) și `P(starea reală | severitatea publicată)` sunt generate automat în raport.

**Situațiile noi din Vision** (laptopurile 2 și 3), din raport:

| Situație | Rezultat |
|---|---|
| laptopul feței nu trimite 8–40 s (șofer atent) | alarme false în 0/10 segmente test, 0/8 shift; fără câmpul `cameras` din `merge.py`: 7/10, 5/8 |
| laptopul corpului nu trimite 8–40 s | 0/10, 0/8 alarme false |
| prăbușire văzută doar de camera corpului (50 de urgențe dedicate) | P(urgență) ≥ 0.5 în 10/10 (în față) și 10/10 (într-o parte); filtrul vechi 2/10 și 0/10 |
| căzut într-o parte (`slumped_side`) | severitate 3 în 10/10, după 2 s |
| aplecat (`bad_posture`) / ieșit din poziție (`out_of_position`) | memento vocal (severitate 1) în 18/18, respectiv 18/18 segmente |

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
| viteză medie: atent · distras · obosit · urgență | 97% · 92% · 65% · 24% | 97% · 92% · 66% · 24% |
| condus normal încetinit sub 90%, la ≥ 15 s după un episod | 2.5% | 2.7% |
| fără ochi (cameră oprită sau laptopul feței căzut): încetinit, intenționat | 71% | 65% |
| MRM atins în urgențe | 7/7, mediană 11.0 s | 5/5, mediană 10.0 s |
| MRM în cele 50 de urgențe dedicate (inclusiv doar camera corpului) | 50/50, mediană 10.0 s | |
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
           "bayes_severity": 3, "observed": true, "evidence": 0.2}}
```

`mode`: `NORMAL` | `CAUTION` (viteză < 95% sau distanță mărită) | `ASSIST` (asistența ≥ 60%) | `MRM`.

## Limite

* Matricele sunt învățate pe același simulator care produce datele de test: rezultatele compară variante între
  ele, nu prezic comportamentul pe drum. Pe date reale se re-învață emisia (care depinde de detectorul CV), nu
  codul. Simulatorul imită acum și ce trimit laptopurile 2 și 3: forma `merge.py` (`cameras`, câmpurile unei
  camere căzute lipsesc), evenimentele de postură, respirația oprită în jumătate din călătorii și gesturile
  benigne de 2–5 s cu mâna pe piept.
* Câte gesturi benigne cu mâna pe piept face un șofer real pe oră nu știm: dacă sunt mult mai dese decât în
  simulare, P(urgență) la 0+ e prea mare și ECU-ul încetinește degeaba; se vede repede pe o înregistrare reală.
* Riscul ponderat (`0.5·P(obosit)`) și pragurile fuzzy sunt alese de mână, pentru autostradă. Pentru oraș,
  viteza setată și regulile trebuie revăzute.
* Nu există încă o legătură cu o mașină reală sau cu `highway-env`: `vigil/ecu` e comanda, nu bucla închisă.
