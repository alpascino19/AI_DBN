# Metodele ML din Vigil și de unde vin

> Pentru: echipa și juriul. Leagă deciziile din ML1 de cursurile de Machine Learning
> (V.-C. Miclea, C1–C6, în `ML_C1.pdf` … `ML_C6.pdf`) și de practici din industrie.
> Cursurile nu sunt singura sursă: unde am completat din literatura de specialitate sau din standarde
> auto, e marcat **[extern]**; acolo cifrele trebuie verificate în sursa originală înainte de citare.

## Rezumat: ce am schimbat pe baza cursurilor și a datelor reale CV

| # | Problemă găsită | Cum am găsit-o | Curs / principiu | Schimbare | Efect măsurat |
|---|---|---|---|---|---|
| 1 | CV trimite `phone_use`, `head_on_wheel`, `breathing`, `posture`, `face_detected`; ML1 aștepta alte nume | rularea demo-ului CV prin ML1 | C2: datele reale sunt rareori direct utilizabile; curățare + reprezentare | `normalize_vision` în `cv_adapter.py` | telefonul și prăbușirea pe volan sunt recunoscute |
| 2 | somnolența CV (`eyes_closed` 2.5 s) devenea urgență medicală (3) | demo CV | C1: problema bine definită; C4: decizia e separată de scor | ochi închiși singuri = micro-somn (2); 3 doar cu cap căzut sau fără răspuns | demo CV: somnolență → 2 |
| 3 | după o alertă de oboseală, un șofer prăbușit **nu ajungea la 112** | demo CV (fără răspunsuri ML2) | C5: testează pe condițiile de deployment, nu doar pe date „curate” | cale rapidă și la semn grav nou + „Sunteți bine?” obligatoriu în episodul de severitate 3 | 112 corect în demo CV și 6/6 în evaluare |
| 4 | șoferi cu ochi mici / zgomot → **21 alarme false / oră** | evaluare pe domeniul „shift” | C2: standardizare/robust scaling învățate doar pe train; C5: distribution shift | calibrare personală (mediană/IQR, 30 s, doar cadre curate) | alarme false: 21 → 0 / oră; macro-F1 0.60 [0.50–0.75] → 0.91 [0.84–0.96] (IC disjuncte) |
| 5 | pâlpâirea detectorului de față și regula generică `a < 0.75` | evaluare „shift” | C2: lipsa e informație dar trebuie interpretată; persistență | fața lipsă contează după ≥ 1 s; `a < 0.75` trebuie să dureze ≥ 4 s | alarme false eliminate fără urgențe ratate |
| 6 | comparațiile între variante erau afectate de noroc (răspunsuri aleatoare diferite) | diagnosticarea evaluării | C5: compară modelele în aceleași condiții | comportamentul șoferului simulat fixat per episod (comparație pereche) | diferențele reflectă doar sistemul |
| 7 | latențe nerealist de mici (0.1 s) și un câștig LLM care a dispărut după corecție | verificarea rezultatelor | C5: un rezultat prea bun e un semnal de alarmă | timpii simulării aliniați exact la cadre | latențe realiste: 0.2–3 s |
| 8 | LLM-ul urca telefonul (1) la oboseală (2) și inventa escaladări | evaluare cu Qwen în buclă | C5: hiperparametrii pe validare | LLM-ul nu mai poate urca peste reguli (`LLM_MAX_RAISE = 0`; pe validare 0 și 1 egale, la egalitate varianta mai sigură); la severitate 3 întrebarea e forțată „Sunteți bine?” | test: LLM+reguli 0.950 [0.90–0.99] vs reguli 0.935 [0.87–0.98], cost 0.012 vs 0.016, 0 alarme false |

### Runda 2: aliniere la codul real CV1/CV2 (după commit-urile „calibrated face monitoring” / „alert integration”)

| # | Problemă găsită | Cum | Curs / principiu | Schimbare | Efect |
|---|---|---|---|---|---|
| 9 | PERCLOS-ul CV e pe 60 s, cu ochii *aproape complet* închiși: normal ≤ 0.05, oboseală ≥ 0.15; ML1 aștepta 0.35 | citirea `eyes.py`, `alerts.py` | C2: reprezentarea trebuie să corespundă măsurătorii | profil de praguri [`profile.py`](../ml1/profile.py) aliniat la CV, comparat pe validare cu cel vechi | validare: cost 0.17 → 0.12; test: macro-F1 0.75 → 0.83 |
| 10 | pleoapele par închise la căscat și la privit în jos (telefon) | regulile CV din `alerts.py` | C1: cunoștința de domeniu ca bias inductiv | excludere a artefactelor (și pentru episodul care a început în timpul lor) | fără „micro-somn” la căscat / telefon |
| 11 | PERCLOS pe 60 s întârzie zeci de secunde la debut și la revenire | trasarea unei călătorii simulate | C2: semnale cu scări de timp diferite | închiderea pe 15 s estimată de ML1, folosită doar dacă e consistentă cu CV | alarma de oboseală se stinge la timp |
| 12 | tracker-ul CV unește activările separate de < 1 s: episodul de „ochi închiși” ≠ închidere continuă | citirea `tracker.py` + simulare fidelă | C5: testează pe forma reală a datelor | 3 s cere închidere continuă și în EAR; episodul se ignoră doar când EAR arată clar ochi deschiși | fără urgențe false din clipiri dese |
| 13 | în primele 30 s un șofer cu ochi mici era comparat cu populația | eroare pe domeniul „shift” | C2: fit pe datele disponibile până acum | baseline provizoriu din mediana cadrelor curate | alarme false în calibrare eliminate |
| 14 | 112 pentru un șofer obosit care nu răspunde era numărat „fals” | diagnosticarea evaluării | C5: eticheta trebuie să reflecte specificația | episoadele în care șoferul simulat tace sunt ambigue 2/3 (tabelul de severitate: oboseală + fără răspuns = 3) | 112: 0 ratate, 0 false |

## Pe cursuri

### C1 — Introducere: problemă, bias inductiv, „no free lunch”

* **Problema definită înainte de model.** Un eșantion = un cadru `vigil/vision` (+ contextul episodului);
  ținta = severitatea 0–3; informația disponibilă la momentul predicției = doar trecutul (nicio trăsătură
  nu „vede” viitorul).
* **Bias inductiv explicit.** Am combinat trei familii cu presupuneri diferite: reguli (cunoaștere de
  domeniu), fuziune ponderată (semnalele se compun aproape liniar), LLM (raționament contextual). Nicio
  familie nu e cea mai bună peste tot; evaluarea arată unde câștigă fiecare.
* **ML când regulile sunt greu de scris manual:** combinațiile durată × semn × răspuns sunt multe; acolo
  LLM-ul nuanțează. Măsurat: pe validare macro-F1 0.958 → 0.978 și cost înjumătățit; pe test 0.935 → 0.950,
  dar **intervalele de încredere se suprapun**, deci câștigul nu e încă demonstrat statistic (vezi
  [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)). Onest: alegerea `LLM_MAX_RAISE = 0` a fost făcută după
  ce văzusem rezultatul pe test, deci cifra de test e ușor optimistă.
* **Monitorizare după deployment** (concept drift): `vigil/ml1/status` publică scorul, încrederea,
  calibrarea și statistici, ca derapajele să fie vizibile.

### C2 — Date și preprocesare

| Principiu din curs | Implementare |
|---|---|
| curățare: valori imposibile, inconsistente | `normalize_vision`: ne-finite → `None`, decupare EAR [0, 0.6], PERCLOS [0, 1], clipit [0, 90], telefon [0, 1] |
| mecanismul lipsei contează (MCAR/MAR/MNAR) | camera oprită (privacy, decizie a utilizatorului) → încredere mică, nu „neatenție”; fața lipsă cu camera pornită ≥ 1 s → „nu privește drumul” (MNAR) |
| imputare ca transformare învățată, fără scurgere | trăsăturile modelelor ML: valori neutre + indicator explicit `face_missing`; standardizare fit doar pe train (`Standardizer`) |
| scalare robustă (mediană, IQR) | `DriverBaseline`: valori normale ale șoferului din mediană/IQR, nu medie/abatere |
| fit pe train, transform pe rest | calibrarea se învață din primele 30 s și se aplică doar cadrelor următoare |
| nu „curăța” cazurile rare pe care vrem să le prindem | calibrarea învață doar din cadre curate, dar NU elimină cadrele anormale din decizie; în plus nu acceptă un „normal” cu PERCLOS > 0.20 (n-ar trebui să învățăm oboseala ca normală) |
| dezechilibru de clase: acuratețea înșală | metrici pe clasă + macro-F1; ponderi pe clase în softmax și arbore; setul de test nu e reechilibrat |
| inginerie de trăsături | durate ale evenimentelor, raport EAR / EAR personal, indicatori de postură și respirație |

### C3 — Regresie liniară și optimizare

* Regresia softmax din evaluare e antrenată cu **gradient descent pe tot setul**, cu regularizare L2 și
  trăsături standardizate (C3: scalarea schimbă geometria optimizării; rata de învățare e hiperparametru).
* Fuziunea `a` e un model liniar în semnale (medie ponderată), cu ponderi alese de domeniu; când vor exista
  date reale etichetate, aceleași ponderi pot fi învățate (vezi „Ce urmează”).

### C4 — Clasificare și regresie logistică

* **Probabilitate ≠ decizie.** ML1 separă scorul (`a`, probabilitățile softmax) de prag (severitatea).
  Pragurile sunt decizii de aplicație, nu proprietăți ale modelului.
* **Pragurile alese pe validare, prin cost.** `tune_a_thresholds` alege tăieturile pe `a` minimizând un
  cost asimetric pe setul de validare (a rata o urgență costă de 10× mai mult decât o supraestimare cu o
  treaptă); testul nu e folosit la alegere.
* **Softmax pentru 4 clase** (0–3) cu entropie încrucișată, implementat în numpy, cu softmax stabil
  numeric (scăderea maximului).

### C5 — Evaluare și generalizare

Protocolul complet e în [EVALUATION.md](EVALUATION.md). Pe scurt:

* împărțire pe **șofer** (unitatea reală de deployment), nu pe cadre — cadrele vecine ar scurge informație;
* train / validare (praguri) / test (o dată) / **shift** (alt domeniu, raportat separat);
* macro-F1, recall pe clasă, cost pe cadru, alarme false pe oră, latență, 112 corecte/ratate/false;
* baseline-uri: clasa majoritară, prag simplu pe `a`;
* IC 95% bootstrap pe călătorii; ROC-AUC/AP pentru `1 − a`; diagramă de calibrare + Brier pentru softmax;
* scurgeri evitate: standardizare și praguri doar pe train/val; comportament pereche în comparații.

### C6 — ML clasic: k-NN, Perceptron, SVM, arbori

* **Arbore de decizie** (Gini ponderat, adâncime 4) antrenat pe trăsăturile cadrelor. Avantajul principal
  pentru auto: **reguli citibile** (listate în rezultate) pe care un inginer le poate compara cu
  `rules.py`. Pe test: macro-F1 0.98; pe shift: 0.92 — mai fragil la schimbarea domeniului.
* **Regresia softmax** (model liniar, C4) are cel mai bun macro-F1 pe date simulate, dar asta e o limită
  superioară: învață chiar simulatorul care a generat etichetele (vezi limitele).
* **De ce nu le-am pus în calea de decizie (încă):** fără date reale etichetate, un model învățat pe
  simulator învață presupunerile simulatorului. Le păstrăm ca „model umbră” și ca unealtă pentru momentul
  în care CV produce înregistrări reale.
* k-NN și SVM nu sunt implementate: pe 18 trăsături standardizate și ~37k cadre ar funcționa, dar
  latența de inferență a k-NN și lipsa probabilităților calibrate la SVM (C6) nu aduc nimic față de
  softmax + arbore pentru scopul de aici.

## Practici din afara cursurilor [extern]

* **Reglementare:** Regulamentul UE 2019/2144 (General Safety Regulation) cere sisteme de avertizare la
  somnolență (DDAW) și la distragere (ADDW) în mașinile noi; ADDW devine obligatoriu pentru toate
  înmatriculările noi din iulie 2026.
* **PERCLOS** (proporția de timp cu pleoapele ≥ 80% închise, „P80”) e una dintre cele mai validate măsuri
  ale somnolenței (studiile FHWA/NHTSA de la sfârșitul anilor '90). Pragurile depind de fereastră și de
  populație, de aici calibrarea personală și pragurile alese pe date.
* **Etichete reale pentru somnolență:** scala Karolinska (KSS, 1–9; ≥ 7 = somnoros) raportată de șofer
  la intervale regulate, plus evenimente observate (micro-somn) adnotate video. E standardul în studiile
  de simulator și ar trebui folosită pentru datele reale ale echipei.
* **Micro-somnul** durează de obicei de la ~0.5 s la câteva secunde: de aceea ochii închiși 2–3 s sunt
  tratați ca oboseală gravă (alarmă puternică), nu ca urgență medicală.
* **LLM-uri mici în buclă de siguranță:** ieșire constrânsă de schemă (structured output), meniu fix de
  acțiuni, temperatură 0, timeout cu fallback determinist, validare după generare — aceeași idee ca
  „guardrails” din sistemele LLM în producție.
* **ISO 26262 / ISO 21448 (SOTIF):** vezi [ARCHITECTURE.md](ARCHITECTURE.md#4-arhitectura-de-siguranță).

## Ce urmează când vor exista date reale

1. Înregistrări cu camera echipei CV (cu acord), mai mulți șoferi, etichete KSS + adnotări de evenimente.
2. Împărțire pe șofer (train/val/test), aceeași evaluare (`ml1/evaluate.py` primește doar alt generator).
3. Învățarea ponderilor fuziunii (regresie logistică pe semnale, C4) și a pragurilor (C4/C5) pe validare.
4. Calibrarea probabilităților (Platt / izotonic, C5) dacă un model ML intră ca „al doilea avis” pentru LLM.
5. Raportare pe subgrupuri: lumină de zi/noapte, ochelari, unghiul camerei (C5: distribution shift).
