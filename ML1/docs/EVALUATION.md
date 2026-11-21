# Evaluarea ML1: protocol

> Pentru: echipa și juriul. Cum măsurăm dacă ML1 ia decizii bune, ce înseamnă cifrele și ce NU spun.
> Rezultatele curente (generate automat): [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

```
python -m ml1.evaluate            # ~20 s, doar reguli + modele clasice
python -m ml1.evaluate --llm      # + varianta cu Qwen2.5 1.5B în buclă (~2.5 min, Ollama pornit)
```

## 1. Datele

**Călătorii simulate, etichetate, în forma și semantica reală a mesajelor CV** (după codul din
`Vision/vigil_vision/face/eyes.py`, `stack.py`, `events/tracker.py`, `pipeline.py`):
* închiderea ochilor simulată cadru cu cadru; `perclos` = fracția cadrelor închise pe ultimele **60 s**,
  `null` în primele 10 s;
* tracker-ul CV emulat fidel: evenimente după durata minimă, **pauze < 1 s nu întrerup episodul**;
* artefacte: pleoape coborâte la privit în jos (telefon) și ochi strânși la căscat, măsurate ca închise;
* evenimente CV1 (`eyes_closed`, `microsleep`, `gaze_off_road` cu direcție, `yawn`, `low_blink_rate`,
  `head_drop`) și CV2 (`phone_use`, `hand_on_chest`, `heavy_breathing`, `head_on_wheel`, `breathing`, `posture`).

O călătorie (~6 min, 5 Hz) = 35 s de condus normal (calibrare) + episoade în ordine aleatoare,
separate de 20–35 s de condus normal:

| Episod | Eticheta adevărată | Ce testează |
|---|---|---|
| privire scurtă 0.6–1.6 s, căscat scurt, atingere a pieptului 1.5 s | 0 | alarme false |
| telefon 5–12 s | 1 | distragere |
| oboseală progresivă (PERCLOS, EAR, clipit) cu ochi închiși 2–2.4 s și căscat la vârf | 0 → 2 → 0 | somnolență, micro-somn |
| camera oprită 15 s | 0 | privacy mode |
| final: mână pe piept + respirație grea (uneori urmată de prăbușire), prăbușire, sau nimic | 3 / 0 | urgență, 112 |

**Buclă închisă:** când ML1 pune o întrebare, șoferul simulat răspunde după starea lui reală:
normal → „da, sunt bine” (~1–2 s); obosit → răspunde lent (70–90%) sau tace; mână pe piept →
„nu mă simt bine” sau tace; prăbușit → tace. Comportamentul e fixat per episod, deci două variante
comparate primesc exact aceleași răspunsuri.

**Șoferi:** fiecare are EAR normal, PERCLOS normal, clipit, zgomot, rata de pierdere a feței,
încrederea detectorului și latența de răspuns proprii.

| Împărțire | Șoferi | Folosire |
|---|---|---|
| train | 20 | antrenarea modelelor ML (softmax, arbore) și a standardizării |
| val | 8 | alegerea pragurilor (cost minim) |
| test | 10 | **o singură evaluare**, după ce toate alegerile sunt făcute |
| shift | 8 | alt domeniu: ochi mici (EAR 0.20–0.24), PERCLOS normal 0.12–0.18, zgomot 1.4–2×, fața pierdută 5–10% din cadre |

Împărțirea e **pe șofer**: cadrele vecine ale aceluiași șofer sunt aproape identice, iar o împărțire pe
cadre ar da rezultate artificial de bune (Curs 5: unitatea de împărțire = subiectul / secvența).

## 2. Ferestre „nu contează”

Sistemul are întârzieri intenționate (evenimentele CV apar după 1–3 s, severitatea coboară abia după
5 s de calm). Ca să nu penalizăm comportamentul dorit, cadrele din primele 4 s după o escaladare reală
și din primele 12 s după o de-escaladare nu intră în matricea de confuzie. Latența de detecție e
măsurată separat.

**Episoade ambigue.** Tabelul de severitate spune: oboseală / ochi închiși + fără răspuns vocal = 3. Într-un
episod de oboseală în care șoferul simulat tace, cadrele de severitate 2 nu intră în matrice (sunt
legitim 2 sau 3), iar un 112 după un „Sunteți bine?” fără răspuns e acceptabil (nici corect, nici fals).
Fiecare apel 112 e punctat după segmentul în care a avut loc.

**Profilul de praguri** ([`profile.py`](../ml1/profile.py)) se alege pe validare (cost minim): `legacy`
(pragurile vechi), `cv` (aliniat la CV), `cv_signs` (plus escaladare pe tăcere dacă semnele persistă).

## 3. Metrici

| Metrică | Ce răspunde | De ce |
|---|---|---|
| **macro-F1** [IC 95%] | cât de bine e prinsă fiecare severitate, toate clasele egale | severitatea 3 e rară; acuratețea ar fi ~90% spunând mereu „0” |
| **recall severitate 3** [IC 95%] | din cadrele de urgență, câte sunt recunoscute ca urgență | a rata o urgență e cea mai gravă eroare |
| precizie / recall pe clasă | unde greșește | macro-F1 ascunde compromisul (Curs 5) |
| **cost / cadru** | eroarea ponderată: subestimare cu k trepte = 5k (×2 la urgență), supraestimare = k | erorile nu costă la fel |
| **alarme false / oră** | decizii cu severitate ≥ 1 când adevărul e 0 | ce simte șoferul: un sistem care țipă degeaba e oprit |
| **latența** (mediană, p90, ratate) | cât durează de la debut până la severitatea corectă | |
| **112 corecte / ratate / false** | pe călătorie | false = apel fără urgență; ratate = urgență fără apel |
| ROC-AUC, AP pentru `1 − a` | cât de bine ordonează scorul de atenție cadrele afectate (≥ 2) | discriminare, independentă de prag |
| Brier + diagramă de calibrare | dacă probabilitățile softmax înseamnă ce spun | calibrare ≠ discriminare (Curs 5) |

IC 95% vin din bootstrap pe călătorii (400 eșantionări): diferențe mai mici decât intervalul nu sunt
semnificative.

## 4. Variante comparate

| Variantă | Ce e |
|---|---|
| clasa majoritară | spune mereu 0 (baseline trivial) |
| prag pe `a` | 3 tăieturi pe scorul de atenție, alese pe validare |
| softmax, arbore | modele clasice antrenate pe train (Curs 4, 6), pe cadre, fără buclă închisă |
| reguli fără calibrare | motorul ML1 fără calibrare personală |
| **reguli + calibrare** | motorul ML1 actual, fără LLM |
| **LLM + reguli + calibrare** | sistemul complet, cu Qwen2.5 1.5B (opțiunea `--llm`) |

## 5. Ce spun și ce NU spun cifrele

**Spun:** dacă o modificare strică sau îmbunătățește ceva (regresii), unde sunt compromisurile (alarme
false vs urgențe ratate), cum se comportă sistemul pe un domeniu diferit, dacă 112 e sunat corect, cât
adaugă LLM-ul peste reguli.

**NU spun** cât de bine funcționează Vigil pe șoferi reali. Etichetele vin din același simulator care
generează datele, deci:
* softmax / arborele ating ~0.98–0.99 pentru că învață simulatorul (o limită superioară, nu o promisiune);
* ROC-AUC = 1.00 pentru `1 − a` arată că simulatorul separă clar stările; pe date reale va fi mai mic;
* zgomotul real (lumină, ochelari, unghiuri, mișcare) e mai bogat decât cel simulat.

## 6. Date reale (pasul următor)

1. Înregistrări cu pipeline-ul CV (doar cifre, fără video salvat), cu acordul participanților.
2. Etichete: KSS la fiecare ~5 min, adnotări de evenimente (telefon, căscat, micro-somn), scenarii jucate
   pentru urgențe (mână pe piept, prăbușire) — niciodată urgențe reale provocate.
3. Aceeași evaluare: `ml1/evaluate.py` are nevoie doar de alt generator care citește înregistrările
   și etichetele; împărțirea pe șofer, metricile și raportul rămân.
4. Raportare pe subgrupuri (zi/noapte, ochelari, unghiul camerei) — Curs 5: distribution shift.
