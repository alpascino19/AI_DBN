# Vigil — Pitch ML 1 (3 minute demo)

## Fraza de deschidere (15 s)
„Din iulie 2026, sistemele de avertizare la neatenție sunt obligatorii în orice mașină nouă din UE.
Dar ele doar *observă*. Vigil înțelege ce se întâmplă cu șoferul și decide ce face mașina."

## Ce face Vigil (30 s)
Camera vede (ochi, privire, cap, mână pe piept, telefon, respirație). Un scor de atenție `a ∈ [0,1]`
combină semnalele. Un **LLM local** primește situația și alege severitatea 0–3 și acțiunile:
memento vocal, alarmă, pauză, „Sunteți bine?”, apel 112 simulat cu raport.

## De ce un LLM, nu doar if/else (30 s)
O privire de 1 s spre telefon **nu** e „mână pe piept + respirație grea + cap căzut”.
Combinațiile, duratele și răspunsul șoferului schimbă decizia: un „sunt bine” rapid coboară
alarma, un „nu mă simt bine” o urcă. LLM-ul judecă contextul; regulile fixe n-ar acoperi toate combinațiile.

## De ce poți avea încredere în el (45 s) — punctul forte
1. **Regulile sunt calea de siguranță** (deterministe, testate, ISO 26262). LLM-ul e consilier:
   nu poate coborî un caz clar de severitate 3, nu poate ridica o alertă fără semn în date
   (a inventat „telefon în mână” când nu era niciun telefon — garda a prins-o),
   și nu poate suna la 112 fără dovezi.
2. **LLM-ul nu scrie ce aude șoferul.** Alege dintr-un meniu; frazele sunt fixe și verificate.
3. **Plan B garantat:** LLM căzut, lent (> 2.5 s) sau JSON invalid → reguli fixe, aceeași interfață.
4. **Siguranța nu așteaptă modelul:** la severitate 3 acționează regulile imediat.

## De ce rulează într-o mașină (30 s)
- `qwen2.5:1.5b`, ~1 GB, licență **Apache 2.0** (fără restricții comerciale). Alternativa
  Qwen2.5 3B are licență non-comercială, deci am exclus-o.
- Local, fără internet, fără cadre video trimise nicăieri (privacy mode).
- Măsurat pe laptopul de demo (RTX 4050): **mediană 364 ms, p95 448 ms** per decizie, 100% GPU.
- LLM apelat doar la schimbarea situației: 11 apeluri în 30 s de somnolență, 0 la condus normal.

## Demo (vezi tabelul din TASKS.md)
Telefon → memento blând · Somnolență → alarmă + „Doriți o pauză?” · Mână pe piept → „Sunteți bine?” ·
Cap căzut fără răspuns → alarmă urgentă, buzzer, **112 simulat cu raport** · Camera oprită → încredere scăzută.

## Întrebări probabile
| Întrebare | Răspuns |
|---|---|
| Ce se întâmplă dacă LLM-ul greșește? | Garda îl corectează sau regulile preiau; vezi `safety_notes` pe dashboard. |
| E diagnostic medical? | Nu. Raportul 112 spune „semne observate de cameră, nu diagnostic”. |
| De ce 1.5B, nu un model mare? | Încape lângă YOLO + Whisper, răspunde sub 0.5 s, licență comercială; regulile duc siguranța. |
| De ce nu Llama 3.2 1B? | E ~20% mai mic, dar licență cu condiții, fără română oficială, tool use slab (BFCL 25.7); e rezerva (`VIGIL_MODEL`). |
| Ce limitări? | Benchmark-ul are 5 situații; modelul a greșit în scenarii complete (alerte inventate), de aceea există gardurile. Pragurile se reglează pe date reale. |
| Și pe un ECU clasic? | LLM-ul nu: rulează pe controllerul de cockpit (Snapdragon Cockpit, DRIVE Orin, R-Car). Comanda pentru vehicul da: controlerul fuzzy e compilat într-un tabel C de 6 KB (`ML1/ecu/vigil_fuzzy_lut.h`), fără LLM și fără alocare. |
| Cât de sigură e o alarmă? | Matricea de probabilitate o spune: un „2” publicat e oboseală reală în 58% din cadre, un „3” e urgență în 77–88%. De aceea mașina frânează doar când și filtrul HMM confirmă. |
