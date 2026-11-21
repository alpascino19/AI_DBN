# Documentația ML1 (motorul de decizie Vigil)

| Document | Pentru ce |
|---|---|
| [TECHNICAL_DEEP_DIVE.pdf](TECHNICAL_DEEP_DIVE.pdf) | aceeași prezentare tehnică, ca PDF de 18 pagini cu diagrame și grafice (sursa LaTeX: [latex/deep_dive.tex](latex/deep_dive.tex), `latexmk -xelatex deep_dive.tex`) |
| [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md) | **de aici începi pentru înțelegerea tehnică**: un cadru real urmărit prin tot sistemul, fiecare modul cu formule și linii de cod, limite, exerciții, bibliografie (în engleză) |
| [API.md](API.md) | API-ul HTTP pentru serverul Spring Boot: rute, formatul cadrelor, ce tolerează ML1, exemple Java |
| [CV_INTEGRATION.md](CV_INTEGRATION.md) | semantica reală a semnalelor CV, ce s-a aliniat în ML1, ce cerem echipei CV |
| [ARCHITECTURE.md](ARCHITECTURE.md) | fluxul de date, contractele MQTT, cum citește ML1 mesajele CV, severitate → acțiuni, siguranță, latență |
| [ML_METHODS.md](ML_METHODS.md) | ce metode ML am folosit, legate de cursurile C1–C6, și ce am schimbat pe baza datelor CV |
| [EVALUATION.md](EVALUATION.md) | protocolul de evaluare (împărțire pe șofer, metrici, baseline-uri, limite) |
| [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) | rezultatele curente, generate de `python -m ml1.evaluate` |
| [MODEL_CHOICE.md](MODEL_CHOICE.md) | de ce Qwen2.5 1.5B și ce s-a măsurat |
| [RUNBOOK.md](RUNBOOK.md) | comenzi: teste, dashboard, MQTT cu CV/ML2, benchmark, evaluare |

## Pe scurt

* **Intrare:** `vigil/vision` de la CV (forma reală din `Vision/vigil_vision/schemas.py`) și
  `vigil/driver_response` de la ML2.
* **Ieșire:** `vigil/decision` (severitate 0–3 + acțiuni: voce, alarmă, ESP32, „Sunteți bine?”, 112 simulat).
* **Cum:** normalizare + calibrare personală + scor de atenție + reguli deterministe (calea de siguranță)
  + LLM local Qwen2.5 1.5B (nuanțează, alege acțiunile, explică) + gărzi.
* **Intrare viitoare:** server Spring Boot + cameră live prin [API.md](API.md) (REST + SSE) sau MQTT.
* **Măsurat (simulare în semantica reală CV, șoferi nevăzuți):** 0 urgențe ratate, 0 apeluri 112 false;
  macro-F1 0.75 (praguri vechi) → 0.83 (aliniat la CV + calibrare) → 0.89 (cu LLM); ~14–20 alarme false / oră
  în simulare (care pune un eveniment „capcană” la ~30 s, deci mult mai des decât condusul real).
* **Limita principală:** totul e măsurat pe date simulate în forma reală a mesajelor CV; următorul pas
  sunt înregistrări reale etichetate ([EVALUATION.md](EVALUATION.md#6-date-reale-pasul-următor)).
