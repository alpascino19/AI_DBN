# Alegerea LLM-ului: Qwen2.5 1.5B

> Pentru: echipa și juriul. Ce model folosește ML1, de ce, și ce s-a măsurat.

**Modelul folosit: `qwen2.5:1.5b`** (Qwen2.5-1.5B-Instruct, cuantizat Q4_K_M, ~1 GB, prin Ollama).
Rezerva configurabilă: `llama3.2:1b` (`VIGIL_MODEL=llama3.2:1b`). Llama 3.2 **nu a fost instalat și nu a
fost testat** pe laptopul de demo.

## De ce un model local mic

* Ținta e controllerul de cockpit / DMS din mașină (Snapdragon Cockpit, NVIDIA DRIVE Orin, Renesas R-Car),
  nu un ECU clasic (microcontrollerele nu pot rula un LLM). Bugetul realist pentru LLM: ~1–1.5 GB și
  sute de milisecunde, pe un cip împărțit cu navigația și camerele.
* Local = fără internet, fără cost per apel, nicio informație despre șofer nu pleacă din mașină.

## Qwen2.5 1.5B vs Llama 3.2 1B

| | Qwen2.5 1.5B Instruct | Llama 3.2 1B Instruct |
|---|---|---|
| Parametri | 1.54 B | 1.23 B |
| Memorie (Q4) | ~1 GB | ~0.8 GB |
| Licență | **Apache 2.0** (fără condiții) | Llama 3.2 Community License (atribuire, politică de utilizare, plafon de utilizatori) |
| Limbi oficiale | multe (29+) | 8; **româna nu e oficială** |
| Tool use (BFCL V2) | neraportat la această mărime | 25.7 (slab) |
| Ecosistem pentru cipuri auto/mobile | llama.cpp / Ollama | lansat cu Qualcomm, MediaTek, Arm; versiuni cuantizate oficiale |
| Măsurat la noi | da (mai jos) | nu |

Scorurile publicate (MMLU-redux 50.7 pentru Qwen, MMLU 49.3 pentru Llama) vin din teste diferite și nu
pot fi comparate direct. Pentru ML1, alegerea s-a făcut pe licență, ieșire structurată și pe măsurători.

## Ce s-a măsurat (laptop de demo, RTX 4050 6 GB, 100% GPU, 1.1 GB în memorie)

| Măsurătoare | Rezultat |
|---|---|
| latență per decizie (5 contexte × 5 rulări) | mediană 253–364 ms, p95 380–448 ms |
| JSON valid / severitate corectă pe cele 5 contexte de bază | 100% / 100% |
| intervenția gărzilor pe cele 5 contexte | 20% (omite întrebarea vocală la oboseală → adăugată de gardă) |
| evaluare în buclă închisă pe 10 șoferi de test (simulare în semantica reală CV) | macro-F1 0.887 [0.82–0.94] vs 0.825 [0.76–0.87] doar reguli; cost 0.078 vs 0.090; 6/6 apeluri 112 corecte, 0 false; alarme false 17 vs 14 / oră |
| prima încărcare după repornire | ~90 s (o singură dată) |

Greșeli reale ale modelului, prinse de gărzi (cu teste de regresie): a inventat „telefon în mână” fără
telefon, a urcat telefonul (1) la oboseală (2), a cerut „Doriți o pauză?” la o urgență, a omis întrebarea
vocală la oboseală. De aceea LLM-ul nu decide singur: nu poate urca peste reguli și nu poate suna la 112
fără dovezi.

## Pentru a compara cu Llama 3.2 pe același hardware

```
ollama pull llama3.2:1b-instruct-q4_K_M
ollama pull qwen2.5:1.5b-instruct-q4_K_M
python -m ml1.bench --models qwen2.5:1.5b-instruct-q4_K_M llama3.2:1b-instruct-q4_K_M --runs 5
```

Regulă de decizie propusă: un model trece dacă are ≥ 90% severitate corectă și p95 < 1 s; dacă trec
amândouă, câștigă cel cu mai puține intervenții ale gărzilor, apoi cel mai mic.
