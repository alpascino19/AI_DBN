"""Profilul de praguri care depind de SEMANTICA semnalelor CV.

De ce un modul separat: pragurile de mai jos au sens doar împreună cu felul în care CV le măsoară.
Când echipa CV schimbă definiția (fereastra PERCLOS, pragul de „ochi închiși”), se schimbă aici,
iar evaluarea (`ml1/evaluate.py`) compară profilele pe aceleași date (Curs 5: deciziile pe validare).

* `cv`     — aliniat la codul real CV1 (Vision/vigil_vision/face/eyes.py, alerts.py, stack.py):
             PERCLOS = fracția de timp cu ochii APROAPE COMPLET închiși pe 60 s (normal ~0.02–0.05,
             oboseală >= 0.15 la CV), ochi închiși >= 1.5 s = micro-somn, >= 3 s = critic,
             pleoapele par închise la căscat și când privește în jos (telefon) -> nu e micro-somn.
* `legacy` — pragurile inițiale ML1, alese pe datele false proprii (PERCLOS 0.35 etc.). Păstrat doar
             pentru comparație în evaluare.
"""

from __future__ import annotations

PROFILES: dict[str, dict] = {
    "cv": {
        "perclos_drowsy": 0.15,      # CV alerts.py: oboseală la PERCLOS >= 0.15
        "perclos_yawn": 0.12,        # căscat + ochi obosiți (0.08 dădea alarme false: strângerea ochilor la căscat)
        "perclos_good": 0.05,        # rampa scorului de ochi în fuziune: 1 sub atât...
        "perclos_bad": 0.30,         # ...0 peste atât
        "perclos_recovered": 0.05,   # CV eyes.py: recovery_perclos_max pe ultimele 15 s
        "eyes_sev2_sec": 1.5,        # CV eyes.py: long_closed_sec
        "eyes_sev3_sec": 3.0,        # CV alerts.py: eyes_closed_critical
        "artifact_exclusion": True,  # căscat / privire în jos / telefon: pleoapele par închise
        "perclos_recent": True,      # ML1 estimează închiderea pe 15 s (PERCLOS pe 60 s revine încet)
        "pop_perclos": 0.03,         # PERCLOS normal al populației, în semantica CV
        "perclos_max_baseline": 0.10,  # un „normal” mai mare la calibrare = probabil deja obosit
        "silence_needs_signs": False,  # tăcere -> 3 dacă a < 0.5 (False) sau și dacă semnele de oboseală persistă (True)
        "eyes_crit_continuous": True,  # „ochi închiși >= 3 s” cere închidere CONTINUĂ și în EAR (nu episod unit de tracker)
    },
    "legacy": {
        "perclos_drowsy": 0.35,
        "perclos_yawn": 0.20,
        "perclos_good": 0.15,
        "perclos_bad": 0.45,
        "perclos_recovered": None,
        "eyes_sev2_sec": 2.0,
        "eyes_sev3_sec": None,
        "artifact_exclusion": False,
        "perclos_recent": False,
        "pop_perclos": 0.06,
        "perclos_max_baseline": 0.20,
        "silence_needs_signs": False,
        "eyes_crit_continuous": False,
    },
}

PROFILES["cv_signs"] = {**PROFILES["cv"], "silence_needs_signs": True}

ACTIVE = "cv"
P: dict = dict(PROFILES[ACTIVE])


def use(name: str) -> None:
    """Schimbă profilul activ (folosit de evaluare pentru comparații)."""
    global ACTIVE
    ACTIVE = name
    P.clear()
    P.update(PROFILES[name])
