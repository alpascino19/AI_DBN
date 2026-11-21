"""Calibrare personală a șoferului (primele ~30 s).

Bazat pe Curs 2 (preprocesare) și Curs 5 (distribution shift):
  * Pragurile absolute (EAR 0.30, PERCLOS 0.06, 16 clipiri/min) descriu un șofer „mediu”.
    Un șofer cu ochi mai mici are EAR ~0.22 și PERCLOS măsurat mai mare: fără calibrare,
    sistemul l-ar considera permanent obosit (fals pozitive).
  * Învățăm valorile normale din fereastra de calibrare și le aplicăm DOAR cadrelor viitoare
    (fit pe „train”, transform pe restul — fără scurgere de informație din viitor).
  * Statistici robuste: mediana și IQR, nu media/abaterea standard (Curs 2: robust scaling),
    ca un căscat sau o clipire lungă în calibrare să nu strice valorile normale.
  * Curățare: învățăm doar din cadre „curate” (privire la drum, fără evenimente, fără telefon);
    dacă șoferul e deja obosit la pornire, NU învățăm o stare anormală ca fiind normală
    (limitele `EAR_BOUNDS` / `profile.P["perclos_max_baseline"]`), iar după 3 × fereastra revenim la valorile populației.
"""

from __future__ import annotations

import math
from typing import Any

from .profile import P

POPULATION = {"ear": 0.30, "blink": 16.0, "perclos": 0.06}
EAR_BOUNDS = (0.18, 0.40)   # un baseline în afara acestui interval nu e plauzibil -> limitat
# Limita PERCLOS pentru un „normal” plauzibil și PERCLOS-ul populației: vezi profile.py (depind de semantica CV).
BLINK_BOUNDS = (6.0, 30.0)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class DriverBaseline:
    def __init__(self, seconds: float = 30.0, min_samples: int = 60):
        self.seconds, self.min_samples = seconds, min_samples
        self.first_ts: float | None = None
        self.samples: dict[str, list[float]] = {"ear": [], "blink": [], "perclos": []}
        self.ready = False
        self.source = "calibrating"  # calibrating | driver | population
        self.stats: dict[str, dict[str, float]] = {}

    @staticmethod
    def is_clean(msg: dict) -> bool:
        if msg.get("camera_on") is False or msg.get("face_detected") is False:
            return False
        if msg.get("gaze_on_road") is False or msg.get("head_drop") is True:
            return False
        if (_num(msg.get("phone")) or 0.0) >= 0.3:
            return False
        evs = msg.get("events")
        return not any(isinstance(e, dict) and (_num(e.get("confidence")) or 0) >= 0.5 for e in (evs or []))

    def update(self, msg: dict, ts: float) -> bool:
        """Adaugă un cadru. Întoarce True în momentul în care calibrarea tocmai s-a încheiat."""
        if self.ready or not isinstance(msg, dict):
            return False
        if self.first_ts is None:
            self.first_ts = ts
        if self.is_clean(msg):
            for key, field in (("ear", "ear"), ("blink", "blink_rate"), ("perclos", "perclos")):
                v = _num(msg.get(field))
                if v is not None:
                    self.samples[key].append(v)
        elapsed = ts - self.first_ts
        if elapsed >= self.seconds and len(self.samples["ear"]) >= self.min_samples:
            self._finish("driver")
            return True
        if elapsed >= 3 * self.seconds:
            self._finish("population")
            return True
        return False

    def _finish(self, source: str) -> None:
        self.ready, self.source = True, source
        for key, xs in self.samples.items():
            if source == "driver" and len(xs) >= self.min_samples // 2:
                med = _quantile(xs, 0.5)
                iqr = _quantile(xs, 0.75) - _quantile(xs, 0.25)
            else:
                med, iqr = POPULATION[key], float("nan")
            self.stats[key] = {"median": med, "iqr": iqr, "n": len(xs)}
        # plauzibilitate: nu învățăm o stare anormală ca fiind „normală”
        self.stats["ear"]["median"] = min(EAR_BOUNDS[1], max(EAR_BOUNDS[0], self.stats["ear"]["median"]))
        self.stats["blink"]["median"] = min(BLINK_BOUNDS[1], max(BLINK_BOUNDS[0], self.stats["blink"]["median"]))
        if source != "driver" or self.stats["perclos"]["n"] < self.min_samples // 2:
            self.stats["perclos"]["median"] = P["pop_perclos"]
        self.stats["perclos"]["median"] = min(P["perclos_max_baseline"], max(0.0, self.stats["perclos"]["median"]))

    def provisional_ear(self) -> float | None:
        """Mediana EAR din cadrele curate de până acum (după >= 15 cadre), cât calibrarea nu s-a încheiat.
        Fără asta, în primele 30 s un șofer cu ochi mici era comparat cu populația (EAR 0.30) și
        primea „atenție redusă” (evaluarea pe domeniul shift)."""
        xs = self.samples["ear"]
        if self.ready or len(xs) < 15:
            return None
        return min(EAR_BOUNDS[1], max(EAR_BOUNDS[0], _quantile(xs, 0.5)))

    # ------------------------------------------------------------ rezultate
    @property
    def ear(self) -> float | None:
        return self.stats["ear"]["median"] if self.ready else None

    @property
    def blink(self) -> float | None:
        return self.stats["blink"]["median"] if self.ready else None

    @property
    def perclos_offset(self) -> float:
        """Cât PERCLOS are șoferul „din construcție” peste media populației (de scăzut)."""
        if not self.ready:
            return 0.0
        return max(0.0, self.stats["perclos"]["median"] - P["pop_perclos"])

    def calibrated_perclos(self, perclos: float | None) -> float | None:
        p = _num(perclos)
        return None if p is None else max(0.0, p - self.perclos_offset)

    def to_dict(self) -> dict:
        return {"ready": self.ready, "source": self.source,
                "ear": None if self.ear is None else round(self.ear, 3),
                "blink": None if self.blink is None else round(self.blink, 1),
                "perclos_offset": round(self.perclos_offset, 3)}
