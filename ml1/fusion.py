"""ML1 — Fuziune: semnale -> scor de atenție `a` in [0, 1].

a = 1  -> șofer complet atent
a = 0  -> șofer care nu reacționează

Pași:
  1. Fiecare semnal disponibil -> sub-scor s_i in [0, 1] (1 = atent).
  2. a_raw = sum(w_i * s_i) / sum(w_i)   (doar peste semnalele disponibile)
  3. Netezire EMA asimetrică în timp: scade repede, crește încet
     (o privire de 1 s nu prăbușește scorul, dar un cap căzut da).
  4. Încredere: depinde de sursele disponibile (fără cameră -> mai mică).

Intrare: mesaje `vigil/vision` (contractul din TASKS.md) + opțional
`vigil/driver_response` + opțional un scor de condus din simulare.
Câmpuri opționale în plus față de contract: `ts` (secunde) și `camera_on` (bool).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- config ----

# Ponderi pentru a_raw (se renormalizează peste semnalele disponibile).
WEIGHTS: dict[str, float] = {
    "eyes": 0.30,    # PERCLOS + EAR
    "gaze": 0.15,    # gaze_on_road
    "head": 0.15,    # head_drop
    "phone": 0.15,   # YOLO cell phone
    "blink": 0.05,   # rata de clipire anormală
    "events": 0.10,  # evenimente CV 2 (hand_on_chest, yawn, ...)
    "voice": 0.05,   # răspunsul la verificarea vocală
    "driving": 0.05, # comportament la volan (simulare)
}

CAMERA_SIGNALS = ("eyes", "gaze", "head", "phone", "blink", "events")

# Cât contribuie fiecare sursă la încrederea totală (suma = 1).
SOURCE_CONFIDENCE = {"camera": 0.6, "voice": 0.2, "driving": 0.2}

# Cât de grav e fiecare eveniment (multiplicat cu confidence-ul evenimentului).
EVENT_SEVERITY = {
    "eyes_closed": 1.0,
    "head_drop": 1.0,
    "microsleep": 0.8,
    "hand_on_chest": 0.6,
    "heavy_breathing": 0.4,
    "phone_in_hand": 0.4,
    "gaze_off_road": 0.3,
    "yawn": 0.3,
}

# Plafoane: unele semnale grave nu trebuie "diluate" de restul mediei.
# a_raw <= floor + (1 - floor) * s_i   (ex. ochi închiși -> a_raw <= 0.35)
CAPS = {"eyes": 0.35, "events": 0.40}

TAU_DOWN = 0.7   # s — cât de repede scade a
TAU_UP = 4.0      # s — cât de repede revine a
VOICE_TTL = 30.0  # s — cât timp e valabil un răspuns vocal
DRIVING_TTL = 5.0 # s — cât timp e valabil scorul de condus


# --------------------------------------------------------------- helpers ----

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _num(v: Any) -> float | None:
    """Număr finit sau None (tolerant la date murdare de la CV)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ramp_down(x: float, good: float, bad: float) -> float:
    """1 la x<=good, 0 la x>=bad, liniar între."""
    return _clamp((bad - x) / (bad - good))


# ----------------------------------------------------- sub-scoruri (0..1) ----

def score_eyes(perclos: float | None, ear: float | None, baseline_ear: float | None) -> float | None:
    parts: list[tuple[float, float]] = []
    if perclos is not None:
        parts.append((0.7, _ramp_down(_clamp(perclos), good=0.15, bad=0.45)))
    if ear is not None:
        base = baseline_ear if baseline_ear and baseline_ear > 0 else 0.30
        ratio = ear / base  # ochi deschiși normal ≈ 1
        parts.append((0.3, _clamp((ratio - 0.55) / (0.85 - 0.55))))
    if not parts:
        return None
    return sum(w * s for w, s in parts) / sum(w for w, _ in parts)


def score_blink(rate: float | None) -> float | None:
    """Clipiri/min: normal 8–25. Prea rar (somnolență) sau prea des (anormal) scade."""
    if rate is None or rate < 0:
        return None
    if rate < 8:
        return _clamp((rate - 2) / (8 - 2))
    if rate > 25:
        return _ramp_down(rate, good=25, bad=45)
    return 1.0


def score_events(events: Any) -> float | None:
    if not isinstance(events, list):
        return None
    worst = 0.0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = ev.get("event")
        sev = EVENT_SEVERITY.get(name, 0.0)
        conf = _clamp(_num(ev.get("confidence")) or 0.0)
        dur = _num(ev.get("duration_sec")) or 0.0
        # evenimentele scurte contează mai puțin (plin după 3 s)
        dur_factor = 0.5 + 0.5 * _clamp(dur / 3.0)
        penalty = sev * conf * dur_factor
        if name == "hand_on_chest" and ev.get("breathing") == "heavy":
            penalty = max(penalty, 0.9 * conf * dur_factor)
        worst = max(worst, penalty)
    return 1.0 - worst


def score_voice(resp: dict | None) -> float | None:
    if not resp:
        return None
    if resp.get("answered") is False:
        return 0.0
    lat = _num(resp.get("latency_sec"))
    if lat is None:
        return None
    return _ramp_down(lat, good=2.0, bad=8.0)


# ---------------------------------------------------------------- fuziune ----

@dataclass
class FusionResult:
    ts: float
    a: float
    a_raw: float | None
    confidence: float
    camera_on: bool
    critical: bool
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)
    capped_by: str | None = None  # semnalul care a plafonat a_raw (pentru dashboard)

    def to_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "a": round(self.a, 3),
            "a_raw": None if self.a_raw is None else round(self.a_raw, 3),
            "confidence": round(self.confidence, 3),
            "camera_on": self.camera_on,
            "critical": self.critical,
            "contributions": self.contributions,
            "capped_by": self.capped_by,
        }


class AttentionFusion:
    def __init__(self, baseline_ear: float | None = None, initial_a: float = 1.0):
        self.baseline_ear = baseline_ear
        self.a = initial_a
        self._last_ts: float | None = None
        self._voice: tuple[float, dict] | None = None
        self._driving: tuple[float, float] | None = None

    # -- intrări secundare ---------------------------------------------------
    def update_driver_response(self, resp: dict, ts: float | None = None) -> None:
        """Mesaj de pe `vigil/driver_response`."""
        self._voice = (ts if ts is not None else time.time(), resp)

    def update_driving(self, score: float, ts: float | None = None) -> None:
        """Scor 0..1 din simulare (1 = corecții normale, fără ieșiri de pe bandă)."""
        s = _num(score)
        if s is not None:
            self._driving = (ts if ts is not None else time.time(), _clamp(s))

    # -- intrarea principală -------------------------------------------------
    def update(self, msg: dict | None) -> FusionResult:
        """Procesează un mesaj `vigil/vision`. `None` = niciun cadru (camera oprită)."""
        msg = msg if isinstance(msg, dict) else {}
        ts = _num(msg.get("ts"))
        if ts is None:
            ts = time.time()

        camera_on = self._camera_on(msg)
        scores: dict[str, float] = {}

        if camera_on:
            cand = {
                "eyes": score_eyes(_num(msg.get("perclos")), _num(msg.get("ear")), self.baseline_ear),
                "gaze": None if not isinstance(msg.get("gaze_on_road"), bool) else float(msg["gaze_on_road"]),
                "head": None if not isinstance(msg.get("head_drop"), bool) else (0.0 if msg["head_drop"] else 1.0),
                "phone": None if _num(msg.get("phone")) is None else 1.0 - _clamp(_num(msg["phone"])),
                "blink": score_blink(_num(msg.get("blink_rate"))),
                "events": score_events(msg.get("events")),
            }
            scores.update({k: v for k, v in cand.items() if v is not None})

        if self._voice and ts - self._voice[0] <= VOICE_TTL:
            v = score_voice(self._voice[1])
            if v is not None:
                scores["voice"] = v
        if self._driving and ts - self._driving[0] <= DRIVING_TTL:
            scores["driving"] = self._driving[1]

        # a_raw ---------------------------------------------------------------
        total_w = sum(WEIGHTS[k] for k in scores)
        a_raw = sum(WEIGHTS[k] * s for k, s in scores.items()) / total_w if total_w else None
        capped_by = None
        if a_raw is not None:
            for k, floor in CAPS.items():
                if k in scores:
                    cap = floor + (1.0 - floor) * scores[k]
                    if cap < a_raw:
                        a_raw, capped_by = cap, k

        contributions = {
            k: {
                "score": round(s, 3),
                "weight": round(WEIGHTS[k] / total_w, 3),
                "contrib": round(WEIGHTS[k] * s / total_w, 3),
            }
            for k, s in scores.items()
        }

        # semnale critice: ocolesc netezirea (nu așteptăm 1 s la cap căzut)
        eyes_closed = scores.get("eyes", 1.0) < 0.3
        critical = (
            (scores.get("head") == 0.0 and eyes_closed)        # cap căzut + ochi închiși
            or scores.get("events", 1.0) <= 0.1                # eveniment grav, sigur, prelungit
            or (scores.get("voice") == 0.0 and eyes_closed)    # fără răspuns + ochi închiși
        )

        # netezire ------------------------------------------------------------
        dt = 0.0 if self._last_ts is None else max(0.0, ts - self._last_ts)
        self._last_ts = ts
        if a_raw is not None:
            if critical:
                self.a = min(self.a, a_raw)
            else:
                tau = TAU_DOWN if a_raw < self.a else TAU_UP
                alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
                self.a += alpha * (a_raw - self.a)
            self.a = _clamp(self.a)
        # fără niciun semnal: păstrăm ultimul a, dar încrederea e 0

        # încredere -----------------------------------------------------------
        confidence = 0.0
        if any(k in scores for k in CAMERA_SIGNALS):
            cam_cov = sum(WEIGHTS[k] for k in CAMERA_SIGNALS if k in scores) / sum(WEIGHTS[k] for k in CAMERA_SIGNALS)
            confidence += SOURCE_CONFIDENCE["camera"] * cam_cov
        if "voice" in scores:
            confidence += SOURCE_CONFIDENCE["voice"]
        if "driving" in scores:
            confidence += SOURCE_CONFIDENCE["driving"]

        return FusionResult(ts, self.a, a_raw, _clamp(confidence), camera_on, bool(critical),
                            contributions, capped_by)

    @staticmethod
    def _camera_on(msg: dict) -> bool:
        if isinstance(msg.get("camera_on"), bool):
            return msg["camera_on"]
        return any(k in msg for k in ("ear", "perclos", "gaze_on_road", "head_drop", "phone"))
