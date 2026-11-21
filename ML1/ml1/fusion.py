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

from .profile import P

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
    # eyes_closed singur = micro-somn (oboseală), nu inconștiență: 0.8 ca să nu fie „critic”
    # fără cap căzut; datele CV trimit eyes_closed ~2.5 s și în scenariul de somnolență.
    "eyes_closed": 0.8,
    "head_drop": 1.0,
    "head_on_wheel": 1.0,   # CV2: prăbușit pe volan
    "slumped_side": 1.0,    # CV2: căzut într-o parte și nemișcat
    "out_of_position": 0.3, # CV2: corpul nu se vede / s-a îndepărtat mult de scaun
    "bad_posture": 0.2,     # CV2: aplecat / cocoșat de peste 10 s
    "phone_use": 0.4,      # numele CV2 (normalizat oricum la phone_in_hand)
    "low_blink_rate": 0.3,  # CV1: clipit rar (privit în gol / somnolență)
    "hyper_blinking": 0.3,  # CV1: clipit foarte des
    "frequent_yawning": 0.5,  # CV1: căscat des = oboseală
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

FACE_MISSING_SEC = 1.0  # fața lipsă cel puțin atât = „nu privește drumul”
LOW_A = 0.75            # pragul regulii generice „atenție redusă”

TAU_DOWN = 0.7  # s — cât de repede scade a
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

def eye_artifact(events: Any, phone: float | None = None, direction: Any = None) -> bool:
    """Pleoapele PAR închise fără să fie somnolență: căscat, privire în jos, telefon în mână.
    Cunoștință de domeniu preluată din regulile CV (Vision/vigil_vision/face/alerts.py)."""
    if not P["artifact_exclusion"]:
        return False
    if (phone or 0.0) >= 0.5 or direction == "down":   # `direction` dacă CV îl publică (HeadState)
        return True
    for e in events if isinstance(events, list) else []:
        if not isinstance(e, dict) or (_num(e.get("confidence")) or 0.0) < 0.5:
            continue
        name = e.get("event")
        if name in ("yawn", "phone_in_hand", "phone_use"):
            return True
        if name == "gaze_off_road" and e.get("direction") == "down":
            return True
    return False


def score_eyes(perclos: float | None, ear: float | None, baseline_ear: float | None,
               artifact: bool = False) -> float | None:
    parts: list[tuple[float, float]] = []
    if perclos is not None:
        parts.append((0.7, _ramp_down(_clamp(perclos), good=P["perclos_good"], bad=P["perclos_bad"])))
    if ear is not None and not artifact:  # EAR instantaneu e înșelător la căscat / privire în jos
        base = baseline_ear if baseline_ear and baseline_ear > 0 else 0.30
        ratio = ear / base  # ochi deschiși normal ≈ 1
        parts.append((0.3, _clamp((ratio - 0.55) / (0.85 - 0.55))))
    if not parts:
        return None
    return sum(w * s for w, s in parts) / sum(w for w, _ in parts)


def score_blink(rate: float | None, baseline: float | None = None) -> float | None:
    """Clipiri/min: normal 8–25 (sau în jurul valorii personale din calibrare).
    Prea rar (somnolență) sau prea des (anormal) scade."""
    if rate is None or rate < 0:
        return None
    low, high = 8.0, 25.0
    if baseline:
        low, high = min(low, 0.5 * baseline), max(high, 1.8 * baseline)
    if rate < low:
        return _clamp((rate - 2) / (low - 2))
    if rate > high:
        return _ramp_down(rate, good=high, bad=high + 20)
    return 1.0


def score_events(events: Any) -> float | None:
    if not isinstance(events, list):
        return None
    worst = 0.0
    artifact = eye_artifact(events)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = ev.get("event")
        if artifact and name in ("eyes_closed", "microsleep"):
            continue  # pleoape coborâte la căscat / privire în jos: nu e micro-somn
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
    low_a_sec: float = 0.0        # de cât timp a < LOW_A (persistența, pentru regula generică)
    eyes_open_now: bool = False   # EAR clar deschis acum (folosit să nu credem un eyes_closed „unit” de tracker)
    eyes_closed_sec: float = 0.0  # de cât timp EAR arată ochii închiși CONTINUU
    ear_available: bool = False   # EAR valid în acest cadru (față vizibilă, fără artefact)
    eyes_closed_artifact: bool = False  # episodul curent de ochi închiși a început la căscat / privit în jos
    # cât din semnalele de cameră au sosit (ponderat, 0..1): ~1 cu ambele laptopuri, ~0.7 doar fața,
    # ~0.45 doar corpul, 0 cu camera oprită. Matricea de probabilitate cântărește cadrul după el.
    camera_cov: float = 0.0

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
            "low_a_sec": round(self.low_a_sec, 2),
            "eyes_open_now": self.eyes_open_now,
            "eyes_closed_sec": round(self.eyes_closed_sec, 2),
            "ear_available": self.ear_available,
            "eyes_closed_artifact": self.eyes_closed_artifact,
            "camera_cov": round(self.camera_cov, 3),
        }


class AttentionFusion:
    def __init__(self, baseline_ear: float | None = None, initial_a: float = 1.0):
        self.baseline_ear = baseline_ear
        self.baseline_blink: float | None = None
        self.perclos_offset = 0.0   # din calibrare: PERCLOS „din construcție” al șoferului
        self._face_missing_since: float | None = None
        self._ear_closed_since: float | None = None
        self._art_episode_start: float | None = None
        self.eyes_closed_artifact = False
        self.eyes_open_now = False
        self.eyes_closed_sec = 0.0
        self.ear_available = False
        self._low_since: float | None = None
        self.low_a_sec = 0.0        # de cât timp a < LOW_A (pentru regula generică de severitate 1)
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
            perclos = _num(msg.get("perclos"))
            # PERCLOS pe 60 s (CV) întârzie zeci de secunde și la debut, și la revenire. Când avem închiderea
            # pe ultimele 15 s (de la CV sau estimarea ML1, verificată că e consistentă cu CV), o folosim.
            recent = _num(msg.get("perclos_recent")) if P["perclos_recent"] else None
            if recent is not None:
                perclos = recent
            if perclos is not None:
                perclos = max(0.0, perclos - self.perclos_offset)
            gaze = None if not isinstance(msg.get("gaze_on_road"), bool) else float(msg["gaze_on_road"])
            if msg.get("face_detected") is False:
                if self._face_missing_since is None:
                    self._face_missing_since = ts
                # Fața nu se vede, deși camera merge: lipsa e informație (MNAR, Curs 2) — șoferul
                # nu privește spre drum. DAR doar dacă lipsește ≥ FACE_MISSING_SEC: un cadru pierdut
                # de detector (pâlpâire) nu e neatenție (evaluarea pe domeniul „shift” a arătat
                # ~45 alarme false/oră din cauza asta).
                if gaze is None and ts - self._face_missing_since >= FACE_MISSING_SEC:
                    gaze = 0.0
            else:
                self._face_missing_since = None
            cand = {
                "eyes": score_eyes(perclos, _num(msg.get("ear")), self.baseline_ear,
                                   eye_artifact(msg.get("events"), _num(msg.get("phone")), msg.get("direction"))),
                "gaze": gaze,
                "head": None if not isinstance(msg.get("head_drop"), bool) else (0.0 if msg["head_drop"] else 1.0),
                "phone": None if _num(msg.get("phone")) is None else 1.0 - _clamp(_num(msg["phone"])),
                "blink": score_blink(_num(msg.get("blink_rate")), self.baseline_blink),
                "events": score_events(msg.get("events")),
            }
            scores.update({k: v for k, v in cand.items() if v is not None})

            # „Ochi închiși ACUM” (Curs 2: semnale cu scări de timp diferite nu se mediază orbește):
            # PERCLOS e istoria pe 60 s și urcă încet; o închidere susținută (>= 0.4 s, deci nu o clipire)
            # sau un eveniment CV de ochi închiși plafonează direct scorul ochilor. Altfel un șofer
            # prăbușit ar avea scorul ochilor ~0.7 pentru că „ultimul minut a fost normal”.
            art = eye_artifact(msg.get("events"), _num(msg.get("phone")), msg.get("direction"))
            # Un episod de ochi închiși care a ÎNCEPUT în timpul unui căscat / privit în jos rămâne artefact
            # până se termină, chiar dacă evenimentul de căscat a dispărut între timp (tracker-ul CV
            # prelungește episodul peste pauze < 1 s).
            ec = next((e for e in msg.get("events") or [] if isinstance(e, dict) and e.get("event") == "eyes_closed"), None)
            if ec is None:
                self._art_episode_start = None
            else:
                start = ts - (_num(ec.get("duration_sec")) or 0.0)
                if art:
                    self._art_episode_start = start
                elif self._art_episode_start is not None and abs(start - self._art_episode_start) <= 0.6:
                    art = True
            self.eyes_closed_artifact = bool(art and ec is not None)
            ear_now = _num(msg.get("ear"))
            if ear_now is not None and not art and ear_now < 0.6 * (self.baseline_ear or 0.30):
                self._ear_closed_since = ts if self._ear_closed_since is None else self._ear_closed_since
            else:
                self._ear_closed_since = None
            closed_now = self._ear_closed_since is not None and ts - self._ear_closed_since >= 0.4
            if not art:
                for e in msg.get("events") if isinstance(msg.get("events"), list) else []:
                    if (isinstance(e, dict) and e.get("event") in ("eyes_closed", "microsleep")
                            and (_num(e.get("confidence")) or 0.0) >= 0.6):
                        closed_now = True
            if closed_now:
                scores["eyes"] = min(scores.get("eyes", 1.0), 0.1)
            self.eyes_closed_sec = (ts - self._ear_closed_since) if self._ear_closed_since is not None else 0.0
            self.ear_available = ear_now is not None and not art
            self.eyes_open_now = (ear_now is not None and not art
                                  and ear_now > 0.85 * (self.baseline_ear or 0.30))

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
        eyes_closed = scores.get("eyes", 1.0) < 0.3   # include închiderea susținută de mai sus
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
        if self.a < LOW_A:
            self._low_since = ts if self._low_since is None else self._low_since
            self.low_a_sec = ts - self._low_since
        else:
            self._low_since, self.low_a_sec = None, 0.0

        # încredere -----------------------------------------------------------
        confidence = 0.0
        cam_cov = 0.0
        if any(k in scores for k in CAMERA_SIGNALS):
            cam_cov = sum(WEIGHTS[k] for k in CAMERA_SIGNALS if k in scores) / sum(WEIGHTS[k] for k in CAMERA_SIGNALS)
            confidence += SOURCE_CONFIDENCE["camera"] * cam_cov
        if "voice" in scores:
            confidence += SOURCE_CONFIDENCE["voice"]
        if "driving" in scores:
            confidence += SOURCE_CONFIDENCE["driving"]

        return FusionResult(ts, self.a, a_raw, _clamp(confidence), camera_on, bool(critical),
                            contributions, capped_by, self.low_a_sec, self.eyes_open_now if camera_on else False,
                            self.eyes_closed_sec if camera_on else 0.0, self.ear_available if camera_on else False,
                            self.eyes_closed_artifact if camera_on else False, cam_cov)

    @staticmethod
    def _camera_on(msg: dict) -> bool:
        if isinstance(msg.get("camera_on"), bool):
            return msg["camera_on"]
        return any(k in msg for k in ("ear", "perclos", "gaze_on_road", "head_drop", "phone"))
