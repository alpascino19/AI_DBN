"""Schelet uman fals (mock) generat din cadrele `vigil/vision`.

Până vine CV 2 cu MediaPipe Pose, dashboard-ul desenează corpul din aceste puncte.
Formatul e simplu și poate fi înlocuit 1:1 cu landmark-urile reale (normalizate 0..1,
x spre dreapta, y în jos, șofer văzut din față):

    {"head": [x, y, r], "neck": [x, y], "sh_l": ..., "sh_r": ..., "el_l": ..., "el_r": ...,
     "wr_l": ..., "wr_r": ..., "hip_l": ..., "hip_r": ..., "chest": [x, y],
     "tilt": rad, "eyes": {"open": 0..1, "gaze": [dx, dy]}, "mouth": 0..1,
     "phone": [x, y] | null, "breath": -1..1, "flags": {...}}

Corpul reacționează la semnale: telefon (mâna la față), mână pe piept, cap căzut,
respirație grea (umeri mai mari și mai rapizi), ochi închiși, privire în jos, căscat.
"""

from __future__ import annotations

import math


def _f(v, default=None):
    if isinstance(v, bool) or v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _lerp(a, b, k):
    return a + (b - a) * k


def _mix(p, q, k):
    return [_lerp(p[0], q[0], k), _lerp(p[1], q[1], k)]


def _r(p):
    return [round(p[0], 4), round(p[1], 4)]


class PoseModel:
    RATE = 3.5  # cât de repede ajunge corpul la poziția țintă (1/s)

    def __init__(self):
        self.s = {"chest": 0.0, "drop": 0.0, "phone": 0.0, "heavy": 0.0, "eyes": 1.0,
                  "gx": 0.0, "gy": 0.0, "yawn": 0.0}
        self.phase = 0.0
        self.last_t: float | None = None

    def update(self, msg: dict, t: float) -> dict:
        msg = msg if isinstance(msg, dict) else {}
        dt = 0.2 if self.last_t is None else _clamp(t - self.last_t, 0.0, 1.0)
        self.last_t = t
        k = _clamp(dt * self.RATE)

        active = {}
        evs = msg.get("events")
        for e in evs if isinstance(evs, list) else []:
            if isinstance(e, dict) and isinstance(e.get("event"), str) and (_f(e.get("confidence"), 0) >= 0.5):
                active[e["event"]] = e
        ear = _f(msg.get("ear"))
        perclos = _f(msg.get("perclos"), 0.0)
        phone = _f(msg.get("phone"), 0.0)
        gaze_on = msg.get("gaze_on_road") is not False
        br = msg.get("breathing")
        breathing_heavy = ("heavy_breathing" in active or (active.get("hand_on_chest") or {}).get("breathing") == "heavy"
                           or (isinstance(br, dict) and br.get("state") == "heavy") or msg.get("breathing_state") == "heavy")

        target = {
            "chest": 1.0 if "hand_on_chest" in active else 0.0,
            "drop": 1.0 if (msg.get("head_drop") is True or "head_drop" in active or "head_on_wheel" in active)
                    else _clamp(perclos - 0.15) * 0.8,
            "phone": 1.0 if (phone >= 0.5 or "phone_in_hand" in active or "phone_use" in active) else 0.0,
            "heavy": 1.0 if breathing_heavy else 0.0,
            "eyes": _clamp(((ear if ear is not None else 0.30) - 0.09) / (0.27 - 0.09)),
            "gx": 0.0 if gaze_on else -0.6,
            "gy": 0.0 if gaze_on else 0.7,
            "yawn": 1.0 if "yawn" in active else 0.0,
        }
        s = self.s
        for key, tv in target.items():
            s[key] += (tv - s[key]) * k

        self.phase += dt * 2 * math.pi * (0.25 + 0.55 * s["heavy"])
        breath = math.sin(self.phase)
        amp = 0.004 + 0.016 * s["heavy"]
        br = breath * amp

        drop, ph, ch = s["drop"], s["phone"], s["chest"]
        slump = 0.035 * drop

        head = [0.5 + 0.03 * ph - 0.02 * drop, 0.21 + 0.16 * drop + 0.012 * s["yawn"], 0.085]
        neck = [0.5, 0.315 + 0.03 * drop]
        sh_l = [0.34, 0.37 + br + slump]
        sh_r = [0.66, 0.37 + br + slump]
        el_l = [0.28, 0.57 + slump]
        wr_l = [0.33, 0.75]

        el_r, wr_r = [0.72, 0.57 + slump], [0.67, 0.75]
        el_r = _mix(el_r, [0.69, 0.47], ph)
        wr_r = _mix(wr_r, [0.60, 0.29], ph)
        el_r = _mix(el_r, [0.64, 0.60], ch)
        wr_r = _mix(wr_r, [0.53, 0.48], ch)

        phone_pos = [wr_r[0] + 0.03, wr_r[1] - 0.05] if ph > 0.3 else None
        return {
            "head": [round(v, 4) for v in head],
            "neck": _r(neck), "sh_l": _r(sh_l), "sh_r": _r(sh_r),
            "el_l": _r(el_l), "el_r": _r(el_r), "wr_l": _r(wr_l), "wr_r": _r(wr_r),
            "hip_l": [0.43, 0.92], "hip_r": [0.57, 0.92],
            "chest": _r([0.5, 0.47 + br + slump * 0.6]),
            "tilt": round(0.55 * drop + 0.18 * ph, 3),
            "eyes": {"open": round(s["eyes"], 3), "gaze": [round(s["gx"], 3), round(s["gy"], 3)]},
            "mouth": round(max(0.55 * s["heavy"], s["yawn"]), 3),
            "phone": None if phone_pos is None else _r(phone_pos),
            "breath": round(breath, 3),
            "flags": {"chest": ch > 0.5, "drop": drop > 0.5, "phone": ph > 0.5, "heavy": s["heavy"] > 0.5,
                      "yawn": s["yawn"] > 0.5, "eyes_closed": s["eyes"] < 0.25, "gaze_off": not gaze_on},
        }
