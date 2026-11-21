"""Datele panoului din ferestrele `vigil - FATA` / `vigil - CORP`, pentru dashboard-ul React. [CV1 + CV2]

Aceleași valori pe care face/monitor.py le desenează în coloana din dreapta și în banda de avertizări,
ca dict JSON (numere rotunjite, fără tipuri numpy). run_face / run_body le adaugă la mesajul MQTT ca
`face_panel` / `body_panel`; merge.py le trece mai departe în vigil/vision. Câmpuri în plus față de
contract: ML1 le ignoră, iar un dashboard mai vechi la fel.
"""
from __future__ import annotations

from typing import Any, Optional


def _n(v: Any, nd: int = 2) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if f == f and abs(f) != float("inf") else None


def alerts(am) -> list[dict]:
    """Avertizările active (cele mai grave primele), ca în banda de jos a ferestrei."""
    return [{"level": int(a.level), "message": str(a.message), "sec": _n(a.active_sec, 0)}
            for a in getattr(am, "active", [])[:3]]


def face_panel(snap, calib, am, t: float, fps: float = 0.0) -> Optional[dict]:
    if snap is None:
        return None
    eyes, head, yawn, us = snap.eyes, snap.head, snap.yawn, snap.unconscious
    calibrating = calib is not None and not calib.done
    return {
        "fps": _n(fps, 0),
        "level": 0 if calibrating else int(am.level),
        "calibration": _n(calib.progress(t), 2) if calibrating else None,
        "eyes": {
            "visible": bool(eyes.face_visible), "closed": bool(eyes.eyes_closed),
            "closed_sec": _n(eyes.closed_sec, 1), "closure": _n(eyes.closure),
            "blink_rate": _n(eyes.blink_rate, 0), "perclos": _n(eyes.perclos, 3),
            "perclos_recent": _n(eyes.perclos_recent, 3), "blinks_recent": eyes.blinks_recent,
            "recovered": bool(eyes.recovered),
        },
        "head": {
            "visible": bool(head.face_visible), "direction": head.direction,
            "yaw": _n(head.yaw, 0), "pitch": _n(head.pitch, 0), "eye_down": _n(head.eye_down),
            "off_road_sec": _n(head.off_road_sec, 1), "off_road_ratio": _n(head.off_road_ratio),
        },
        "unconscious": {
            "state": us.state, "score": _n(us.score), "since_sec": _n(us.since_sec, 0),
            "dropped": bool(us.dropped), "still_sec": _n(us.still_sec, 1),
        },
        "yawn": {"yawning": bool(yawn.yawning), "open_sec": _n(yawn.open_sec, 1),
                 "recent": int(yawn.yawns_recent), "total": int(yawn.yawn_count)},
        "alerts": alerts(am) if not calibrating else [],
    }


def body_panel(ctx, am, fps: float = 0.0) -> Optional[dict]:
    if ctx is None:
        return None

    def det(name: str):
        d = ctx.detections.get(name)
        return d if d is not None and d.active else None

    s = ctx.signals
    b = s.get("breathing") or {}
    sl = s.get("slump") or {}
    pd = ctx.detections.get("phone_use")
    zone = s.get("chest_zone")
    geom = getattr(ctx, "geom", None)
    return {
        "fps": _n(fps, 0),
        "level": int(am.level),
        "visible": ctx.pose is not None,
        "chest": {"on": det("hand_on_chest") is not None, "src": s.get("chest_src"), "reject": s.get("chest_reject")},
        "hand_scale": _n(s.get("chest_hand_scale")),
        "breathing": {"state": b.get("state"), "score": _n(b.get("score")), "rate_bpm": _n(b.get("rate_bpm"), 0),
                      "calibrated": bool(b.get("calibrated", True))},
        "phone": {"score": _n(s.get("phone", 0.0)), "active": bool(pd and pd.active),
                  "usage": (pd.attrs or {}).get("usage") if pd else None},
        "posture": {"state": sl.get("state"), "calibrated": bool(sl.get("calibrated", True))},
        # pentru desen: zona pieptului și baza gâtului (linia posturii din fereastra CORP)
        "chest_zone": [_n(v, 3) for v in zone] if zone else None,
        "neck": [_n(geom.sx, 3), _n(geom.sy, 3)] if geom is not None else None,
        "alerts": alerts(am),
    }
