"""Adaptor pentru datele echipei CV (pachetul `Vision/vigil_vision`).

1. `normalize_vision(msg)`: aduce mesajul real `vigil/vision` (schemas.VisionMessage) la forma
   pe care o folosește ML1, fără să piardă informație:
     * nume de evenimente: `phone_use` -> `phone_in_hand` (CV2 folosește alt nume);
       `head_on_wheel` (CV2: prăbușit pe volan) rămâne și e tratat ca semn grav;
     * `breathing` / `posture` de nivel superior -> `breathing_state`, `posture_state`, `still_sec`;
     * curățare (Curs 2): valori ne-finite -> None, decupare la intervale fizice (EAR, PERCLOS...);
     * `landmarks` / `body_landmarks` și panourile ferestrelor (`face_panel` / `body_panel`, doar pentru
       dashboard) eliminate: nu ajung la LLM, nu ocupă memorie;
     * fața lipsă (`face_detected: false`) e păstrată explicit: lipsa e informație (MNAR, Curs 2).
2. `cv_demo_messages()`: rulează generatorul fals al echipei CV (`vigil_vision.fake`) determinist,
   ca ML1 să fie testat pe exact forma mesajelor reale.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

EVENT_ALIASES = {
    "yawning": "yawn",                 # numele intern CV1 (yawn.py)
    "eyes_closed_long": "microsleep",  # numele intern CV1 (eyes.py)
    "phone_use": "phone_in_hand",      # CV2
    "phone": "phone_in_hand",
    "head_on_wheel": "head_on_wheel",  # CV2: prăbușit în față
    "slumped": "head_on_wheel",
}

# Intervale fizice plauzibile (decupare / winsorizare, Curs 2)
RANGES = {"ear": (0.0, 0.6), "perclos": (0.0, 1.0), "blink_rate": (0.0, 90.0), "phone": (0.0, 1.0)}


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
ENVELOPE_KEYS = ("payload", "data", "vision", "message", "body")
KNOWN = {"ear", "perclos", "blink_rate", "gaze_on_road", "head_drop", "phone", "events", "camera_on", "ts"}


def _snake(key: str) -> str:
    """camelCase (Jackson / Spring Boot) -> snake_case: blinkRate -> blink_rate."""
    return _CAMEL.sub("_", key).lower() if isinstance(key, str) and any(c.isupper() for c in key) else key


def _snake_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_snake(k): _snake_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snake_keys(v) for v in obj]
    return obj


def normalize_ts(v: Any) -> float | None:
    """Timpul în secunde epoch. Acceptă secunde, milisecunde (Java `currentTimeMillis`),
    microsecunde și șiruri ISO-8601 (`Instant.toString()` din Java)."""
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            v = _num(v)
    t = _num(v)
    if t is None:
        return None
    if t > 1e14:      # microsecunde
        return t / 1e6
    if t > 1e11:      # milisecunde
        return t / 1e3
    return t


def unwrap(msg: Any) -> Any:
    """Scoate mesajul din plicuri de tip {"type": "...", "payload": {...}} (server / broker)."""
    for _ in range(3):
        if not isinstance(msg, dict) or KNOWN & set(map(_snake, msg)):
            return msg
        inner = next((msg[k] for k in ENVELOPE_KEYS if isinstance(msg.get(k), dict)), None)
        if inner is None:
            return msg
        msg = inner
    return msg


def normalize_vision(msg: Any) -> dict:
    msg = _snake_keys(unwrap(msg))
    if not isinstance(msg, dict):
        return {}
    out = {k: v for k, v in msg.items()
           if k not in ("landmarks", "body_landmarks", "face_panel", "body_panel", "events", "breathing", "posture")}
    if "ts" in msg:
        ts = normalize_ts(msg.get("ts"))
        if ts is None:
            out.pop("ts", None)
        else:
            out["ts"] = ts

    for key, (lo, hi) in RANGES.items():
        if key in msg:
            v = _num(msg.get(key))
            out[key] = None if v is None else min(hi, max(lo, v))

    events = []
    for e in msg.get("events") if isinstance(msg.get("events"), list) else []:
        if isinstance(e, dict) and isinstance(e.get("event"), str):
            events.append({**e, "event": EVENT_ALIASES.get(e["event"], e["event"])})
    out["events"] = events

    br = msg.get("breathing")
    if isinstance(br, dict) and isinstance(br.get("state"), str):
        out["breathing_state"] = br["state"]
        if br.get("state") == "heavy":
            for e in events:  # completăm atributul din contract dacă lipsește
                if e["event"] == "hand_on_chest":
                    e.setdefault("breathing", "heavy")
    elif isinstance(br, str):
        out["breathing_state"] = br

    po = msg.get("posture")
    if isinstance(po, dict) and isinstance(po.get("state"), str):
        out["posture_state"] = po["state"]
        s = _num(po.get("still_sec"))
        if s is not None:
            out["still_sec"] = s
    return out


# --------------------------------------------------------------- date CV ----

VISION_DIR = Path(__file__).resolve().parents[2] / "Vision"   # CodeLap/ML1/ml1 -> CodeLap/Vision


def _import_cv_fake():
    if str(VISION_DIR) not in sys.path:
        sys.path.insert(0, str(VISION_DIR))
    from vigil_vision.fake import fake_vision  # noqa: WPS433 (import la cerere)
    return fake_vision


def cv_available() -> bool:
    try:
        _import_cv_fake()
        return True
    except Exception:  # pachetul CV lipsește sau are dependențe neinstalate
        return False


def cv_demo_messages(hz: int = 5, seed: int = 7, phases: list[tuple[str, float]] | None = None) -> list[dict]:
    """Mesajele demo ale echipei CV (aceeași formă ca de la cameră), cu `ts` de la 0."""
    fv = _import_cv_fake()
    random.seed(seed)
    out, t0 = [], 0.0
    for name, dur in phases or fv.DEMO:
        gen = fv.SCENARIOS[name]
        n = int(dur * hz)
        for i in range(n):
            t = i / hz
            d = gen(t).to_dict()
            d["ts"] = round(t0 + t, 2)
            d["cv_phase"] = name
            out.append(d)
        t0 += dur
    return out
