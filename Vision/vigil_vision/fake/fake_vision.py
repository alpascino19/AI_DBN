"""Generator de date false pe vigil/vision (scenarii demo). [comun]

    python -m vigil_vision --fake                    # tot scenariul demo, în buclă
    python -m vigil_vision --fake --scenario chest   # o singură fază, în buclă

Mesajele au exact forma celor reale (schemas.VisionMessage), ca ML1/ML2 și
dashboard-ul să poată lucra fără cameră.
"""
from __future__ import annotations

import logging
import math
import random
import time
from typing import Callable

from ..publisher import Publisher
from ..schemas import HAND_ON_CHEST, HEAD_ON_WHEEL, HEAVY_BREATHING, PHONE_USE, Event, VisionMessage

log = logging.getLogger(__name__)


def _base(t: float) -> VisionMessage:
    return VisionMessage(
        ear=0.30 + 0.01 * math.sin(t), perclos=0.05, blink_rate=random.uniform(14, 18),
        gaze_on_road=True, head_drop=False, phone=0.0,
        breathing={"state": "normal", "rate_bpm": 15.0, "amp_ratio": 1.0, "mouth_open": False},
        posture={"state": "upright", "score": 0.0, "drop": 0.0, "still_sec": 0.0, "calibrated": True},
        ts=time.time(), face_detected=True, body_detected=True, fps=25.0,
    )


def normal(t: float) -> VisionMessage:
    return _base(t)


def phone(t: float) -> VisionMessage:
    m = _base(t)
    m.gaze_on_road = t % 4 > 2.5                          # se uită din când în când la drum
    m.phone = min(0.9, 0.5 + 0.1 * t)
    if t > 1:
        m.events.append(Event(PHONE_USE, t, 0.85, {"in_hand": True, "near_face": False,
                                                   "looking_at_phone": not m.gaze_on_road,
                                                   "usage": "looking"}))
    return m


def drowsy(t: float) -> VisionMessage:
    m = _base(t)
    m.ear, m.perclos, m.blink_rate = 0.17, min(0.6, 0.2 + 0.04 * t), 6
    if t > 3:
        m.events.append(Event("eyes_closed", min(t - 3, 2.5), 0.75))
    if 4 < t < 7:
        m.events.append(Event("yawn", t - 4, 0.7))
    return m


def chest(t: float) -> VisionMessage:
    m = _base(t)
    m.breathing = {"state": "heavy", "rate_bpm": 32.0, "amp_ratio": 3.1, "mouth_open": True}
    m.blink_rate = 34
    if t > 2:
        m.events.append(Event(HAND_ON_CHEST, t, min(0.9, 0.6 + 0.04 * t),
                              {"side": "left", "both_hands": False, "breathing": "heavy"}))
    if t > 3:
        m.events.append(Event(HEAVY_BREATHING, t - 1, 0.75, {"rate_bpm": 32.0, "mouth_open": True}))
    return m


def unconscious(t: float) -> VisionMessage:
    m = chest(t + 10)
    m.ear, m.perclos, m.blink_rate = 0.08, 0.95, 0
    m.head_drop, m.gaze_on_road = True, False
    m.events = [e for e in m.events if e.event != HAND_ON_CHEST]
    m.events += [Event("head_drop", t, 0.9), Event("eyes_closed", t, 0.9)]
    if t > 3:
        m.events.append(Event(HEAD_ON_WHEEL, t - 3, 0.85, {"still_sec": t - 1, "face_visible": False}))
    m.posture = {"state": "slumped" if t > 3 else "leaning", "score": 0.8, "drop": 1.0,
                 "still_sec": round(t, 1), "face_visible": False, "calibrated": True}
    m.face_detected = False
    return m


SCENARIOS: dict[str, Callable[[float], VisionMessage]] = {
    "normal": normal, "phone": phone, "drowsy": drowsy, "chest": chest, "unconscious": unconscious,
}
DEMO = [("normal", 10), ("phone", 8), ("normal", 5), ("drowsy", 12), ("normal", 5),
        ("chest", 12), ("unconscious", 15)]


def run_fake(cfg: dict, pub: Publisher, scenario: str = "demo", loop: bool = True) -> None:
    phases = DEMO if scenario == "demo" else [(scenario, 15)]
    for name, _ in phases:
        if name not in SCENARIOS:
            raise SystemExit(f"Scenariu necunoscut: {name}. Alege din: demo, {', '.join(SCENARIOS)}")
    period = 1.0 / max(1, cfg["publish_hz"])
    try:
        while True:
            for name, dur in phases:
                log.info("Scenariu fals: %s (%d s)", name, dur)
                t0 = time.monotonic()
                while (t := time.monotonic() - t0) < dur:
                    pub.publish(SCENARIOS[name](t))
                    time.sleep(period)
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
