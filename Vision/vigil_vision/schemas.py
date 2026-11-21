"""Contractul mesajului vigil/vision. [comun: CV1 + CV2]

Exemplu (TASKS.md):
{"ear":0.21,"perclos":0.3,"blink_rate":28,"gaze_on_road":false,"head_drop":false,
 "phone":0.1,"events":[{"event":"hand_on_chest","duration_sec":4,"confidence":0.8,
 "breathing":"heavy"}],"landmarks":[...]}

Câmpurile de bază sunt cele din contract; restul (ts, camera_on, breathing,
body_landmarks...) sunt extra și pot fi ignorate de ML1.
Fără pydantic, ca să nu depindă de nimic: to_dict() -> json.dumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Numele evenimentelor (folosite de tracker și de LLM)
HAND_ON_CHEST = "hand_on_chest"      # CV2
HEAVY_BREATHING = "heavy_breathing"  # CV2
PHONE_USE = "phone_use"              # CV2
HEAD_ON_WHEEL = "head_on_wheel"      # CV2: prăbușit în față / capul pe volan
# CV1 poate adăuga: "eyes_closed", "head_drop", "yawn", "gaze_off_road", ...


@dataclass
class Event:
    event: str
    duration_sec: float
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)   # ex. {"breathing": "heavy"}

    def to_dict(self) -> dict[str, Any]:
        d = {"event": self.event,
             "duration_sec": round(self.duration_sec, 1),
             "confidence": round(self.confidence, 2)}
        d.update(self.extra)
        return d


@dataclass
class VisionMessage:
    # --- CV1 (față) ---
    ear: Optional[float] = None
    perclos: Optional[float] = None
    blink_rate: Optional[float] = None
    gaze_on_road: Optional[bool] = None
    head_drop: Optional[bool] = None
    # --- CV2 (corp) ---
    phone: float = 0.0                        # scor 0..1 pentru telefon
    breathing: Optional[dict[str, Any]] = None  # {"state":"heavy","rate_bpm":30,...}
    posture: Optional[dict[str, Any]] = None    # {"state":"slumped","still_sec":6,...}
    # --- comun ---
    events: list[Event] = field(default_factory=list)
    landmarks: list = field(default_factory=list)        # scheletul feței (CV1)
    body_landmarks: list = field(default_factory=list)   # umeri/brațe (CV2), [[x,y],...]
    ts: float = 0.0                 # time.time()
    camera_on: bool = True
    face_detected: bool = False
    body_detected: bool = False
    fps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        r = lambda v, n=3: None if v is None else round(float(v), n)
        return {
            "ear": r(self.ear), "perclos": r(self.perclos), "blink_rate": r(self.blink_rate, 1),
            "gaze_on_road": self.gaze_on_road, "head_drop": self.head_drop,
            "phone": round(self.phone, 2),
            "breathing": self.breathing,
            "posture": self.posture,
            "events": [e.to_dict() for e in self.events],
            "landmarks": self.landmarks,
            "body_landmarks": self.body_landmarks,
            "ts": round(self.ts, 3), "camera_on": self.camera_on,
            "face_detected": self.face_detected, "body_detected": self.body_detected,
            "fps": round(self.fps, 1),
        }
