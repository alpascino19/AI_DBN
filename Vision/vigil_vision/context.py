"""FrameContext: rezultatele unui cadru, citite de toți analizorii. [comun]

Fluxul pentru un cadru:
    perception (MediaPipe / YOLO)  -> umple pose, hands, objects (și face de la CV1)
    analizori (body/*, face/*)     -> citesc ctx, scriu ctx.signals și ctx.detections
    EventTracker                   -> ctx.detections -> evenimente cu durată
Coordonatele sunt normalizate 0..1 (x spre dreapta, y în jos), ca în MediaPipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .camera import Frame


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class Hand:
    landmarks: list[Landmark]          # 21 puncte MediaPipe Hands
    handedness: str = "Unknown"        # "Left" / "Right"
    score: float = 1.0

    def palm_center(self) -> tuple[float, float]:
        idx = (0, 5, 9, 13, 17)        # încheietura + bazele degetelor
        return (sum(self.landmarks[i].x for i in idx) / 5,
                sum(self.landmarks[i].y for i in idx) / 5)


@dataclass
class BBox:
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float                           # normalizate 0..1

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x: float, y: float, pad: float = 0.0) -> bool:
        w, h = self.x2 - self.x1, self.y2 - self.y1
        return (self.x1 - pad * w <= x <= self.x2 + pad * w and
                self.y1 - pad * h <= y <= self.y2 + pad * h)


@dataclass
class Detection:
    """Ce a văzut un analizor în ACEST cadru (fără durată; durata o face tracker-ul)."""
    name: str
    active: bool
    confidence: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceOutput:
    """Ce trimite CV1 pentru un cadru (vezi pipeline.py: hook-ul pentru față).
    signals: ear, perclos, blink_rate, gaze_on_road, head_drop, jaw_open, ...
    """
    signals: dict[str, Any] = field(default_factory=dict)
    detections: dict[str, Detection] = field(default_factory=dict)
    landmarks: list = field(default_factory=list)
    detected: bool = False
    timestamp: float = 0.0


@dataclass
class FrameContext:
    frame: Frame
    t: float                                        # secunde (monotonic)
    pose: Optional[list[Landmark]] = None           # 33 puncte MediaPipe Pose
    hands: list[Hand] = field(default_factory=list)
    objects: list[BBox] = field(default_factory=list)
    face: Optional[FaceOutput] = None               # de la CV1 (poate lipsi)
    signals: dict[str, Any] = field(default_factory=dict)
    detections: dict[str, Detection] = field(default_factory=dict)
    events: list = field(default_factory=list)      # [schemas.Event], pus de tracker
    geom: Any = None                                # body.geometry.Geom (unitate, direcție)

    def signal(self, key: str, default: Any = None) -> Any:
        """Caută un semnal întâi în ctx.signals, apoi în ce a trimis CV1."""
        if key in self.signals:
            return self.signals[key]
        if self.face is not None:
            return self.face.signals.get(key, default)
        return default

    def add(self, det: Detection) -> None:
        self.detections[det.name] = det


class Analyzer(Protocol):
    """Orice analizor (chest, breathing, phone, eyes, ...)."""
    def update(self, ctx: FrameContext) -> None: ...


# Indici MediaPipe Pose folosiți în CV2
class P:
    NOSE = 0
    MOUTH_L, MOUTH_R = 9, 10
    SHOULDER_L, SHOULDER_R = 11, 12
    ELBOW_L, ELBOW_R = 13, 14
    WRIST_L, WRIST_R = 15, 16
    INDEX_L, INDEX_R = 19, 20
    HIP_L, HIP_R = 23, 24
