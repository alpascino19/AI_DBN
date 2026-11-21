"""Stiva CV1 completă pentru un cadru: față -> ochi, privire, căscat, calibrare. [CV1]

Aceasta e "fața" pe care o folosește pipeline-ul CV2 (pipeline.py, FaceWorker):
    stack = create(cfg)                  # din vigil_vision.face
    out = stack.process(frame)           # -> context.FaceOutput

FaceOutput.signals (contractul vigil/vision + extra):
    ear, perclos, blink_rate, gaze_on_road, head_drop, jaw_open,
    eyes_closed, closed_sec, direction, yaw, pitch, yawning, yawns_recent, calibrated
FaceOutput.detections (trec prin EventTracker, nume pe care le știe ML1):
    eyes_closed, microsleep, gaze_off_road, head_drop, unconscious, yawn, hyper_blinking, low_blink_rate

Ultimele stări (EyeState, HeadState, YawnState, FaceResult) rămân în `stack.last`
pentru monitor / dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..camera import Frame
from ..context import Detection, FaceOutput
from ..perception.face import FaceDetector, FaceResult, Idx
from .calibration import Calibrator
from .eyes import EyeAnalyzer, EyeState
from .head_pose import HeadPoseAnalyzer, HeadState
from .unconscious import UnconsciousAnalyzer, UnconsciousState
from .yawn import YawnAnalyzer, YawnState

# subset de puncte pentru scheletul feței (privacy mode / dashboard), nu toate 478
SKELETON_IDX = sorted(set(
    Idx.RIGHT_EYE + Idx.LEFT_EYE
    + (Idx.NOSE_TIP, Idx.CHIN, Idx.FOREHEAD, Idx.MOUTH_LEFT, Idx.MOUTH_RIGHT, Idx.LIP_TOP, Idx.LIP_BOTTOM)
    # conturul feței
    + (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
       152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
    # sprâncene
    + (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
))



@dataclass
class FaceSnapshot:
    res: Optional[FaceResult]
    eyes: EyeState
    head: HeadState
    yawn: YawnState
    unconscious: UnconsciousState


class FaceStack:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        fc = (cfg or {}).get("face", {})
        calib_sec = float(fc.get("calib_sec", (cfg or {}).get("view", {}).get("calib_sec", 30)))
        self.detector = FaceDetector()
        self.eyes = EyeAnalyzer()
        self.head = HeadPoseAnalyzer()
        self.yawn = YawnAnalyzer()
        self.unconscious = UnconsciousAnalyzer()
        self.calib = Calibrator(duration_sec=calib_sec, min_samples=1 if calib_sec == 0 else 30)
        self.last: Optional[FaceSnapshot] = None

    def process(self, frame: Frame) -> FaceOutput:
        t = frame.timestamp
        res = self.detector.process(frame)
        es = self.eyes.update(res, t)
        hs = self.head.update(res, t)
        ys = self.yawn.update(res, t)
        if not self.calib.done and self.calib.update(es, hs, t):
            self.calib.apply(self.eyes, self.head)
        us = self.unconscious.update(es, hs, t)
        self.last = FaceSnapshot(res, es, hs, ys, us)
        return self._output(res, es, hs, ys, us, t)

    def detections(self) -> dict[str, tuple[bool, float]]:
        """Detecțiile brute ale analizorilor (pentru face/alerts.py)."""
        return {**self.eyes.detections(), **self.yawn.detections(), **self.unconscious.detections()}

    # ---------- FaceOutput pentru pipeline / vigil/vision ----------
    def _output(self, res, es: EyeState, hs: HeadState, ys: YawnState, us: UnconsciousState,
                t: float) -> FaceOutput:
        head_drop = us.head_drop if hs.face_visible or us.state == "unconscious" else None
        signals = {
            "ear": es.ear, "perclos": es.perclos, "blink_rate": es.blink_rate,
            "gaze_on_road": hs.gaze_on_road, "head_drop": head_drop, "jaw_open": ys.jaw_open,
            "eyes_closed": es.eyes_closed, "closed_sec": round(es.closed_sec, 2),
            "direction": hs.direction, "yaw": hs.yaw, "pitch": hs.pitch,
            "yawning": ys.yawning, "yawns_recent": ys.yawns_recent,
            "consciousness": us.state, "unconscious_score": us.score,
            "calibrated": self.calib.done,
        }
        raw = self.detections()
        dets: dict[str, Detection] = {}

        def add(name: str, active: bool, conf: float, **attrs) -> None:
            dets[name] = Detection(name, bool(active), round(float(conf), 2), attrs)

        add("eyes_closed", es.eyes_closed, 0.6 + 0.4 * (es.closure or 0.0),
            closed_sec=round(es.closed_sec, 1))
        # microsleep: tracker-ul adaugă min_duration (implicit 1 s) -> eveniment la ~1.5 s închiși
        add("microsleep", es.eyes_closed and es.closed_sec >= 0.5, raw["eyes_closed_long"][1] or 0.6)
        add("gaze_off_road", hs.gaze_on_road is False, 0.7, direction=hs.direction)
        add("head_drop", *raw["head_drop"])
        add("unconscious", *raw["unconscious"], score=us.score, still_sec=us.still_sec, sudden=us.sudden)
        add("yawn", ys.yawning, raw["yawning"][1])
        for name in ("hyper_blinking", "low_blink_rate"):
            if name in raw:
                add(name, *raw[name])

        landmarks = []
        if res is not None:
            landmarks = [[round(float(res.landmarks[i, 0]), 3), round(float(res.landmarks[i, 1]), 3)]
                         for i in SKELETON_IDX]
        return FaceOutput(signals=signals, detections=dets, landmarks=landmarks,
                          detected=res is not None, timestamp=t)

    def close(self) -> None:
        self.detector.close()
