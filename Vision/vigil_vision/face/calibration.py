"""Calibrarea din primele N secunde (implicit 30). [CV1]

Șoferul privește drumul normal. Strângem EAR-ul cu ochii deschiși și unghiurile
capului, apoi folosim mediana ca "normalul" acestui șofer:
    - EyeAnalyzer.set_open_baseline(ear_open)  -> prag de ochi închiși personal
    - HeadPoseAnalyzer.set_neutral(yaw0, pitch0, iris0) -> "drept înainte" personal
      (contează mult: camera laptopului e de obicei mai jos decât fața)

Utilizare:
    calib = Calibrator(duration_sec=30)
    ...per cadru:
        if not calib.done:
            calib.update(eye_state, head_state, t)
            if calib.done:
                calib.apply(eyes, head)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional

from .eyes import EyeAnalyzer, EyeState
from .head_pose import HeadPoseAnalyzer, HeadState


@dataclass
class Calibration:
    ear_open: float
    yaw0: float
    pitch0: float
    iris0: float
    samples: int
    down0: float = 0.0   # eyeLookDown normal când privește drumul

    def to_dict(self) -> dict:
        return asdict(self)


class Calibrator:
    def __init__(self, duration_sec: float = 30.0, min_samples: int = 30, min_ear: float = 0.15) -> None:
        self.duration_sec = duration_sec
        self.min_samples = min_samples
        self.min_ear = min_ear       # sub prag = clipire, nu intră în baseline
        self.result: Optional[Calibration] = None
        self._t0: Optional[float] = None
        self._ear: list[float] = []
        self._yaw: list[float] = []
        self._pitch: list[float] = []
        self._iris: list[float] = []
        self._down: list[float] = []

    @property
    def done(self) -> bool:
        return self.result is not None

    def progress(self, t: float) -> float:
        if self.duration_sec <= 0:
            return 1.0
        return 0.0 if self._t0 is None else min(1.0, (t - self._t0) / self.duration_sec)

    def update(self, eyes: EyeState, head: HeadState, t: float) -> bool:
        if self.done:
            return True
        if self._t0 is None:
            self._t0 = t
        if eyes.face_visible and eyes.ear is not None and eyes.ear > self.min_ear:
            self._ear.append(eyes.ear)
        if head.face_visible and head.yaw_raw is not None:
            self._yaw.append(head.yaw_raw)
            self._pitch.append(head.pitch_raw)
            if head.iris_x is not None:
                # iris_x e deja relativ la iris0 (= 0 înainte de calibrare)
                self._iris.append(head.iris_x)
            if head.eye_down is not None:
                self._down.append(head.eye_down)

        if t - self._t0 >= self.duration_sec and len(self._ear) >= self.min_samples and self._yaw:
            self.result = Calibration(
                ear_open=round(median(self._ear), 4),
                yaw0=round(median(self._yaw), 2),
                pitch0=round(median(self._pitch), 2),
                iris0=round(median(self._iris), 4) if self._iris else 0.0,
                samples=len(self._ear),
                down0=round(median(self._down), 4) if self._down else 0.0,
            )
        return self.done

    def apply(self, eyes: EyeAnalyzer, head: HeadPoseAnalyzer) -> None:
        if self.result is None:
            return
        eyes.set_open_baseline(self.result.ear_open)
        head.set_neutral(self.result.yaw0, self.result.pitch0, self.result.iris0, self.result.down0)
