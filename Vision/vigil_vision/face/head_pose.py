"""Pitch/yaw/roll + direcția privirii -> gaze_on_road. [CV1]

Unghiurile se calculează din landmarks 3D (fără solvePnP):
    vectorul "în afara feței" = vârful nasului - centrul (colțuri ochi + colțuri gură)
    yaw   > 0  = capul spre DREAPTA șoferului (imaginea e oglindită)
    pitch > 0  = capul în JOS
    roll  > 0  = capul înclinat spre umărul drept
Unghiurile raportate sunt RELATIVE la poziția neutră din calibrare.

Privirea (iris) se adaugă peste cap: chiar cu capul drept, ochii pot fi
mutați lateral sau în jos (telefon în poală).

Utilizare:
    head = HeadPoseAnalyzer()
    st = head.update(face_result, frame.timestamp)
    st.yaw, st.pitch, st.direction, st.gaze_on_road, st.off_road_sec
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..perception.face import FaceResult, Idx

CENTER, LEFT, RIGHT, DOWN, UP = "center", "left", "right", "down", "up"


@dataclass
class HeadConfig:
    yaw_limit: float = 25.0        # grade față de neutru, stânga/dreapta
    pitch_down_limit: float = 18.0
    pitch_up_limit: float = 20.0
    iris_x_limit: float = 0.18     # deplasarea irisului în ochi (-0.5..0.5)
    iris_gain_deg: float = 60.0    # cât "adaugă" irisul la yaw-ul efectiv
    eye_down_limit: float = 0.35   # eyeLookDown (relativ la neutru) peste = privire în jos cu ochii
    eye_down_gain_deg: float = 35.0  # cât "adaugă" privirea în jos la pitch-ul efectiv
    smoothing: float = 0.35        # EMA: 1 = fără netezire
    window_sec: float = 10.0       # pentru off_road_ratio


@dataclass
class HeadState:
    face_visible: bool = False
    yaw: Optional[float] = None        # relativ la neutru, grade
    pitch: Optional[float] = None
    roll: Optional[float] = None
    yaw_raw: Optional[float] = None    # absolut (pentru calibrare)
    pitch_raw: Optional[float] = None
    iris_x: Optional[float] = None     # >0 = ochii spre dreapta
    eye_down: Optional[float] = None   # >0 = ochii privesc în jos (telefon în poală)
    direction: Optional[str] = None    # center / left / right / down / up
    gaze_on_road: Optional[bool] = None
    off_road_sec: float = 0.0          # de cât timp privește continuu în afara drumului
    off_road_ratio: Optional[float] = None  # fracțiunea din ultimele 10 s

    def to_dict(self) -> dict:
        return asdict(self)


def head_angles(face: FaceResult) -> tuple[float, float, float]:
    """(yaw, pitch, roll) absolute, în grade, din landmarks."""
    w, h = face.width, face.height
    p = face.landmarks * np.array([w, h, w], dtype=np.float32)  # z are scara lui x
    center = p[[Idx.RIGHT_EYE[0], Idx.LEFT_EYE[3], Idx.MOUTH_LEFT, Idx.MOUTH_RIGHT]].mean(axis=0)
    n = p[Idx.NOSE_TIP] - center           # iese din față, spre cameră (z negativ)
    yaw = math.degrees(math.atan2(n[0], -n[2]))
    pitch = math.degrees(math.atan2(n[1], -n[2]))
    eye_r, eye_l = p[Idx.RIGHT_EYE[0]], p[Idx.LEFT_EYE[3]]
    roll = math.degrees(math.atan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0]))
    return yaw, pitch, roll


def iris_offset(face: FaceResult) -> Optional[float]:
    """Poziția orizontală a irisurilor în ochi: -0.5 (stânga) .. 0.5 (dreapta)."""
    if len(face.landmarks) <= Idx.LEFT_IRIS:
        return None
    x = face.landmarks[:, 0]
    vals = []
    for iris, (a, b) in ((Idx.RIGHT_IRIS, (Idx.RIGHT_EYE[0], Idx.RIGHT_EYE[3])),
                         (Idx.LEFT_IRIS, (Idx.LEFT_EYE[0], Idx.LEFT_EYE[3]))):
        lo, hi = sorted((x[a], x[b]))
        if hi - lo > 1e-6:
            vals.append((x[iris] - lo) / (hi - lo) - 0.5)
    return float(np.mean(vals)) if vals else None


def eye_down_score(face: FaceResult) -> Optional[float]:
    """Cât privesc ochii în jos (blendshapes eyeLookDownLeft/Right), 0..1.
    Cu telefonul în mână mulți șoferi lasă doar ochii în jos, nu capul."""
    if not face.blendshapes:
        return None
    return (face.blend("eyeLookDownLeft") + face.blend("eyeLookDownRight")) / 2.0


class HeadPoseAnalyzer:
    def __init__(self, config: Optional[HeadConfig] = None) -> None:
        self.cfg = config or HeadConfig()
        self.yaw0 = 0.0
        self.pitch0 = 0.0
        self.iris0 = 0.0
        self.down0 = 0.0
        self.state = HeadState()
        self._ema: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None
        self._off_since: Optional[float] = None
        self._samples: deque[tuple[float, float, bool]] = deque()  # (t, dt, off_road)

    def set_neutral(self, yaw0: float, pitch0: float, iris0: float = 0.0, down0: float = 0.0) -> None:
        """Poziția "mă uit la drum" a șoferului (din calibrare)."""
        self.yaw0, self.pitch0, self.iris0, self.down0 = yaw0, pitch0, iris0, down0

    def update(self, face: Optional[FaceResult], t: float) -> HeadState:
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        s = self.state
        if face is None:
            s.face_visible = False
            s.yaw = s.pitch = s.roll = s.iris_x = s.eye_down = s.direction = s.gaze_on_road = None
            self._ema = None
            self._off_since = None
            s.off_road_sec = 0.0
            self._window(t)
            return s

        yaw, pitch, roll = head_angles(face)
        iris = iris_offset(face)
        return self.update_angles(yaw, pitch, roll, iris, t, dt, eye_down=eye_down_score(face))

    def update_angles(self, yaw: float, pitch: float, roll: float, iris: Optional[float],
                      t: float, dt: Optional[float] = None,
                      eye_down: Optional[float] = None) -> HeadState:
        """Ca update(), dar direct cu unghiuri (pentru teste / date false)."""
        if dt is None:
            dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
            self._t_prev = t
        c, s = self.cfg, self.state
        raw = np.array([yaw, pitch, roll,
                        iris if iris is not None else self.iris0,
                        eye_down if eye_down is not None else self.down0], dtype=np.float64)
        self._ema = raw if self._ema is None else c.smoothing * raw + (1 - c.smoothing) * self._ema
        yaw_s, pitch_s, roll_s, iris_s, down_s = self._ema

        s.face_visible = True
        s.yaw_raw, s.pitch_raw = float(yaw_s), float(pitch_s)
        s.yaw = round(float(yaw_s - self.yaw0), 1)
        s.pitch = round(float(pitch_s - self.pitch0), 1)
        s.roll = round(float(roll_s), 1)
        s.iris_x = None if iris is None else round(float(iris_s - self.iris0), 3)
        s.eye_down = None if eye_down is None else round(float(down_s - self.down0), 3)

        # yaw / pitch efectiv = cap + ochi
        eff_yaw = s.yaw + (s.iris_x or 0.0) * c.iris_gain_deg
        eff_pitch = s.pitch + max(0.0, s.eye_down or 0.0) * c.eye_down_gain_deg
        if eff_pitch > c.pitch_down_limit or (s.eye_down or 0.0) > c.eye_down_limit:
            s.direction = DOWN
        elif s.pitch < -c.pitch_up_limit:
            s.direction = UP
        elif eff_yaw > c.yaw_limit or (s.iris_x or 0) > c.iris_x_limit:
            s.direction = RIGHT
        elif eff_yaw < -c.yaw_limit or (s.iris_x or 0) < -c.iris_x_limit:
            s.direction = LEFT
        else:
            s.direction = CENTER
        s.gaze_on_road = s.direction == CENTER

        if s.gaze_on_road:
            self._off_since = None
        elif self._off_since is None:
            self._off_since = t
        s.off_road_sec = 0.0 if self._off_since is None else round(t - self._off_since, 2)

        self._samples.append((t, dt, not s.gaze_on_road))
        self._window(t)
        return s

    def _window(self, t: float) -> None:
        while self._samples and self._samples[0][0] < t - self.cfg.window_sec:
            self._samples.popleft()
        total = sum(dt for _, dt, _ in self._samples)
        off = sum(dt for _, dt, o in self._samples if o)
        self.state.off_road_ratio = round(off / total, 3) if total > 1.0 else None
