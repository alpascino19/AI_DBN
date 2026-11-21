"""EAR, PERCLOS, rata de clipire. [CV1]

Intrare:  FaceResult (din perception/face.py) sau None dacă nu e față.
Ieșire:   EyeState cu cifrele pentru `vigil/vision` + detecții brute
          (ochi închiși prelungit, clipit rar / prea des).

Ochi închiși = un SCOR DE ÎNCHIDERE 0..1, nu doar EAR sub un prag fix:
    1) EAR normalizat la ochii ACESTUI șofer: 0 = deschis normal, 1 = închis complet
       (referința de "închis" se învață din clipirile lui)
    2) blendshape-urile MediaPipe eyeBlinkLeft/Right (antrenate pe închidere)
    3) compensare la zâmbet / râs / ochi mijiți (mouthSmile*, eyeSquint*):
       obrajii împing pleoapa de jos și EAR scade, deși ochii sunt deschiși,
       așa că atunci cerem o închidere mai clară.
Ochi întredeschiși (~jumătate) NU mai sunt "închiși"; închis = aproape complet.

Toate ferestrele sunt în SECUNDE (timestamp-ul cadrului), nu în cadre.

Utilizare:
    eyes = EyeAnalyzer()
    for frame in reader:
        state = eyes.update(face.process(frame), frame.timestamp)
        state.ear, state.closure, state.perclos, state.blink_rate, state.closed_sec
        eyes.detections()   # {"eyes_closed_long": (True, 0.9), ...}

Calibrare (face/calibration.py): eyes.set_open_baseline(ear_mediu_ochi_deschiși)
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional

import numpy as np

from ..perception.face import FaceResult, Idx


@dataclass
class EyeConfig:
    ear_open: float = 0.28          # EAR cu ochii deschiși (înlocuit de calibrare)
    closed_ratio: float = 0.45      # EAR "închis complet" ≈ ear_open * ratio (până învățăm din clipiri)
    ear_weight: float = 0.6         # cât contează EAR vs. eyeBlink în scor (când avem blendshapes)
    close_on: float = 0.72          # scor peste = ochi închiși (~sub 25-30% deschiși)
    close_off: float = 0.50         # scor sub = ochi deschiși din nou (histerezis)
    smile_on: float = 0.35          # zâmbet / ochi mijiți peste prag -> compensare
    smile_close_on: float = 0.85    # pragul de închidere în timpul zâmbetului
    blink_min_sec: float = 0.05     # mai scurt = zgomot
    blink_max_sec: float = 0.5      # mai lung = nu mai e clipit, e închidere
    long_closed_sec: float = 1.5    # ochi închiși prelungit / micro-somn
    perclos_window_sec: float = 60.0
    blink_window_sec: float = 60.0
    blink_rate_low: float = 8.0     # clipiri/min: sub = privit în gol / somnolență
    blink_rate_high: float = 22.0  #clipiri/min: peste = clipit anormal
    min_window_sec: float = 10.0    # până atunci nu raportăm rate (prea puține date)
    # revenire rapidă: oboseala (fereastra de 60 s) se stinge dacă ultimele 15 s arată normal
    recovery_window_sec: float = 15.0
    recovery_perclos_max: float = 0.05   # ochi închiși max 5% din ultimele 15 s
    recovery_min_blinks: int = 2         # cel puțin atâtea clipiri normale în ultimele 15 s
    recovery_min_visible: float = 0.8    # fața văzută cel puțin 80% din cele 15 s


@dataclass
class EyeState:
    face_visible: bool = False
    ear: Optional[float] = None          # media celor doi ochi
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    closure: Optional[float] = None      # 0 = deschis normal, 1 = închis complet
    smile: Optional[float] = None        # max(zâmbet, ochi mijiți), 0..1
    eyes_closed: bool = False
    closed_sec: float = 0.0              # de cât timp sunt închiși acum
    perclos: Optional[float] = None      # 0..1 în ultimele 60 s
    blink_rate: Optional[float] = None   # clipiri / minut
    blink_count: int = 0                 # total de la pornire
    last_blink_sec: Optional[float] = None
    blink_score: Optional[float] = None  # media eyeBlinkLeft/Right
    perclos_recent: Optional[float] = None  # 0..1 în ultimele 15 s
    blinks_recent: int = 0                  # clipiri în ultimele 15 s
    recovered: bool = False                 # ultimele 15 s arată normal (stinge oboseala)

    def to_dict(self) -> dict:
        return asdict(self)


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|), puncte în ordinea din Idx."""
    p1, p2, p3, p4, p5, p6 = pts
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


class EyeAnalyzer:
    def __init__(self, config: Optional[EyeConfig] = None) -> None:
        self.cfg = config or EyeConfig()
        self.ear_open = self.cfg.ear_open
        self.ear_closed = self.cfg.ear_open * self.cfg.closed_ratio
        self.state = EyeState()

        self._t_prev: Optional[float] = None
        self._t_start: Optional[float] = None
        self._closed_since: Optional[float] = None
        self._min_ear_episode: Optional[float] = None
        self._blink_minima: deque[float] = deque(maxlen=20)        # EAR minim din clipiri
        self._samples: deque[tuple[float, float, bool]] = deque()  # (t, dt, closed)
        self._blinks: deque[float] = deque()                       # momentele clipirilor

    # ---------- calibrare ----------
    def set_open_baseline(self, ear_open: float) -> None:
        """EAR-ul tipic cu ochii deschiși al șoferului (din calibrare)."""
        self.ear_open = ear_open
        if len(self._blink_minima) < 3:
            self.ear_closed = ear_open * self.cfg.closed_ratio
        else:
            self._learn_closed()

    @property
    def threshold(self) -> float:
        """EAR echivalent pragului de închidere (doar pentru afișare / compatibilitate)."""
        return self.ear_open - self.cfg.close_on * (self.ear_open - self.ear_closed)

    # ---------- per cadru ----------
    def update(self, face: Optional[FaceResult], t: float) -> EyeState:
        if face is None:
            self._tick(t)
            s = self.state
            # Fără față nu știm starea ochilor: nu o numărăm în PERCLOS.
            s.face_visible = False
            s.ear = s.ear_left = s.ear_right = s.blink_score = s.closure = s.smile = None
            s.closed_sec = (t - self._closed_since) if (s.eyes_closed and self._closed_since is not None) else 0.0
            self._update_windows(t)
            return self.state

        pts = face.points_px
        ear_r = eye_aspect_ratio(pts[list(Idx.RIGHT_EYE)])
        ear_l = eye_aspect_ratio(pts[list(Idx.LEFT_EYE)])
        b = face.blend
        blink = (b("eyeBlinkLeft") + b("eyeBlinkRight")) / 2.0 if face.blendshapes else None
        smile = None
        if face.blendshapes:
            smile = max((b("mouthSmileLeft") + b("mouthSmileRight")) / 2.0,
                        (b("eyeSquintLeft") + b("eyeSquintRight")) / 2.0)
        return self._step(ear_l, ear_r, blink, smile, t)

    def update_ear(self, ear: float, t: float, blink: Optional[float] = None,
                   smile: Optional[float] = None) -> EyeState:
        """Ca update(), dar direct cu valori (pentru teste / date false)."""
        return self._step(ear, ear, blink, smile, t)

    # ---------- detecții brute pentru events/tracker.py ----------
    def detections(self) -> dict[str, tuple[bool, float]]:
        """{eveniment: (activ, încredere)}. Tracker-ul (CV2) le transformă în
        evenimente cu durată; aici doar spunem ce vedem acum."""
        s, c = self.state, self.cfg
        out: dict[str, tuple[bool, float]] = {}

        long_closed = s.eyes_closed and s.closed_sec >= c.long_closed_sec
        conf = min(1.0, s.closed_sec / (2 * c.long_closed_sec)) if long_closed else 0.0
        out["eyes_closed_long"] = (long_closed, round(0.5 + 0.5 * conf, 2) if long_closed else 0.0)

        if s.blink_rate is not None:
            low = s.blink_rate < c.blink_rate_low
            high = s.blink_rate > c.blink_rate_high
            out["low_blink_rate"] = (low, 0.6 if low else 0.0)
            out["hyper_blinking"] = (high, min(1.0, 0.5 + (s.blink_rate - c.blink_rate_high) / 40) if high else 0.0)
        return out

    # ---------- intern ----------
    def _tick(self, t: float) -> float:
        if self._t_start is None:
            self._t_start = t
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        return dt

    def _closure(self, ear: float, blink: Optional[float]) -> float:
        span = max(1e-6, self.ear_open - self.ear_closed)
        ear_c = float(np.clip((self.ear_open - ear) / span, 0.0, 1.0))
        if blink is None:
            return ear_c
        w = self.cfg.ear_weight
        return w * ear_c + (1 - w) * float(np.clip(blink, 0.0, 1.0))

    def _step(self, ear_l: float, ear_r: float, blink: Optional[float],
              smile: Optional[float], t: float) -> EyeState:
        dt = self._tick(t)
        s, c = self.state, self.cfg
        s.face_visible = True
        s.ear_left, s.ear_right = round(ear_l, 4), round(ear_r, 4)
        ear = (ear_l + ear_r) / 2.0
        s.ear = round(ear, 4)
        s.blink_score = None if blink is None else round(blink, 3)
        s.smile = None if smile is None else round(smile, 3)
        s.closure = round(self._closure(ear, blink), 3)

        close_on = c.smile_close_on if (smile or 0.0) > c.smile_on else c.close_on
        if not s.eyes_closed and s.closure > close_on:
            s.eyes_closed = True
            self._closed_since = t
            self._min_ear_episode = ear
        elif s.eyes_closed:
            self._min_ear_episode = min(self._min_ear_episode or ear, ear)
            if s.closure < c.close_off:
                dur = 0.0 if self._closed_since is None else t - self._closed_since
                s.eyes_closed = False
                self._closed_since = None
                if c.blink_min_sec <= dur <= c.blink_max_sec:
                    s.blink_count += 1
                    s.last_blink_sec = round(dur, 3)
                    self._blinks.append(t)
                    if self._min_ear_episode is not None:
                        self._blink_minima.append(self._min_ear_episode)
                        self._learn_closed()

        self._samples.append((t, dt, s.eyes_closed))
        s.closed_sec = (t - self._closed_since) if (s.eyes_closed and self._closed_since is not None) else 0.0
        self._update_windows(t)
        return s

    def _learn_closed(self) -> None:
        """Referința de "închis complet" = mediana EAR-ului minim din clipiri,
        limitată ca să nu ajungă absurdă (clipiri incomplete / zgomot)."""
        if len(self._blink_minima) >= 3:
            learned = median(self._blink_minima)
            self.ear_closed = float(np.clip(learned, 0.25 * self.ear_open, 0.6 * self.ear_open))

    def _update_windows(self, t: float) -> None:
        s, c = self.state, self.cfg
        while self._samples and self._samples[0][0] < t - c.perclos_window_sec:
            self._samples.popleft()
        while self._blinks and self._blinks[0] < t - c.blink_window_sec:
            self._blinks.popleft()

        elapsed = 0.0 if self._t_start is None else t - self._t_start
        if elapsed < c.min_window_sec:
            s.perclos = s.blink_rate = None
            return

        total = sum(dt for _, dt, _ in self._samples)
        closed = sum(dt for _, dt, cl in self._samples if cl)
        s.perclos = round(closed / total, 3) if total > 0 else None

        window = min(elapsed, c.blink_window_sec)
        s.blink_rate = round(len(self._blinks) * 60.0 / window, 1)

        # ----- ultimele 15 s (revenire rapidă) -----
        t_rec = t - c.recovery_window_sec
        rec_total = sum(dt for ts, dt, _ in self._samples if ts >= t_rec)
        rec_closed = sum(dt for ts, dt, cl in self._samples if ts >= t_rec and cl)
        s.perclos_recent = round(rec_closed / rec_total, 3) if rec_total > 0 else None
        s.blinks_recent = sum(1 for tb in self._blinks if tb >= t_rec)
        s.recovered = (
            elapsed >= c.recovery_window_sec
            and rec_total >= c.recovery_min_visible * c.recovery_window_sec
            and s.perclos_recent is not None and s.perclos_recent <= c.recovery_perclos_max
            and s.blinks_recent >= c.recovery_min_blinks
            and not s.eyes_closed
        )


# ---------- test live: python -m vigil_vision.face.eyes [--show] ----------
if __name__ == "__main__":
    import argparse
    import logging

    import cv2

    from ..camera import CV1, Camera
    from ..perception.face import FaceDetector

    p = argparse.ArgumentParser(description="Test EAR / PERCLOS / clipit pe cameră")
    p.add_argument("--index", default=0, type=int)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    eyes = EyeAnalyzer()
    with Camera(index=args.index) as cam, FaceDetector() as face:
        try:
            for frame in cam.reader(CV1):
                res = face.process(frame)
                s = eyes.update(res, frame.timestamp)
                fmt = lambda v, f=".3f": "  -  " if v is None else format(v, f)
                line = (f"EAR {fmt(s.ear)} închidere {fmt(s.closure, '.2f')} zâmbet {fmt(s.smile, '.2f')} | "
                        f"închiși {'DA' if s.eyes_closed else 'nu'} {s.closed_sec:4.1f}s | "
                        f"clipiri {s.blink_count} ({fmt(s.blink_rate, '.1f')}/min) | PERCLOS {fmt(s.perclos)}")
                print(line + "      ", end="\r")

                if args.show:
                    img = frame.image.copy()
                    if res is not None:
                        for i in Idx.LEFT_EYE + Idx.RIGHT_EYE:
                            x, y = res.px(i).astype(int)
                            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255) if s.eyes_closed else (0, 255, 0), -1)
                    cv2.imshow("vigil eyes", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
    cv2.destroyAllWindows()
