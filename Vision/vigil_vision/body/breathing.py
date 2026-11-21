"""Respirație grea: mișcarea pieptului (optical flow) + scor. [CV2]

Două surse de semnal, alese automat:
  A. FLOW (principală): optical flow (Farneback) în zona pieptului, între cadre
     consecutive. Mediana mișcării (x și y) pe pixelii cu textură, MINUS mișcarea
     capului (scade mișcarea întregului corp / vibrațiile), integrată în timp.
     Semnalul e proiecția pe direcția principală a mișcării (PCA): de față
     respirația urcă/coboară umerii, din lateral/45° pieptul iese în față —
     PCA găsește singur direcția, oricare ar fi unghiul camerei.
  B. LANDMARKS (rezervă, când tricoul nu are textură): distanța umeri–nas din
     Pose, în unități de corp (body/geometry.py); dacă tot corpul se mișcă, umerii
     și nasul se mișcă împreună și se anulează.

Pe o fereastră de ~10 s, de 5 ori pe secundă: reeșantionare 10 Hz, eliminarea
trendului, FFT în banda 0.13–1.0 Hz (8–60 respirații/min) -> ritm, amplitudine,
periodicitate. Fiecare sursă își învață normalul în primele calib_sec.

Decizia (C) e un SCOR 0..1, nu reguli "și/și/și":
    amplitudine față de normal (ultimele 4 s)  0.50
    ritm                                       0.20
    periodicitate (respirația e ritmică)       0.15
    gura deschisă (CV1: jaw_open), dacă există 0.15
Heavy = scor >= score_enter câteva evaluări la rând; iese sub score_exit.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np

from ..context import Detection, FrameContext, P
from ..schemas import HAND_ON_CHEST, HEAVY_BREATHING
from ..smoothing import Ema
from .geometry import ensure

FS = 10.0                  # Hz, după reeșantionare
BAND = (0.13, 1.0)         # Hz
EVAL_EVERY = 0.2           # s
RECENT_SEC = 4.0           # s; amplitudinea recentă -> intrare/ieșire rapidă


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class _Signal:
    """Buffer + calibrare + analiză spectrală pentru o sursă."""

    def __init__(self, name: str, window: float, calib_sec: float, floor: float) -> None:
        self.name = name
        self.window, self.calib_sec, self.floor = window, calib_sec, floor
        self.buf: deque[tuple[float, float, float, float]] = deque()  # (t, x, y, scară)
        self.calib: list[float] = []
        self.calib_t0: Optional[float] = None
        self.baseline: Optional[float] = None
        self.trace: list[float] = []
        self.axis = (0.0, 1.0)              # direcția mișcării (x, y)

    def add(self, t: float, vx: float, vy: float, scale: float) -> None:
        self.buf.append((t, vx, vy, scale))

    def trim(self, t: float) -> None:
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()
        if self.buf and t - self.buf[-1][0] > 1.5:               # sursa a lipsit
            self.buf.clear()

    def fresh(self, t: float) -> bool:
        return bool(self.buf) and t - self.buf[-1][0] < 0.5

    def coverage(self) -> float:
        return self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0

    def analyze(self, t: float, min_window: float) -> Optional[dict]:
        if len(self.buf) < 20 or self.coverage() < min_window:
            return None
        arr = np.array(self.buf)
        ts = arr[:, 0]
        scale = float(np.median(arr[:, 3]))
        tu = np.arange(ts[0], ts[-1], 1.0 / FS)
        detr = lambda a: a - np.polyval(np.polyfit(tu, a, 1), tu)
        xu, yu = detr(np.interp(tu, ts, arr[:, 1])), detr(np.interp(tu, ts, arr[:, 2]))
        freqs = np.fft.rfftfreq(len(yu), 1.0 / FS)
        m = (freqs >= BAND[0]) & (freqs <= BAND[1])
        if m.sum() < 2:
            return None
        if np.any(xu):
            # 2D: proiecție pe direcția principală a mișcării în banda respirației
            bp = lambda a: np.fft.irfft(np.where(m, np.fft.rfft(a), 0), n=len(a))
            _, vecs = np.linalg.eigh(np.cov(np.vstack([bp(xu), bp(yu)])))
            ax = vecs[:, -1]
            yu = ax[0] * xu + ax[1] * yu
            self.axis = (round(float(ax[0]), 2), round(float(ax[1]), 2))
        power = np.abs(np.fft.rfft(yu * np.hanning(len(yu)))) ** 2
        pk = int(np.argmax(np.where(m, power, 0)))
        pm = np.where(m, power, 0.0)                               # doar banda respirației
        period = float(pm[max(pk - 1, 0):pk + 2].sum() / (pm.sum() + 1e-12))
        filt = np.fft.irfft(np.where(m, np.fft.rfft(yu), 0), n=len(yu)) / scale
        n_recent = int(RECENT_SEC * FS)
        amp = float(np.std(filt))
        amp_recent = float(np.std(filt[-n_recent:])) if len(filt) >= n_recent else amp
        self.trace = filt.round(4).tolist()

        if self.baseline is None:                                  # calibrare
            if self.calib_t0 is None:
                self.calib_t0 = ts[0]
            self.calib.append(amp)
            if t - self.calib_t0 >= self.calib_sec:
                # percentila 30: dacă șoferul respiră mai tare o clipă în calibrare, nu strică normalul
                self.baseline = max(float(np.percentile(self.calib, 30)), self.floor)
        return {"rate_bpm": freqs[pk] * 60.0, "amp": amp, "amp_recent": amp_recent,
                "periodicity": period, "calibrated": self.baseline is not None}


class BreathingAnalyzer:
    def __init__(self, cfg: dict) -> None:
        b = cfg["breathing"]
        self.cfg = cfg
        self.source_mode = b.get("source", "auto")
        self.min_window = b["min_window_sec"]
        self.jaw_thr = b["jaw_open_thr"]
        self.r_lo, self.r_hi = b["amp_ratio_low"], b["amp_ratio_high"]
        self.rate_lo, self.rate_hi = b["rate_low_bpm"], b["rate_high_bpm"]
        self.enter, self.exit = b["score_enter"], b["score_exit"]
        self.enter_n, self.exit_n = b["enter_evals"], b["exit_evals"]
        self.flow_width = b["flow_width"]

        self.min_texture = b["min_texture_frac"]
        self.min_vis = max(cfg["chest"]["min_visibility"], 0.7)

        self.flow = _Signal("flow", b["window_sec"], b["calib_sec"], b["baseline_floor"])
        self.lmk = _Signal("landmarks", b["window_sec"], b["calib_sec"], b["baseline_floor"])
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_t = -1e9
        self._cum = (0.0, 0.0)                # deplasarea integrată a pieptului (flow, x/y)
        self._score = Ema(0.4)
        self._heavy = False
        self._streak = 0
        self._last_eval = -1e9
        self._last: dict = {"state": "unknown"}

    # ---------- A: optical flow ----------
    def _motion(self, a: np.ndarray, b: np.ndarray) -> Optional[tuple[float, float]]:
        """Mediana mișcării (dx, dy) în px pe pixelii cu textură, sau None."""
        if a.shape[0] < 12 or a.shape[1] < 12:
            return None
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 3, 5, 1.1, 0)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        mask = np.hypot(gx, gy) > 20
        if mask.mean() < self.min_texture:
            return None
        return float(np.median(flow[..., 0][mask])), float(np.median(flow[..., 1][mask]))

    def _flow_sample(self, ctx: FrameContext) -> None:
        img = ctx.frame.image
        h, w = img.shape[:2]
        W = self.flow_width
        H = max(1, int(round(h * W / w)))
        gray = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        prev, prev_t = self._prev_gray, self._prev_t
        self._prev_gray, self._prev_t = gray, ctx.t
        zone = ctx.signals.get("chest_zone")
        g = ensure(ctx, self.cfg)
        if prev is None or zone is None or g is None or ctx.t - prev_t > 0.3:
            return

        def crop(box):
            x1, y1, x2, y2 = (int(np.clip(v, 0, 1) * s) for v, s in zip(box, (W, H, W, H)))
            return (slice(y1, y2), slice(x1, x2)), max(x2 - x1, 1)

        sl, roi_w = crop(zone)
        chest = self._motion(prev[sl], gray[sl])
        if chest is None:
            return
        nose, u = ctx.pose[P.NOSE], g.unit
        head = None
        if nose.visibility >= 0.5:
            hsl, _ = crop((nose.x - 0.3 * u, nose.y - 0.3 * u, nose.x + 0.3 * u, nose.y + 0.2 * u))
            head = self._motion(prev[hsl], gray[hsl])
        hx, hy = head or (0.0, 0.0)
        ix, iy = (chest[0] - hx) / roi_w, (chest[1] - hy) / roi_w
        if np.hypot(ix, iy) > 0.05:          # mâna trece prin zonă / salt -> ignorăm
            return
        self._cum = (self._cum[0] + ix, self._cum[1] + iy)
        self.flow.add(ctx.t, self._cum[0], self._cum[1], 1.0)

    # ---------- B: landmarks ----------
    def _landmark_sample(self, ctx: FrameContext) -> None:
        g = ensure(ctx, self.cfg)
        if g is None:
            return
        pose = ctx.pose
        ls, rs, nose = pose[P.SHOULDER_L], pose[P.SHOULDER_R], pose[P.NOSE]
        if nose.visibility < self.min_vis or pose[g.near].visibility < self.min_vis:
            return
        u = g.unit
        for hnd in ctx.hands:                 # mână pe umăr -> punctul umărului se mută
            hx, hy = hnd.palm_center()
            if min(abs(hx - ls.x) + abs(hy - ls.y), abs(hx - rs.x) + abs(hy - rs.y)) < 0.2 * u:
                return
        v = g.sy - nose.y
        if self.lmk.buf and abs(v - self.lmk.buf[-1][2]) > 0.08 * u and ctx.t - self.lmk.buf[-1][0] < 0.5:
            return
        self.lmk.add(ctx.t, 0.0, v, u)

    # ---------- update ----------
    def update(self, ctx: FrameContext) -> None:
        if self.source_mode in ("auto", "flow"):
            self._flow_sample(ctx)
        if self.source_mode in ("auto", "landmarks"):
            self._landmark_sample(ctx)
        self.flow.trim(ctx.t)
        self.lmk.trim(ctx.t)

        if ctx.t - self._last_eval >= EVAL_EVERY:
            self._last_eval = ctx.t
            self._last = self._evaluate(ctx)

        res = self._last
        ctx.signals["breathing"] = res
        sig = self.flow if res.get("source") == "flow" else self.lmk
        ctx.signals["breathing_trace"] = sig.trace
        heavy = res.get("state") == "heavy"
        ctx.add(Detection(HEAVY_BREATHING, heavy, res.get("score", 0.0) if heavy else 0.0,
                          {k: res[k] for k in ("rate_bpm", "mouth_open", "source") if k in res}))
        chest = ctx.detections.get(HAND_ON_CHEST)            # ca în contract: "breathing":"heavy"
        if chest is not None and chest.active and res.get("state") in ("heavy", "normal"):
            chest.attrs["breathing"] = res["state"]

    def _rule_score(self, feat: dict) -> float:
        """Scorul 0..1 din reguli (ponderi fixe)."""
        s_amp = _ramp(feat["ratio"], self.r_lo, self.r_hi)
        parts = [(0.50, s_amp), (0.20, _ramp(feat["rate_bpm"], self.rate_lo, self.rate_hi)),
                 (0.15, _ramp(feat["periodicity"], 0.2, 0.6))]
        if feat.get("jaw_open") is not None:
            parts.append((0.15, _ramp(feat["jaw_open"], 0.15, 0.5)))
        raw = sum(w * s for w, s in parts) / sum(w for w, _ in parts)
        if s_amp == 0.0:                        # fără amplitudine mare nu e respirație grea
            raw = min(raw, self.exit - 0.05)
        return raw

    def _evaluate(self, ctx: FrameContext) -> dict:
        f_flow = self.flow.analyze(ctx.t, self.min_window) if self.flow.fresh(ctx.t) else None
        f_lmk = self.lmk.analyze(ctx.t, self.min_window) if self.lmk.fresh(ctx.t) else None
        # alegerea sursei: flow dacă e calibrată și are date, altfel landmarks
        if f_flow and f_flow["calibrated"]:
            src, f = self.flow, f_flow
        elif f_lmk and f_lmk["calibrated"]:
            src, f = self.lmk, f_lmk
        elif f_flow or f_lmk:
            src, f = (self.flow, f_flow) if f_flow else (self.lmk, f_lmk)
            return {"state": "calibrating", "source": src.name, "rate_bpm": round(f["rate_bpm"], 1),
                    "calibrated": False}
        else:
            self._heavy, self._streak = False, 0
            self._score.reset()
            return {"state": "unknown"}

        ratio = f["amp_recent"] / src.baseline
        jaw = ctx.signal("jaw_open")
        feat = {"ratio": ratio, "rate_bpm": f["rate_bpm"], "periodicity": f["periodicity"],
                "jaw_open": jaw}
        score = self._score(self._rule_score(feat), ctx.t)

        want = score >= self.enter if not self._heavy else score >= self.exit
        if want == self._heavy:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= (self.enter_n if want else self.exit_n):
                self._heavy, self._streak = want, 0

        return {
            "state": "heavy" if self._heavy else "normal",
            "score": round(score, 2),
            "source": src.name,
            "rate_bpm": round(f["rate_bpm"], 1),
            "amp_ratio": round(ratio, 2),
            "periodicity": round(f["periodicity"], 2),
            "mouth_open": None if jaw is None else bool(jaw >= self.jaw_thr),
            "calibrated": True,
        }
