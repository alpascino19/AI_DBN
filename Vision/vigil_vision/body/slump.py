"""Capul pe volan / prăbușit în față (posibilă pierdere a cunoștinței). [CV2]

Când șoferul cade cu capul pe volan, fața nu se mai vede (CV1 o poate pierde),
dar Pose încă vede umerii. Folosim corpul:

    1. căderea capului: distanța pe verticală umeri -> cap (nas / urechi / ochi),
       în unități de corp (body/geometry.py — merge de față, la 45° și din lateral),
       comparată cu poziția normală a șoferului (învățată în primele calib_sec).
       drop = 0 drept, ~1 capul la nivelul umerilor.
    2. aplecarea în față: umerii coboară, corpul pare mai mare (se apropie de
       cameră), iar la 45°/lateral capul iese în fața umerilor și trunchiul se
       înclină (dacă se văd șoldurile).
    3. fața nu se mai vede (de la CV1 sau punctele capului dispar).
    4. nemișcarea: un om inconștient nu se mișcă; cineva care ridică ceva de jos da.
       Fără nemișcare (still_min_sec) nu raportăm, ca să nu dăm alarme false.

Dacă Pose pierde complet persoana DUPĂ ce era prăbușită, păstrăm starea până la
hold_sec (un om inconștient nu pleacă din scaun).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from ..context import Detection, FrameContext, P
from ..schemas import HEAD_ON_WHEEL
from ..smoothing import Ema
from .geometry import ensure

HEAD_IDX = (P.NOSE, 2, 5, 7, 8)          # nas, ochi stâng/drept, urechi stângă/dreaptă


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class SlumpAnalyzer:
    def __init__(self, cfg: dict) -> None:
        c = cfg["slump"]
        self.calib_sec = c["calib_sec"]
        self.default_gap = c["default_gap"]
        self.drop_lo, self.drop_hi = c["drop_low"], c["drop_high"]
        self.enter, self.exit = c["score_enter"], c["score_exit"]
        self.still_thr = c["still_thr"]
        self.still_min = c["still_min_sec"]
        self.hold_sec = c["hold_sec"]
        self.lean_cfg = c["lean"]
        self.min_vis = cfg["chest"]["min_visibility"]
        self.cfg = cfg

        self._gap = Ema(0.3)
        self._calib: list[tuple[float, float, float, float, float]] = []  # gap, umeri_y, unit, față, trunchi
        self._t0: Optional[float] = None
        self.base_gap: Optional[float] = None
        self.base_sy: Optional[float] = None
        self.base_unit: Optional[float] = None
        self.base_fwd = 0.0
        self.base_torso: Optional[float] = None
        self._motion: deque[tuple[float, np.ndarray]] = deque()  # (t, puncte cheie)
        self._still_since: Optional[float] = None
        self._slumped = False
        self._last_slumped_t = -1e9
        self._last: dict = {}

    # ---------- măsurători ----------
    def _head(self, pose) -> Optional[tuple[float, float]]:
        pts = [(pose[i].x, pose[i].y) for i in HEAD_IDX if pose[i].visibility >= 0.5]
        return tuple(np.mean(pts, axis=0)) if pts else None   # type: ignore[return-value]

    def _stillness(self, ctx: FrameContext, unit: float) -> float:
        """Câte secunde stă nemișcat (umeri, cap, încheieturi)."""
        # punctele nevăzute (ex. încheieturile în poală) sunt NaN, ca tremuratul lor să nu conteze
        pts = np.array([[ctx.pose[i].x, ctx.pose[i].y] if ctx.pose[i].visibility >= 0.5 else [np.nan, np.nan]
                        for i in (P.NOSE, P.SHOULDER_L, P.SHOULDER_R, P.WRIST_L, P.WRIST_R)])
        self._motion.append((ctx.t, pts))
        while self._motion and ctx.t - self._motion[0][0] > 1.0:
            self._motion.popleft()
        if len(self._motion) < 5:
            return 0.0
        arr = np.stack([p for _, p in self._motion])                # (n, 5, 2)
        valid = np.isfinite(arr[..., 0]).mean(axis=0) >= 0.5         # puncte văzute în >= 50% din cadre
        with np.errstate(all="ignore"):
            per_point = np.nanstd(arr[:, valid], axis=0).max(axis=1)
        movement = float(np.max(per_point)) / unit if per_point.size else 1.0
        if movement < self.still_thr:
            if self._still_since is None:
                self._still_since = self._motion[0][0]
            return ctx.t - self._still_since
        self._still_since = None
        return 0.0

    # ---------- update ----------
    def update(self, ctx: FrameContext) -> None:
        pose = ctx.pose
        g = ensure(ctx, self.cfg)
        if g is None:
            self._motion.clear()
            self._still_since = None
            held = self._slumped and ctx.t - self._last_slumped_t <= self.hold_sec
            res = {"state": "slumped" if held else "unknown", "pose_lost": True}
            if held:
                res["score"] = self._last.get("score", 0.6)
            else:
                self._slumped = False
            self._emit(ctx, res, held, still_sec=self._last.get("still_sec", 0.0))
            return

        unit, sy = g.unit, g.sy
        head = self._head(pose)
        head_missing = head is None
        gap = self._gap((sy - head[1]) / unit, ctx.t) if not head_missing else None
        fwd = g.forward(head[0]) if not head_missing else None          # capul în fața umerilor
        torso = g.torso_angle

        # calibrare: poziția normală (dreaptă) a șoferului
        if self.base_gap is None:
            if self._t0 is None:
                self._t0 = ctx.t
            if gap is not None:
                self._calib.append((gap, sy, unit, fwd or 0.0, np.nan if torso is None else torso))
            if ctx.t - self._t0 >= self.calib_sec and len(self._calib) >= 10:
                arr = np.array(self._calib)
                self.base_gap = max(float(np.median(arr[:, 0])), 0.2)
                self.base_sy, self.base_unit = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
                self.base_fwd = float(np.median(arr[:, 3]))
                tor = arr[:, 4][np.isfinite(arr[:, 4])]
                self.base_torso = float(np.median(tor)) if tor.size else None
        base_gap = self.base_gap or self.default_gap

        drop = 1.0 if head_missing else 1.0 - gap / base_gap
        lean = 0.0
        if self.base_sy is not None:
            L = self.lean_cfg                                                  # config.yaml: slump.lean
            parts = [_ramp((sy - self.base_sy) / self.base_unit, *L["shoulder_drop"]),  # umerii coboară
                     _ramp(unit / self.base_unit, *L["closer"])]                        # se apropie de cameră
            if fwd is not None and g.facing:
                parts.append(_ramp(fwd - self.base_fwd, *L["head_forward"]))            # capul iese în față
            if torso is not None and self.base_torso is not None:
                parts.append(_ramp(torso - self.base_torso, *L["torso_deg"]))           # trunchiul se înclină
            lean = max(parts)
        face_detected = ctx.face.detected if ctx.face is not None else None
        s_face = 1.0 if head_missing or face_detected is False else (0.0 if face_detected else 0.5)

        score = 0.6 * _ramp(drop, self.drop_lo, self.drop_hi) + 0.2 * s_face + 0.2 * lean
        if drop < self.drop_lo:
            score = min(score, self.exit - 0.05)        # fără căderea capului nu e "pe volan"
        still_sec = self._stillness(ctx, unit)

        posture = score >= (self.exit if self._slumped else self.enter)
        active = posture and still_sec >= self.still_min
        if self._slumped and posture:                   # odată prăbușit, mici mișcări nu-l scot
            active = True
        self._slumped = active
        if active:
            self._last_slumped_t = ctx.t

        res = {
            "state": "slumped" if active else ("leaning" if posture else "upright"),
            "score": round(score, 2),
            "drop": round(max(drop, 0.0), 2),
            "lean": round(lean, 2),
            "still_sec": round(still_sec, 1),
            "face_visible": None if face_detected is None else bool(face_detected and not head_missing),
            "calibrated": self.base_gap is not None,
        }
        self._emit(ctx, res, active, still_sec)

    def _emit(self, ctx: FrameContext, res: dict, active: bool, still_sec: float) -> None:
        self._last = {**res, "still_sec": res.get("still_sec", still_sec)}
        ctx.signals["slump"] = self._last
        conf = res.get("score", 0.0) * (0.7 + 0.3 * _ramp(still_sec, self.still_min, 5.0)) if active else 0.0
        attrs = {"still_sec": self._last["still_sec"]}
        attrs["pose_lost"] = bool(res.get("pose_lost", False))
        if res.get("face_visible") is not None:
            attrs["face_visible"] = res["face_visible"]
        ctx.add(Detection(HEAD_ON_WHEEL, active, min(conf, 1.0), attrs))
