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

import math
import warnings
from collections import deque
from typing import Optional

import numpy as np

from ..context import Detection, FrameContext, P
from ..schemas import HEAD_ON_WHEEL
from ..smoothing import Ema
from .geometry import ensure

HEAD_IDX = (P.NOSE, 2, 5, 7, 8)          # nas, ochi stâng/drept, urechi stângă/dreaptă
SLUMPED_SIDE = "slumped_side"            # căzut într-o parte și nemișcat (urgență)
BAD_POSTURE = "bad_posture"              # aplecat / cocoșat (alerta vine după bad_posture_sec)
OUT_OF_POSITION = "out_of_position"      # nu se vede corpul sau e mult mai departe decât normal


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
        # pragul pentru starea "leaning" (afișare / linia portocalie), separat de alarmă
        self.leaning_deg = c["leaning_deg"]
        self.leaning_drop = c["leaning_drop"]
        self.min_vis = cfg["chest"]["min_visibility"]
        # căzut într-o parte (grade față de poziția calibrată) + ieșit din poziție
        self.side_enter = c.get("side_deg", 25.0)
        self.side_exit = c.get("side_exit_deg", 18.0)
        self.far_ratio = c.get("far_ratio", 0.65)
        self.cfg = cfg

        self._gap = Ema(0.3)
        self._calib: list[tuple[float, ...]] = []  # gap, umeri_y, unit, față, trunchi, unghi
        self._t0: Optional[float] = None
        self.base_gap: Optional[float] = None
        self.base_sy: Optional[float] = None
        self.base_unit: Optional[float] = None
        self.base_fwd = 0.0
        self.base_angle = 0.0
        self.base_torso: Optional[float] = None
        self.base_sh_dy = 0.0                   # diferența de înălțime a umerilor în poziția normală
        self.base_lat = 0.0                     # înclinarea laterală cap -> umeri (doar camera din față)
        self._side = False
        self._last_side_t = -1e9
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
            held_side = self._side and ctx.t - self._last_side_t <= self.hold_sec
            res = {"state": "slumped" if held or held_side else "unknown", "pose_lost": True}
            if held:
                res["score"] = self._last.get("score", 0.6)
            else:
                self._slumped = False
            if held_side:
                res["direction"] = "side"
            else:
                self._side = False
            # nu se vede corpul și nu era prăbușit înainte -> a ieșit din poziție (sau scaun gol)
            out = None if held or held_side else "lost"
            self._emit(ctx, res, held, still_sec=self._last.get("still_sec", 0.0), side=held_side, out=out)
            return

        unit, sy = g.unit, g.sy
        head = self._head(pose)
        head_missing = head is None
        gap = self._gap((sy - head[1]) / unit, ctx.t) if not head_missing else None
        fwd = g.forward(head[0]) if not head_missing else None          # capul în fața umerilor
        # unghiul liniei cap -> umeri față de verticală (0° = capul drept deasupra umerilor);
        # din lateral/45° se măsoară spre față, de față spre orice parte
        angle = None
        if not head_missing:
            dx = (head[0] - g.sx) * g.facing if g.facing else abs(head[0] - g.sx)
            angle = math.degrees(math.atan2(dx, max(sy - head[1], 1e-6)))
        torso = g.torso_angle
        # lateral: umerii la înălțimi diferite (merge din orice unghi) și, cu camera din față,
        # linia cap -> umeri înclinată spre stânga/dreapta
        ls, rs = pose[P.SHOULDER_L], pose[P.SHOULDER_R]
        sh_dy = (ls.y - rs.y) / unit if min(ls.visibility, rs.visibility) >= self.min_vis * 0.6 else None
        lat = None
        if not head_missing and g.view == "front":
            lat = math.degrees(math.atan2(head[0] - g.sx, max(sy - head[1], 1e-6)))

        # calibrare: poziția normală (dreaptă) a șoferului
        if self.base_gap is None:
            if self._t0 is None:
                self._t0 = ctx.t
            if gap is not None:
                self._calib.append((gap, sy, unit, fwd or 0.0, np.nan if torso is None else torso,
                                    angle or 0.0, np.nan if sh_dy is None else sh_dy,
                                    np.nan if lat is None else lat))
            if ctx.t - self._t0 >= self.calib_sec and len(self._calib) >= 10:
                arr = np.array(self._calib)
                with warnings.catch_warnings():             # coloană numai NaN (ex. un umăr ascuns)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(arr[:, 6:8], axis=0)
                self.base_sh_dy = float(med[0]) if np.isfinite(med[0]) else 0.0
                self.base_lat = float(med[1]) if np.isfinite(med[1]) else 0.0
                self.base_gap = max(float(np.median(arr[:, 0])), 0.2)
                self.base_sy, self.base_unit = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
                self.base_fwd = float(np.median(arr[:, 3]))
                self.base_angle = float(np.median(arr[:, 5]))
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
        # aplecat (doar stare/afișare): unghiul a crescut sau capul a coborât puțin
        leaning = head_missing or drop >= self.leaning_drop or \
            (angle is not None and angle - self.base_angle >= self.leaning_deg)
        active = posture and still_sec >= self.still_min
        if self._slumped and posture:                   # odată prăbușit, mici mișcări nu-l scot
            active = True
        self._slumped = active
        if active:
            self._last_slumped_t = ctx.t

        # căzut într-o parte: înclinare laterală mare + nemișcat (cineva care ia ceva de pe scaun se mișcă)
        side_parts = []
        if sh_dy is not None:
            side_parts.append(math.degrees(math.atan(abs(sh_dy - self.base_sh_dy))))
        if lat is not None:
            side_parts.append(abs(lat - self.base_lat))
        side_deg = max(side_parts) if side_parts else 0.0
        side_on = self.base_gap is not None and side_deg >= (self.side_exit if self._side else self.side_enter)
        side = side_on and (still_sec >= self.still_min or self._side)
        self._side = side
        if side:
            self._last_side_t = ctx.t

        # prea departe de poziția normală (s-a dat înapoi / s-a ridicat de pe scaun)
        out = None
        if self.base_unit and unit / self.base_unit < self.far_ratio:
            out = "far"

        state = "slumped" if active or side else ("leaning" if posture or leaning else "upright")
        res = {
            "state": state,
            "direction": "side" if side else ("forward" if active else None),
            "side_deg": round(side_deg, 1),
            "score": round(score, 2),
            "drop": round(max(drop, 0.0), 2),
            "lean": round(lean, 2),
            "angle": None if angle is None else round(angle - self.base_angle, 1),   # față de normal
            "still_sec": round(still_sec, 1),
            "face_visible": None if face_detected is None else bool(face_detected and not head_missing),
            "calibrated": self.base_gap is not None,
        }
        self._emit(ctx, res, active, still_sec, side=side, out=out,
                   bad_posture=state == "leaning")

    def _emit(self, ctx: FrameContext, res: dict, active: bool, still_sec: float, side: bool = False,
              out: Optional[str] = None, bad_posture: bool = False) -> None:
        ctx.add(Detection(SLUMPED_SIDE, side, min(1.0, 0.6 + 0.1 * still_sec) if side else 0.0,
                          {"side_deg": res.get("side_deg"), "still_sec": res.get("still_sec", still_sec)}))
        ctx.add(Detection(BAD_POSTURE, bad_posture, 0.6 if bad_posture else 0.0,
                          {"lean": res.get("lean"), "angle": res.get("angle")}))
        ctx.add(Detection(OUT_OF_POSITION, out is not None, 0.7 if out else 0.0, {"reason": out}))
        if out:
            res = {**res, "out": out}
        self._last = {**res, "still_sec": res.get("still_sec", still_sec)}
        ctx.signals["slump"] = self._last
        conf = res.get("score", 0.0) * (0.7 + 0.3 * _ramp(still_sec, self.still_min, 5.0)) if active else 0.0
        attrs = {"still_sec": self._last["still_sec"]}
        attrs["pose_lost"] = bool(res.get("pose_lost", False))
        if res.get("face_visible") is not None:
            attrs["face_visible"] = res["face_visible"]
        ctx.add(Detection(HEAD_ON_WHEEL, active, min(conf, 1.0), attrs))
