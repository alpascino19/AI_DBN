"""Mâna pe piept (încheietura în zona toracelui). [CV2]

Zona pieptului se construiește din geometria corpului (body/geometry.py), deci
merge și cu camera în față, și la ~45°, și din lateral:
    x: între umeri; în 3/4 sau lateral se întinde spre direcția în care privește
       corpul (acolo e partea din față a pieptului): view_forward[view] × unit,
       iar spre spate doar view_back[view] × unit
    y: de la linia umerilor până la ~60% din trunchi (sau zone_bottom × unit
       dacă șoldurile nu se văd — în mașină, de obicei nu se văd)
Punctele mâinii: centrul palmei din Hands (mai precis) și încheietura/arătătorul
din Pose (rezervă). O mână care ține telefonul nu contează (e telefon, nu piept).
Durata ("câteva secunde") o calculează events/tracker.py.
Zona e netezită în timp, ca să nu "respire" odată cu zgomotul MediaPipe.
"""
from __future__ import annotations

from typing import Optional

from ..context import Detection, FrameContext, P
from ..schemas import HAND_ON_CHEST
from ..smoothing import Ema
from .geometry import ensure

Zone = tuple[float, float, float, float]   # x1, y1, x2, y2


class ChestAnalyzer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        c = cfg["chest"]
        self.top = c["zone_top"]
        self.bottom = c["zone_bottom"]
        self.min_vis = c["min_visibility"]
        view = cfg["view"]["mode"]
        self.camera_side = cfg["view"]["camera_side"]
        self.fwd = c["view_forward"][view]
        self.back = c["view_back"][view]
        # lățimea zonei se schimbă lent (1 s), poziția urmărește persoana mai repede (0.25 s)
        self._unit = Ema(c.get("width_tau", 1.0))
        self._half = Ema(c.get("width_tau", 1.0))
        self._cx = Ema(c.get("pos_tau", 0.25))
        self._sy = Ema(c.get("pos_tau", 0.25))
        self._torso = Ema(c.get("width_tau", 1.0))
        self._corners = [Ema(c.get("corner_tau", 0.2)) for _ in range(4)]
        self._use_hips = False
        self._last_t = -1e9

    def _reset(self) -> None:
        for f in (self._unit, self._half, self._cx, self._sy, self._torso, *self._corners):
            f.reset()
        self._use_hips = False

    def zone(self, ctx: FrameContext) -> Optional[Zone]:
        g = ensure(ctx, self.cfg)
        if g is None:
            return None
        pose = ctx.pose
        t = ctx.t
        if t - self._last_t > 0.5:             # persoana a lipsit -> pornim de la zero
            self._reset()
        self._last_t = t

        ls, rs = pose[P.SHOULDER_L], pose[P.SHOULDER_R]
        unit = self._unit(g.unit, t)
        half = self._half(abs(ls.x - rs.x) / 2 if g.shoulders_ok else 0.0, t)   # jumătate din umeri pe x
        cx = self._cx(g.sx, t)
        sy = self._sy(g.sy, t)

        # spre față / spre spate (de față: simetric)
        if g.facing == 0:
            left = right = half + self.back * unit
        else:
            ahead, behind = half + self.fwd * unit, half + self.back * unit
            right, left = (ahead, behind) if g.facing > 0 else (behind, ahead)

        # șoldurile: histerezis, ca zona să nu sară între cele două formule
        hl, hr = pose[P.HIP_L], pose[P.HIP_R]
        hip_vis = min(hl.visibility, hr.visibility)
        if self._use_hips and hip_vis < 0.6:
            self._use_hips = False
        elif not self._use_hips and hip_vis > 0.8:
            self._use_hips = True
        hip_y = (hl.y + hr.y) / 2
        if self._use_hips and hip_y > sy:
            depth = self._torso(0.6 * (hip_y - sy), t)
        else:
            depth = self._torso(self.bottom * unit, t)

        raw = (cx - left, sy + self.top * unit, cx + right, sy + depth)
        return tuple(f(v, t) for f, v in zip(self._corners, raw))   # type: ignore[return-value]

    def _hand_points(self, ctx: FrameContext) -> list[tuple[float, float, str]]:
        pts = [(*h.palm_center(), "hands") for h in ctx.hands]
        if ctx.pose:
            for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
                a, b = ctx.pose[w], ctx.pose[i]
                if min(a.visibility, b.visibility) >= self.min_vis:
                    pts.append(((a.x + b.x) / 2, (a.y + b.y) / 2, "pose"))
        return pts

    def update(self, ctx: FrameContext) -> None:
        z = self.zone(ctx)
        ctx.signals["chest_zone"] = z
        if z is None:
            ctx.add(Detection(HAND_ON_CHEST, False))
            return
        x1, y1, x2, y2 = z
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        phones = [o for o in ctx.objects if o.label == "cell phone"]
        g = ctx.geom
        ls, rs = ctx.pose[P.SHOULDER_L], ctx.pose[P.SHOULDER_R]

        best, side, n_in = 0.0, None, 0
        for x, y, src in self._hand_points(ctx):
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if any(p.contains(x, y, pad=0.1) for p in phones):
                continue
            n_in += 1
            # cât de central e punctul în zonă (1 = centru, 0 = margine)
            dx = abs(x - cx) / ((x2 - x1) / 2)
            dy = abs(y - cy) / ((y2 - y1) / 2)
            central = max(0.0, 1.0 - max(dx, dy))
            conf = (0.65 if src == "hands" else 0.55) + 0.35 * central
            if conf > best:
                best = conf
                idx = P.SHOULDER_L if abs(x - ls.x) < abs(x - rs.x) else P.SHOULDER_R
                if g is not None and g.view != "front" and not g.shoulders_ok:
                    idx = g.near
                side = g.driver_side(idx, self.camera_side) if g is not None else None

        attrs = {}
        if n_in:
            attrs["side"] = side                           # partea șoferului (stânga = inimă)
            attrs["both_hands"] = n_in >= 3 or (n_in >= 2 and len(ctx.hands) >= 2)
        ctx.add(Detection(HAND_ON_CHEST, n_in > 0, min(best, 1.0), attrs))
