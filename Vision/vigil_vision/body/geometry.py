"""Geometria corpului, independentă de unghiul camerei. [CV2]

Camera nu stă neapărat în fața șoferului (în mașină: ~45°, în dreapta lui). Din
lateral, umerii se suprapun în imagine, deci "lățimea umerilor pe x" nu mai e o
unitate bună. Aici calculăm o dată pe cadru, pentru toți analizorii:

    unit        mărimea corpului în imagine: distanța 2D (nu doar pe x) între umeri;
                dacă umerii nu se văd bine, distanța nas–umeri × raportul învățat
                în calibrare. Netezită, deci stabilă.
    facing      încotro privește corpul în imagine pe x: +1 dreapta, -1 stânga,
                0 = de față. Din poziția nasului față de umeri (cu histerezis);
                până se vede clar, din config (camera_side + mirror).
    near/far    umărul apropiat / depărtat de cameră (din z-ul MediaPipe, cu
                histerezis); umărul depărtat e parțial ascuns, deci mai puțin sigur.
    torso_angle înclinarea trunchiului față de verticală (grade), dacă se văd șoldurile.

    geom = ensure(ctx, cfg)    # folosește ctx.geom dacă pipeline-ul l-a calculat,
                               # altfel îl calculează pe loc (fără netezire, ex. în teste)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..context import FrameContext, P
from ..smoothing import Ema

VIEWS = ("front", "three_quarter", "side")


@dataclass
class Geom:
    unit: float                       # mărimea corpului (coordonate normalizate)
    sx: float                         # mijlocul umerilor
    sy: float
    facing: int                       # +1 / -1 / 0
    near: int                         # indice Pose al umărului apropiat de cameră
    far: int
    view: str
    hip_y: Optional[float] = None
    torso_angle: Optional[float] = None
    shoulders_ok: bool = True
    mirror: bool = False              # imaginea e în oglindă -> etichetele L/R din Pose sunt inversate

    def forward(self, x: float) -> float:
        """Cât de în față (spre direcția corpului) e un punct x, în unități de corp."""
        return (x - self.sx) * self.facing / self.unit if self.facing else 0.0

    def driver_side(self, idx: int, camera_side: str) -> str:
        """'left' / 'right' din perspectiva șoferului pentru umărul idx
        (umărul apropiat e cel dinspre cameră)."""
        if self.view == "front":
            label_left = idx == P.SHOULDER_L
            return "left" if label_left != self.mirror else "right"
        near_side = camera_side
        far_side = "left" if camera_side == "right" else "right"
        return near_side if idx == self.near else far_side


def _default_facing(cfg: dict) -> int:
    """Camera în dreapta-față a șoferului -> în imaginea normală el privește spre dreapta."""
    v = cfg["view"]
    if v["mode"] == "front":
        return 0
    f = 1 if v["camera_side"] == "right" else -1
    return -f if cfg["camera"].get("mirror", False) else f


def _measure(ctx: FrameContext, cfg: dict) -> Optional[dict]:
    pose = ctx.pose
    if not pose:
        return None
    ls, rs, nose = pose[P.SHOULDER_L], pose[P.SHOULDER_R], pose[P.NOSE]
    min_vis = cfg["chest"]["min_visibility"]
    # în 3/4 sau lateral, umărul depărtat poate fi mai puțin vizibil: cerem doar unul sigur
    best = max(ls.visibility, rs.visibility)
    if best < min_vis:
        return None
    both = min(ls.visibility, rs.visibility) >= min_vis * 0.6
    sx, sy = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2) if both else \
        ((ls.x, ls.y) if ls.visibility >= rs.visibility else (rs.x, rs.y))
    sh_dist = math.hypot(ls.x - rs.x, ls.y - rs.y) if both else 0.0
    head_dist = math.hypot(nose.x - sx, nose.y - sy) if nose.visibility >= 0.5 else 0.0
    hl, hr = pose[P.HIP_L], pose[P.HIP_R]
    hip_y = torso = None
    if min(hl.visibility, hr.visibility) >= 0.7:
        hx, hip_y = (hl.x + hr.x) / 2, (hl.y + hr.y) / 2
        if hip_y > sy:
            torso = math.degrees(math.atan2(abs(sx - hx), hip_y - sy))
    return {"sx": sx, "sy": sy, "sh_dist": sh_dist, "head_dist": head_dist, "both": both,
            "nose_x": nose.x if nose.visibility >= 0.5 else None,
            "z": (ls.z, rs.z), "hip_y": hip_y, "torso": torso}


class BodyGeometry:
    """Rulează primul în pipeline; pune ctx.geom."""

    def __init__(self, cfg: dict) -> None:
        v = cfg["view"]
        self.cfg = cfg
        self.view = v["mode"]
        self.head_ratio = v["head_ratio"]            # unit ≈ head_dist × raport (învățat)
        self.calib_sec = v["calib_sec"]
        self._unit = Ema(v.get("unit_tau", 1.0))
        self._ratio = Ema(5.0)
        self._t0: Optional[float] = None
        self.facing = _default_facing(cfg)
        self.near = P.SHOULDER_R if v["camera_side"] == "right" else P.SHOULDER_L
        if cfg["camera"].get("mirror", False) and self.view != "front":
            self.near = P.SHOULDER_L if self.near == P.SHOULDER_R else P.SHOULDER_R
        self._last_t = -1e9

    def update(self, ctx: FrameContext) -> None:
        m = _measure(ctx, self.cfg)
        if m is None:
            ctx.geom = None
            return
        t = ctx.t
        if t - self._last_t > 1.0:
            self._unit.reset()
        self._last_t = t
        if self._t0 is None:
            self._t0 = t

        # raportul umeri/cap se învață doar în calibrare (șoferul stă drept)
        ratio = self._ratio.value or self.head_ratio
        if m["both"] and m["head_dist"] > 0 and t - self._t0 < self.calib_sec:
            ratio = self._ratio(m["sh_dist"] / m["head_dist"], t)
        candidates = [m["sh_dist"]] if m["both"] else []
        if m["head_dist"] > 0:
            candidates.append(m["head_dist"] * ratio * (0.8 if m["both"] else 1.0))
        raw = max(candidates) if candidates else 0.0
        if raw < 0.03:
            ctx.geom = None
            return
        unit = self._unit(raw, t)

        if self.view != "front":
            if m["nose_x"] is not None:              # direcția: nasul e în fața umerilor
                d = (m["nose_x"] - m["sx"]) / unit
                if d > 0.15:
                    self.facing = 1
                elif d < -0.15:
                    self.facing = -1
            zl, zr = m["z"]                          # z mai mic = mai aproape de cameră
            if zl < zr - 0.05:
                self.near = P.SHOULDER_L
            elif zr < zl - 0.05:
                self.near = P.SHOULDER_R
        far = P.SHOULDER_L if self.near == P.SHOULDER_R else P.SHOULDER_R
        ctx.geom = Geom(unit, m["sx"], m["sy"], self.facing if self.view != "front" else 0,
                        self.near, far, self.view, m["hip_y"], m["torso"], m["both"],
                        bool(self.cfg["camera"].get("mirror", False)))


def ensure(ctx: FrameContext, cfg: dict) -> Optional[Geom]:
    """ctx.geom dacă există, altfel o geometrie calculată pe loc (fără netezire)."""
    if getattr(ctx, "geom", None) is not None:
        return ctx.geom
    if not ctx.pose:
        return None
    BodyGeometry(cfg).update(ctx)
    return ctx.geom
