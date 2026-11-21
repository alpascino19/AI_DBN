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

Filtre împotriva alarmelor false (o mână e "pe piept" doar dacă trece de toate):
    1. NU e la față: vârfurile degetelor / palma departe de bărbie și gură
       (bărbia vine de la CV1: ctx.signal("chin"); rezervă: gura + nasul din Pose).
       Mâna la bărbie are palma la nivelul gâtului -> fără filtru intra în zonă.
    2. zona începe SUB linia umerilor (zone_top > 0), nu deasupra.
    3. TOATĂ mâna e în zonă (min_inside din cele 21 de puncte), iar degetele nu
       trec deasupra liniei umerilor; din Pose: încheietura ȘI arătătorul în zonă.
    4. mâna STĂ (nu trece prin zonă): deplasarea în ultimele move_window_sec sub
       move_thr × unit. Fără istoric (primul cadru) nu respingem: durata o cere tracker-ul.
    5. mâna e LIPITĂ de corp: z-ul încheieturii (Pose) nu e cu mult în fața umerilor
       (oprit implicit: z-ul din Pose e zgomotos cu camera aproape).
    6. mâna NU e aproape de cameră: o mână ținută în fața camerei pare MARE față de
       corp. Lungimea palmei / unitatea corpului peste max_hand_scale = respinsă.
Motivul ultimei respingeri: ctx.signals["chest_reject"] (pentru reglaj / debug).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

from ..context import Detection, FrameContext, Hand, P
from ..schemas import HAND_ON_CHEST
from ..smoothing import Ema
from .geometry import ensure

Zone = tuple[float, float, float, float]   # x1, y1, x2, y2
TIP_IDX = (4, 8, 12, 16, 20)               # vârfurile degetelor (MediaPipe Hands)


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
        # filtre împotriva alarmelor false
        self.face_radius = c.get("face_radius", 0.45)        # × unit, în jurul bărbiei / gurii
        self.chin_margin = c.get("chin_margin", 0.10)        # degete mai sus de bărbie + atât = la față
        self.min_inside = c.get("min_inside", 0.6)           # fracțiunea din mână în zonă
        self.tips_above = c.get("tips_above_shoulder", 0.05) # degete peste umeri (× unit) = nu e piept
        self.move_window = c.get("move_window_sec", 0.5)
        self.move_thr = c.get("move_thr", 0.25)              # × unit în move_window
        self.depth_max = c.get("depth_max", 0.0)             # încheietura în fața umerilor (× unit); 0 = oprit
        # mâna aproape de cameră pare MARE față de corp (2D, mai sigur decât z-ul din Pose)
        self.max_hand_scale = c.get("max_hand_scale", 0.45)  # palma (încheietură->baza degetelor) / unit
        self.pose_scale_factor = c.get("pose_scale_factor", 1.8)  # din Pose: încheietură->arătător e mai lung
        self._hist: deque[tuple[float, list[tuple[float, float]]]] = deque()

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

    # ---------- filtre ----------
    def _face_points(self, ctx: FrameContext, unit: float) -> list[tuple[float, float]]:
        """Bărbia și gura: de la CV1 dacă există, altfel estimate din Pose."""
        pts = []
        for key in ("chin", "mouth"):
            v = ctx.signal(key)
            if v:
                pts.append((float(v[0]), float(v[1])))
        if not pts and ctx.pose:
            n, ml, mr = ctx.pose[P.NOSE], ctx.pose[P.MOUTH_L], ctx.pose[P.MOUTH_R]
            if min(ml.visibility, mr.visibility) >= 0.5:
                mx, my = (ml.x + mr.x) / 2, (ml.y + mr.y) / 2
                pts.append((mx, my))
                if n.visibility >= 0.5:
                    pts.append((mx, my + 0.8 * max(0.0, my - n.y)))      # bărbia, sub gură
        return pts

    def _at_face(self, pts: list[tuple[float, float]], face: list[tuple[float, float]], unit: float) -> bool:
        if not face:
            return False
        chin_y = max(y for _, y in face)
        r = self.face_radius * unit
        for x, y in pts:
            if any(math.hypot(x - fx, y - fy) < r for fx, fy in face):
                return True
            # degete la nivelul bărbiei sau mai sus, aproape pe orizontală de față
            if y < chin_y + self.chin_margin * unit and any(abs(x - fx) < 2 * r for fx, _ in face):
                return True
        return False

    def _moving(self, x: float, y: float, t: float, unit: float) -> bool:
        """Mâna s-a deplasat mult în ultimele move_window_sec? (fără istoric -> nu)."""
        thr = self.move_thr * unit
        for tt, pts in self._hist:
            if t - tt > self.move_window or not pts:
                continue
            if min(math.hypot(x - px, y - py) for px, py in pts) > thr:
                return True
        return False

    def _too_far_forward(self, ctx: FrameContext, x: float, y: float, unit: float) -> bool:
        """Încheietura (Pose) mult mai aproape de cameră decât umerii = mâna în fața corpului."""
        if not ctx.pose:
            return False
        sh_z = (ctx.pose[P.SHOULDER_L].z + ctx.pose[P.SHOULDER_R].z) / 2
        best = None
        for w in (P.WRIST_L, P.WRIST_R):
            lm = ctx.pose[w]
            if lm.visibility < self.min_vis:
                continue
            d = math.hypot(lm.x - x, lm.y - y)
            if d < 0.5 * unit and (best is None or d < best[0]):
                best = (d, lm.z)
        if best is None:
            return False
        fwd = (sh_z - best[1]) / unit                      # cât de în fața umerilor (× unit)
        ctx.signals["chest_depth"] = round(fwd, 2)         # pentru reglaj (afișat pe monitor)
        # z-ul din Pose e zgomotos (mai ales cu camera aproape): depth_max <= 0 = filtru oprit
        return self.depth_max > 0 and fwd > self.depth_max

    @staticmethod
    def _hand_size(src: str, hand_pts: list[tuple[float, float]]) -> float:
        """Mărimea mâinii în imagine: lungimea palmei (încheietura 0 -> baza mijlociului 9)
        sau lățimea ei (5 -> 17), cea mai mare (palma poate fi văzută oblic).
        Din Pose: încheietura -> arătătorul."""
        d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        if src == "hands" and len(hand_pts) >= 18:
            return max(d(hand_pts[0], hand_pts[9]), d(hand_pts[5], hand_pts[17]))
        if src == "pose" and len(hand_pts) >= 2:
            return d(hand_pts[0], hand_pts[1])
        return 0.0

    def _candidates(self, ctx: FrameContext) -> list[tuple[float, float, str, list[tuple[float, float]], list]]:
        """(x, y, sursă, puncte pentru filtrul de față, puncte pentru "toată mâna")."""
        out = []
        for h in ctx.hands:
            pts = [(l.x, l.y) for l in h.landmarks]
            tips = [pts[i] for i in TIP_IDX if i < len(pts)]
            out.append((*h.palm_center(), "hands", tips + [h.palm_center()], pts))
        if ctx.pose:
            for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
                a, b = ctx.pose[w], ctx.pose[i]
                if min(a.visibility, b.visibility) >= self.min_vis:
                    out.append(((a.x + b.x) / 2, (a.y + b.y) / 2, "pose", [(b.x, b.y)],
                                [(a.x, a.y), (b.x, b.y)]))
        return out

    def _hand_points(self, ctx: FrameContext) -> list[tuple[float, float, str]]:
        return [(x, y, src) for x, y, src, _, _ in self._candidates(ctx)]

    def update(self, ctx: FrameContext) -> None:
        z = self.zone(ctx)
        ctx.signals["chest_zone"] = z
        if z is None:
            self._hist.clear()
            ctx.add(Detection(HAND_ON_CHEST, False))
            return
        x1, y1, x2, y2 = z
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        phones = [o for o in ctx.objects if o.label == "cell phone"]
        g = ctx.geom
        ls, rs = ctx.pose[P.SHOULDER_L], ctx.pose[P.SHOULDER_R]
        unit = max(1e-3, self._unit.value or (g.unit if g is not None else 0.2))
        sh_y = g.sy if g is not None else (ls.y + rs.y) / 2
        face = self._face_points(ctx, unit)
        inside = lambda x, y: x1 <= x <= x2 and y1 <= y <= y2
        cands = self._candidates(ctx)

        # Pose e doar REZERVĂ, folosită numai când Hands nu vede NICIO mână. Altfel o mână
        # respinsă de Hands (ex. lângă cameră) sau brațul celălalt, ghicit de Pose în afara
        # imaginii (mâna în poală), ar trece prin estimarea grosieră din Pose.
        use_pose = not ctx.hands
        on_image = lambda pts: all(0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 for px, py in pts)

        best, side, n_in, reject, best_src = 0.0, None, 0, None, None
        for x, y, src, face_pts, hand_pts in cands:
            if src == "pose" and (not use_pose or not on_image(hand_pts)):
                continue
            if not inside(x, y):
                continue
            if any(p.contains(x, y, pad=0.1) for p in phones):
                reject = "phone"
                continue
            if self._at_face(face_pts, face, unit):                                  # 1
                reject = "face"
                continue
            frac = sum(inside(px, py) for px, py in hand_pts) / max(1, len(hand_pts))
            if frac < (self.min_inside if src == "hands" else 1.0):                  # 3
                reject = "partial"
                continue
            if src == "hands" and min(py for _, py in face_pts) < sh_y - self.tips_above * unit:
                reject = "fingers_up"
                continue
            scale = self._hand_size(src, hand_pts) / unit                            # 6
            limit = self.max_hand_scale * (self.pose_scale_factor if src == "pose" else 1.0)
            if src == "hands":
                ctx.signals["chest_hand_scale"] = round(scale, 2)
            if self.max_hand_scale > 0 and scale > limit:
                reject = "close"
                continue
            if self._moving(x, y, ctx.t, unit):                                      # 4
                reject = "moving"
                continue
            if self._too_far_forward(ctx, x, y, unit):                               # 5
                reject = "depth"
                continue
            n_in += 1
            # cât de central e punctul în zonă (1 = centru, 0 = margine)
            dx = abs(x - cx) / ((x2 - x1) / 2)
            dy = abs(y - cy) / ((y2 - y1) / 2)
            central = max(0.0, 1.0 - max(dx, dy))
            conf = (0.65 if src == "hands" else 0.55) + 0.35 * central
            if conf > best:
                best = conf
                best_src = src
                idx = P.SHOULDER_L if abs(x - ls.x) < abs(x - rs.x) else P.SHOULDER_R
                if g is not None and g.view != "front" and not g.shoulders_ok:
                    idx = g.near
                side = g.driver_side(idx, self.camera_side) if g is not None else None

        # istoric pentru filtrul de mișcare (toate punctele mâinilor, nu doar cele din zonă)
        self._hist.append((ctx.t, [(x, y) for x, y, *_ in cands]))
        while self._hist and ctx.t - self._hist[0][0] > self.move_window:
            self._hist.popleft()

        attrs = {}
        if n_in:
            attrs["side"] = side                           # partea șoferului (stânga = inimă)
            attrs["both_hands"] = n_in >= 3 or (n_in >= 2 and len(ctx.hands) >= 2)
        ctx.signals["chest_reject"] = None if n_in else reject
        ctx.signals["chest_src"] = best_src if n_in else None
        ctx.add(Detection(HAND_ON_CHEST, n_in > 0, min(best, 1.0), attrs))
