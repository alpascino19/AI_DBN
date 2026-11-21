"""Telefon: cutia YOLO corelată cu mâna și privirea. [CV2]

Un telefon văzut de YOLO nu e neapărat folosit (poate sta în suport). Scorul crește dacă:
    - o mână e pe telefon (Hands: palmă/vârfuri de degete; Pose: încheietura)
    - telefonul e lângă față (convorbire)
    - șoferul nu se uită la drum (ctx.signal("gaze_on_road") == False, de la CV1)
ctx.signals["phone"] = scor netezit 0..1  -> câmpul "phone" din vigil/vision
"""
from __future__ import annotations

import math

from ..context import BBox, Detection, FrameContext, P
from ..schemas import PHONE_USE
from .geometry import ensure

FINGER_IDX = (0, 4, 8, 12)        # încheietura, vârful degetului mare, arătător, mijlociu


class PhoneAnalyzer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        c = cfg["phone"]
        self.pad = c["hand_pad"]
        self.min_score = c["min_score"]
        self.score = 0.0             # netezit, pentru câmpul "phone"

    def _in_hand(self, ctx: FrameContext, box: BBox) -> bool:
        for h in ctx.hands:
            if box.contains(*h.palm_center(), pad=self.pad):
                return True
            if any(box.contains(h.landmarks[i].x, h.landmarks[i].y, pad=self.pad) for i in FINGER_IDX):
                return True
        if ctx.pose:
            for i in (P.WRIST_L, P.WRIST_R, P.INDEX_L, P.INDEX_R):
                lm = ctx.pose[i]
                if lm.visibility >= 0.5 and box.contains(lm.x, lm.y, pad=self.pad):
                    return True
        return False

    def _near_face(self, ctx: FrameContext, box: BBox) -> bool:
        if not ctx.pose or ctx.pose[P.NOSE].visibility < 0.5:
            return False
        g = ensure(ctx, self.cfg)
        unit = g.unit if g is not None else 0.2
        nx, ny = ctx.pose[P.NOSE].x, ctx.pose[P.NOSE].y
        bx, by = box.center()
        return math.hypot(bx - nx, by - ny) < 0.6 * unit

    def update(self, ctx: FrameContext) -> None:
        gaze = ctx.signal("gaze_on_road")              # True / False / None (CV1 lipsă)
        best, attrs = 0.0, {}
        for box in ctx.objects:
            if box.label != "cell phone":
                continue
            in_hand = self._in_hand(ctx, box)
            near_face = self._near_face(ctx, box)
            looking = gaze is False
            raw = box.conf * (0.4 + 0.35 * in_hand + 0.15 * near_face + 0.25 * looking)
            raw = min(raw, 1.0)
            if raw > best and (in_hand or near_face):
                best = raw
                attrs = {
                    "in_hand": in_hand,
                    "near_face": near_face,
                    "looking_at_phone": None if gaze is None else looking,
                    "usage": "call" if near_face else ("looking" if looking else "holding"),
                }
            elif raw > best:
                best = min(raw, self.min_score * 0.9)   # telefon în suport: sub prag

        self.score = 0.6 * self.score + 0.4 * best if best < self.score else best
        ctx.signals["phone"] = round(self.score, 3)
        ctx.add(Detection(PHONE_USE, best >= self.min_score and bool(attrs), best, attrs))
