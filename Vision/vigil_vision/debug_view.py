"""Fereastră de debug pentru reglarea pragurilor (doar local, nimic nu se salvează). [CV2]"""
from __future__ import annotations

import cv2
import numpy as np

from .context import FrameContext, P

ARM = [(P.SHOULDER_L, P.SHOULDER_R), (P.SHOULDER_L, P.ELBOW_L), (P.ELBOW_L, P.WRIST_L),
       (P.SHOULDER_R, P.ELBOW_R), (P.ELBOW_R, P.WRIST_R), (P.SHOULDER_L, P.HIP_L),
       (P.SHOULDER_R, P.HIP_R)]


def draw(ctx: FrameContext, fps: float = 0.0) -> np.ndarray:
    img = ctx.frame.image.copy()
    h, w = img.shape[:2]
    px = lambda x, y: (int(x * w), int(y * h))

    if ctx.pose:
        for a, b in ARM:
            la, lb = ctx.pose[a], ctx.pose[b]
            if min(la.visibility, lb.visibility) > 0.5:
                cv2.line(img, px(la.x, la.y), px(lb.x, lb.y), (255, 200, 0), 2)
        nose, ls, rs = ctx.pose[P.NOSE], ctx.pose[P.SHOULDER_L], ctx.pose[P.SHOULDER_R]
        slumped = (ctx.signals.get("slump") or {}).get("state") == "slumped"
        if nose.visibility > 0.5:                       # linia cap -> umeri (căderea capului)
            mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
            cv2.line(img, px(nose.x, nose.y), px(*mid), (0, 0, 255) if slumped else (200, 200, 200), 2)
        g = ctx.geom
        if g is not None:                               # umărul apropiat de cameră + direcția corpului
            near = ctx.pose[g.near]
            cv2.circle(img, px(near.x, near.y), 7, (0, 255, 0), 2)
            if g.facing:
                a = px(g.sx, g.sy)
                cv2.arrowedLine(img, a, (a[0] + int(g.facing * g.unit * w * 0.6), a[1]), (0, 255, 0), 2)
    for hand in ctx.hands:
        for lm in hand.landmarks:
            cv2.circle(img, px(lm.x, lm.y), 2, (0, 255, 255), -1)
        cv2.circle(img, px(*hand.palm_center()), 6, (0, 165, 255), -1)

    z = ctx.signals.get("chest_zone")
    if z:
        active = ctx.detections.get("hand_on_chest")
        col = (0, 0, 255) if active and active.active else (0, 200, 0)
        cv2.rectangle(img, px(z[0], z[1]), px(z[2], z[3]), col, 2)
    for o in ctx.objects:
        cv2.rectangle(img, px(o.x1, o.y1), px(o.x2, o.y2), (255, 0, 255), 2)
        cv2.putText(img, f"phone {o.conf:.2f}", px(o.x1, o.y1 - 0.01),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    _draw_trace(img, ctx.signals.get("breathing_trace") or [],
                (ctx.signals.get("breathing") or {}).get("state") == "heavy")
    b = ctx.signals.get("breathing") or {}
    sl = ctx.signals.get("slump") or {}
    g = ctx.geom
    lines = [
        (f"view {g.view}  unit {g.unit:.2f}  facing {g.facing:+d}  near {'R' if g.near == P.SHOULDER_R else 'L'}"
         if g is not None else "view: corp negasit"),
        f"fps {fps:.1f}  phone {ctx.signals.get('phone', 0):.2f}",
        f"breathing {b.get('state')}  scor {b.get('score', '-')}  [{b.get('source', '-')}]",
        f"  {b.get('rate_bpm', '-')} resp/min  x{b.get('amp_ratio', '-')} normal  "
        f"ritmic {b.get('periodicity', '-')}{'' if b.get('calibrated') else '  (calibrare...)'}",
        f"postura {sl.get('state', '-')}  scor {sl.get('score', '-')}  cadere {sl.get('drop', '-')}  "
        f"nemiscat {sl.get('still_sec', '-')}s{'' if sl.get('calibrated', True) else '  (calibrare...)'}",
    ] + [f"EVENT {e.event} {e.duration_sec:.1f}s conf {e.confidence:.2f}" for e in ctx.events]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if txt.startswith("EVENT") else (255, 255, 255), 2)
    return img


def _draw_trace(img: np.ndarray, trace: list, heavy: bool) -> None:
    """Graficul respirației (ultimele ~10 s) în colțul din dreapta jos."""
    if len(trace) < 2:
        return
    h, w = img.shape[:2]
    gw, gh = 220, 70
    x0, y0 = w - gw - 10, h - gh - 10
    cv2.rectangle(img, (x0, y0), (x0 + gw, y0 + gh), (40, 40, 40), -1)
    scale = max(0.01, max(abs(v) for v in trace))          # scară minimă fixă: zgomotul rămâne mic
    pts = np.array([(x0 + int(i * gw / (len(trace) - 1)), y0 + gh // 2 - int(v / scale * gh / 2.2))
                    for i, v in enumerate(trace)], np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 255) if heavy else (0, 255, 0), 1)
    cv2.putText(img, "respiratie", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
