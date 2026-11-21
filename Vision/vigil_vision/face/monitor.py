"""Monitor live Vigil: FATA (CV1) + CORP (CV2) -> avertizari. [CV1]

    python -m vigil_vision.face.monitor --show                 # față + corp, calibrare 30 s
    python -m vigil_vision.face.monitor --show --calib 5       # calibrare scurtă la test
    python -m vigil_vision.face.monitor --show --no-body       # doar fața (mai rapid)
    python -m vigil_vision.face.monitor --show --publish       # + trimite vigil/vision pe MQTT
    python -m vigil_vision.face.monitor --show --mute          # fără sunete
    python -m vigil_vision.face.monitor --show --min-sound 2   # sunet doar la nivel >= 2

Un singur proces, o singură cameră, un cadru la rând:
    FaceStack (face/stack.py)  -> ochi, privire, căscat, calibrare -> FaceOutput
    Pipeline.process (CV2)     -> Pose + Hands + YOLO -> piept, respirație, telefon, postură
                                  (primește FaceOutput: gaze_on_road, jaw_open)
    AlertManager               -> avertizări din față + corp
Fereastra: camera + coloană cu detalii (ochi, privire, gură, corp) + bandă de avertizări.
În consolă apar doar avertizările noi.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
import unicodedata
from typing import Optional

import cv2
import numpy as np

from ..camera import CV1, Camera
from ..config import load_config
from ..context import FaceOutput, P
from ..perception.face import Idx
from .alerts import AlertManager
from .stack import FaceStack

COLORS = {0: (0, 200, 0), 1: (0, 220, 255), 2: (0, 140, 255), 3: (0, 0, 255)}  # BGR


def ascii_text(s: str) -> str:
    """cv2.putText nu știe diacritice: ă -> a, ș -> s etc."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


# ---------- sunete: un tipar diferit pe nivel ----------
# Fiecare nivel = listă de (frecvență Hz, durată ms). Frecvență 0 = pauză.
# Modifică aici ca să schimbi sunetele.
SOUNDS: dict[int, list[tuple[int, int]]] = {
    1: [(700, 120)],                                       # un bip scurt, grav
    2: [(900, 150), (0, 80), (900, 150)],                  # două bipuri
    3: [(1400, 200), (0, 60), (1400, 200), (0, 60), (1400, 200)],  # trei bipuri ascuțite
}
MIN_SOUND_LEVEL = 1          # nivelul minim care face sunet (1 = toate)

_sound_lock = threading.Lock()


def _play(pattern: list[tuple[int, int]]) -> None:
    if not _sound_lock.acquire(blocking=False):
        return  # deja sună ceva: nu suprapunem sunetele
    try:
        import winsound
        for freq, ms in pattern:
            if freq <= 0:
                time.sleep(ms / 1000)
            else:
                winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)  # fallback: clopoțelul consolei
    finally:
        _sound_lock.release()


def beep(level: int) -> None:
    """Pornește sunetul într-un thread separat, ca imaginea să nu înghețe."""
    pattern = SOUNDS.get(level)
    if pattern and level >= MIN_SOUND_LEVEL:
        threading.Thread(target=_play, args=(pattern,), daemon=True).start()


# ---------- design: [ cameră | coloană cu detalii ] + bandă de avertizări dedesubt ----------
PANEL_W = 300
ALERTS_H = 132
BG = (245, 245, 245)        # fundal panou (gri foarte deschis)
TEXT = (20, 20, 20)         # negru
MUTED = (110, 110, 110)     # etichete
LINE = (210, 210, 210)
GREEN, ORANGE, RED = (60, 170, 60), (0, 140, 255), (0, 0, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DIRECTIONS = {"center": "la drum", "left": "STANGA", "right": "DREAPTA", "down": "JOS", "up": "SUS"}


def text(img, s: str, x: int, y: int, sc: float = 0.48, col=TEXT, th: int = 1, right: bool = False) -> None:
    s = ascii_text(s)
    if right:
        x -= cv2.getTextSize(s, FONT, sc, th)[0][0]
    cv2.putText(img, s, (x, y), FONT, sc, col, th, cv2.LINE_AA)


def bar(img, x: int, y: int, w: int, frac: Optional[float], col) -> None:
    cv2.rectangle(img, (x, y - 9), (x + w, y + 1), LINE, -1)
    if frac is not None:
        cv2.rectangle(img, (x, y - 9), (x + int(w * max(0.0, min(1.0, frac))), y + 1), col, -1)


def grade(v: Optional[float], warn: float, bad: float):
    if v is None or v < warn:
        return GREEN
    return ORANGE if v < bad else RED


class Panel:
    """Scrie rânduri "etichetă ........ valoare" de sus în jos."""

    def __init__(self, img, x0: int, width: int) -> None:
        self.img, self.x0, self.x1, self.y = img, x0 + 16, x0 + width - 16, 0

    def section(self, title: str) -> None:
        self.y += 30
        text(self.img, title, self.x0, self.y, 0.5, MUTED, 2)
        cv2.line(self.img, (self.x0, self.y + 6), (self.x1, self.y + 6), LINE, 1)
        self.y += 4

    def row(self, label: str, value: str, col=TEXT, frac: Optional[float] = None, frac_col=None) -> None:
        self.y += 22
        text(self.img, label, self.x0, self.y, 0.46, MUTED)
        text(self.img, value, self.x1, self.y, 0.5, col, 1 + (col != TEXT), right=True)
        if frac_col is not None:  # rând cu bară
            bar(self.img, self.x0 + 118, self.y - 1, 80, frac, frac_col)


POSTURE = {"upright": ("drept", None), "leaning": ("aplecat", ORANGE), "slumped": ("PRABUSIT", RED)}
# conexiunile MediaPipe Hands (încheietura 0 -> fiecare deget)
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _draw_camera(cam, snap, ctx, level) -> None:
    """Pe cameră: ochi + nas (CV1), zona pieptului, mâini, telefon, linia posturii (CV2)."""
    h, w = cam.shape[:2]
    px = lambda x, y: (int(x * w), int(y * h))
    if ctx is not None:
        z = ctx.signals.get("chest_zone")
        if z:
            on = ctx.detections.get("hand_on_chest")
            cv2.rectangle(cam, px(z[0], z[1]), px(z[2], z[3]), (0, 0, 255) if on and on.active else (0, 200, 0), 2)
        for hand in ctx.hands:                       # toată mâna: 21 de puncte + oasele degetelor
            pts = [px(l.x, l.y) for l in hand.landmarks]
            for a, b in HAND_BONES:
                if a < len(pts) and b < len(pts):
                    cv2.line(cam, pts[a], pts[b], (0, 200, 255), 1, cv2.LINE_AA)
            for q in pts:
                cv2.circle(cam, q, 2, (0, 200, 255), -1)
            cv2.circle(cam, px(*hand.palm_center()), 6, (0, 140, 255), -1)
        for o in ctx.objects:
            cv2.rectangle(cam, px(o.x1, o.y1), px(o.x2, o.y2), (255, 0, 255), 2)
        g = ctx.geom
        if ctx.pose and g is not None:
            head = [(ctx.pose[i].x, ctx.pose[i].y) for i in (P.NOSE, 2, 5, 7, 8) if ctx.pose[i].visibility >= 0.5]
            if head:
                hx, hy = sum(x for x, _ in head) / len(head), sum(y for _, y in head) / len(head)
                state = (ctx.signals.get("slump") or {}).get("state", "upright")
                col = {"leaning": (0, 165, 255), "slumped": (0, 0, 255)}.get(state, (200, 200, 200))
                cv2.line(cam, px(hx, hy), px(g.sx, g.sy), col, 3)
    res = snap.res if snap else None
    if res is not None:
        closed = snap.eyes.eyes_closed
        for i in Idx.LEFT_EYE + Idx.RIGHT_EYE:
            x, y = res.px(i).astype(int)
            cv2.circle(cam, (int(x), int(y)), 2, (0, 0, 255) if closed else (0, 255, 0), -1)
        nx, ny = res.px(Idx.NOSE_TIP).astype(int)
        cv2.circle(cam, (int(nx), int(ny)), 3, (255, 0, 0), -1)
    cv2.rectangle(cam, (0, 0), (w - 1, h - 1), COLORS[level], 4)


def _draw_panel(panel, snap, calib, ctx, level, t, title: str = "VIGIL - SOFER",
                body: bool = True) -> int:
    """Coloana din dreapta. snap=None -> fără secțiunile feței; body=False -> fără CORP.
    Întoarce înălțimea folosită."""
    p = Panel(panel, 0, PANEL_W)
    text(panel, title, p.x0, 30, 0.6, TEXT, 2)
    calibrating = calib is not None and not calib.done
    if calibrating:
        badge, bcol = f"CALIBRARE {calib.progress(t) * 100:.0f}%", ORANGE
    elif level == 0:
        badge, bcol = "OK", GREEN
    else:
        badge, bcol = f"NIVEL {level}", COLORS[level]
    (bw, _), _ = cv2.getTextSize(badge, FONT, 0.5, 2)
    cv2.rectangle(panel, (p.x1 - bw - 12, 12), (p.x1, 38), bcol, -1)
    text(panel, badge, p.x1 - 6, 31, 0.5, TEXT, 2, right=True)
    p.y = 40
    if calibrating:
        bar(panel, p.x0, p.y + 10, p.x1 - p.x0, calib.progress(t), ORANGE)
        p.y += 10

    if snap is not None:
        _face_sections(p, snap)
    if body:
        _body_section(p, ctx)
    return p.y


def _face_sections(p: "Panel", snap) -> None:
    eyes, head, yawn = snap.eyes, snap.head, snap.yawn
    f = lambda v, fmt=".2f": "-" if v is None else format(v, fmt)

    # ----- ochi -----
    p.section("OCHI")
    if not eyes.face_visible:
        p.row("Stare", "fara fata", ORANGE)
    elif eyes.eyes_closed:
        p.row("Stare", f"INCHISI {eyes.closed_sec:.1f}s", RED)
    else:
        p.row("Stare", "deschisi")
    p.row("Inchidere", f(eyes.closure), frac=eyes.closure, frac_col=grade(eyes.closure, 0.5, 0.72))
    p.row("Clipiri / min", f(eyes.blink_rate, ".0f"))
    p.row("PERCLOS", "-" if eyes.perclos is None else f"{eyes.perclos * 100:.0f}%",
          frac=None if eyes.perclos is None else eyes.perclos / 0.3, frac_col=grade(eyes.perclos, 0.10, 0.15))
    p.row("Ultimele 15 s", "-" if eyes.perclos_recent is None else
          f"{eyes.perclos_recent * 100:.0f}% inchisi, {eyes.blinks_recent} clipiri",
          GREEN if eyes.recovered else TEXT)

    # ----- privire -----
    p.section("PRIVIRE")
    if not head.face_visible:
        p.row("Directie", "fara fata", ORANGE)
    else:
        d = head.direction or "-"
        p.row("Directie", DIRECTIONS.get(d, d), TEXT if d == "center" else ORANGE)
    p.row("Cap stg/dr  sus/jos", f"{f(head.yaw, '.0f')} / {f(head.pitch, '.0f')} grade")
    p.row("Ochi in jos", f(head.eye_down), frac=head.eye_down, frac_col=grade(head.eye_down, 0.25, 0.35))
    p.row("In afara drumului", f"{head.off_road_sec:.1f}s", RED if head.off_road_sec >= 2 else TEXT)
    p.row("Ultimele 10 s", "-" if head.off_road_ratio is None else f"{head.off_road_ratio * 100:.0f}%",
          frac=head.off_road_ratio, frac_col=grade(head.off_road_ratio, 0.3, 0.5))

    # ----- conștiență (face/unconscious.py) -----
    us = snap.unconscious
    p.section("CONSTIENTA")
    label = {"ok": "normal", "suspect": "SUSPECT", "unconscious": f"INCONSTIENT {us.since_sec:.0f}s"}[us.state]
    p.row("Stare", label, {"ok": TEXT, "suspect": ORANGE, "unconscious": RED}[us.state],
          frac=us.score, frac_col=grade(us.score, 0.45, 0.65))
    p.row("Cap cazut / nemiscat", f"{'DA' if us.dropped else 'nu'} / {us.still_sec:.1f}s",
          ORANGE if us.dropped else TEXT)

    # ----- gură -----
    p.section("GURA")
    p.row("Cascat", f"DA {yawn.open_sec:.1f}s" if yawn.yawning else "nu", ORANGE if yawn.yawning else TEXT)
    p.row("Cascate (5 min)", f"{yawn.yawns_recent}   (total {yawn.yawn_count})",
          RED if yawn.yawns_recent >= 3 else TEXT)



def _body_section(p: "Panel", ctx) -> None:
    p.section("CORP")
    if ctx is None:
        p.row("Corp", "dezactivat (--no-body)", MUTED)
        return
    if ctx.pose is None:
        p.row("Corp", "nu se vede", ORANGE)

    def on(name: str) -> bool:
        d = ctx.detections.get(name)
        return bool(d and d.active)

    why = {"face": "la fata", "partial": "partial", "fingers_up": "degete sus", "moving": "in miscare",
           "depth": "in aer", "phone": "telefon", "close": "langa camera"}.get(ctx.signals.get("chest_reject"))
    src = {"hands": "mana", "pose": "schelet"}.get(ctx.signals.get("chest_src"), "")
    p.row("Mana pe piept", (f"DA ({src})" if src else "DA") if on("hand_on_chest")
          else (f"nu ({why})" if why else "nu"),
          RED if on("hand_on_chest") else TEXT)
    scale = ctx.signals.get("chest_hand_scale")
    p.row("Marime mana", "-" if scale is None else f"{scale:.2f}",
          ORANGE if ctx.signals.get("chest_reject") == "close" else TEXT)
    b = ctx.signals.get("breathing")
    if b:                                            # doar cu breathing.enabled: true
        heavy = b.get("state") == "heavy"
        rate = b.get("rate_bpm")
        br = "GREA" if heavy else (b.get("state") or "-")
        score = b.get("score") if isinstance(b.get("score"), (int, float)) else None
        p.row("Respiratie", br, RED if heavy else TEXT, frac=score, frac_col=grade(score, 0.40, 0.55))
        rr = f"{rate:.0f} / min" if isinstance(rate, (int, float)) else "-"
        if not b.get("calibrated", True):
            rr += "  (calibrare)"
        p.row("Ritm respiratie", rr)
    phone = float(ctx.signals.get("phone", 0.0) or 0.0)
    pd = ctx.detections.get("phone_use")
    usage = {"call": "apel", "looking": "se uita", "holding": "in mana"}.get(
        (pd.attrs or {}).get("usage") if pd else None)
    if pd and pd.active:
        ph = usage or "da"
    else:
        ph = "nu" if phone < 0.05 else f"{phone:.2f}"
    p.row("Telefon", ph, ORANGE if pd and pd.active else TEXT, frac=phone, frac_col=grade(phone, 0.25, 0.4))
    sl = ctx.signals.get("slump") or {}
    label, col = POSTURE.get(sl.get("state"), (sl.get("state") or "-", None))
    if sl.get("direction") == "side":
        label = "CAZUT LATERAL"
    if sl.get("out"):
        label, col = {"lost": ("nu se vede", ORANGE), "far": ("prea departe", ORANGE)}[sl["out"]]
    if sl and not sl.get("calibrated", True):
        label += " (calib.)"
    p.row("Postura", label, col or TEXT)
    # valorile pentru reglaj live (config.yaml: slump)
    num = lambda k: sl.get(k) if isinstance(sl.get(k), (int, float)) else None
    side, drop, lean, ang = num("side_deg"), num("drop"), num("lean"), num("angle")
    p.row("Inclinare laterala", "-" if side is None else f"{side:.0f} grade",
          frac=None if side is None else side / 40, frac_col=grade(side, 15, 25))
    p.row("Cap coborat", "-" if drop is None else f"{drop:.2f}",
          frac=drop, frac_col=grade(drop, 0.25, 0.6))
    p.row("Aplecare", "-" if lean is None else f"{lean:.2f}", frac=lean, frac_col=grade(lean, 0.3, 0.6))
    p.row("Unghi cap (aplecat)", "-" if ang is None else f"{ang:.0f} grade",
          ORANGE if ang is not None and ang >= 12 else TEXT)
    p.row("Prabusire (scor)", "-" if num("score") is None else f"{num('score'):.2f}",
          frac=num("score"), frac_col=grade(num("score"), 0.38, 0.50))
    p.row("Nemiscat", f"{sl.get('still_sec', 0.0) or 0.0:.1f}s")


def render(frame_img, snap, calib, am, t, ctx=None, fps: float = 0.0,
           title: str = "VIGIL - SOFER", body: bool = True):
    """Imaginea finală: camera + coloana cu detalii + banda de avertizări.
    Doar față: ctx=None, body=False. Doar corp: snap=None, calib=None."""
    cam = frame_img.copy()
    h, w = cam.shape[:2]
    calib_done = calib is None or calib.done
    level = am.level if calib_done else 0
    _draw_camera(cam, snap, ctx, level)

    panel = np.full((1200, PANEL_W, 3), BG, np.uint8)
    used = _draw_panel(panel, snap, calib, ctx, level, t, title, body)
    H = max(h, used + 16)

    canvas = np.full((H + ALERTS_H, w + PANEL_W, 3), BG, np.uint8)
    canvas[:h, :w] = cam
    canvas[:H, w:] = panel[:H]
    cv2.line(canvas, (w, 0), (w, H), LINE, 1)

    # ----- bandă de avertizări -----
    y0, W = H, w + PANEL_W
    cv2.rectangle(canvas, (0, y0), (W, y0 + ALERTS_H), (255, 255, 255), -1)
    cv2.line(canvas, (0, y0), (W, y0), LINE, 1)
    text(canvas, "AVERTIZARI", 16, y0 + 26, 0.5, MUTED, 2)
    if fps:
        text(canvas, f"{fps:.0f} fps", W - 16, y0 + 26, 0.45, MUTED, 1, right=True)
    if not calib_done:
        text(canvas, "Calibrare: priviti drept inainte, stati normal, respirati linistit.", 16, y0 + 60, 0.55)
    elif not am.active:
        text(canvas, "Nicio avertizare - totul in regula.", 16, y0 + 60, 0.55, GREEN, 2)
    else:
        y = y0 + 40
        for a in am.active[:3]:
            cv2.rectangle(canvas, (12, y), (W - 12, y + 26), COLORS[a.level], -1)
            text(canvas, f"NIVEL {a.level}   {a.message}", 22, y + 18, 0.5, TEXT, 1)
            text(canvas, f"{a.active_sec:.0f}s", W - 22, y + 18, 0.5, TEXT, 1, right=True)
            y += 30
    return canvas


class _FaceFeed:
    """Dă pipeline-ului CV2 FaceOutput-ul cadrului curent (în loc de FaceWorker)."""

    def __init__(self) -> None:
        self.out: Optional[FaceOutput] = None

    def latest(self, now: float) -> Optional[FaceOutput]:
        return self.out


class _NoPublish:
    def publish(self, msg) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Vigil live: față + corp, avertizări")
    p.add_argument("--index", type=int, help="indexul camerei (implicit din config.yaml)")
    p.add_argument("--config", help="cale config.yaml (implicit Vision/config.yaml)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--calib", default=30.0, type=float, help="secunde de calibrare față (0 = fără)")
    p.add_argument("--no-body", action="store_true", help="doar fața (fără Pose/Hands/YOLO)")
    p.add_argument("--publish", action="store_true", help="trimite vigil/vision pe MQTT (cu corp)")
    p.add_argument("--mute", action="store_true", help="fără sunete")
    p.add_argument("--min-sound", default=MIN_SOUND_LEVEL, type=int, help="nivelul minim cu sunet (1-3)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.index is not None:
        cfg["camera"]["index"] = args.index
    cfg.setdefault("face", {})["calib_sec"] = args.calib

    am = AlertManager()
    stack = FaceStack(cfg)
    pipeline, feed, pub = None, _FaceFeed(), _NoPublish()

    with Camera(**cfg["camera"]) as cam:
        if not args.no_body:
            from ..pipeline import Pipeline
            if args.publish:
                from ..publisher import Publisher
                pub = Publisher(cfg)
            pipeline = Pipeline(cfg, publisher=pub, camera=cam)
            pipeline.face = feed          # fața o dăm noi, din același cadru
        reader = cam.reader(CV1)
        period = 1.0 / max(1, cfg.get("publish_hz", 10))
        next_pub = 0.0
        try:
            for frame in reader:
                t = frame.timestamp
                feed.out = stack.process(frame)
                snap = stack.last
                ctx = pipeline.process(frame) if pipeline is not None else None

                if stack.calib.done:
                    dets = {n: (d.active, d.confidence) for n, d in (ctx.detections if ctx else {}).items()}
                    dets.update(stack.detections())
                    for w in am.update(snap.eyes, snap.head, dets, t):
                        print(f"{time.strftime('%H:%M:%S')} [nivel {w.level}] {w.message}")
                        if not args.mute and w.level >= args.min_sound:
                            beep(w.level)

                if pipeline is not None and args.publish and t >= next_pub:
                    pub.publish(pipeline.build_message(ctx, reader.fps))
                    next_pub = t + period

                if args.show:
                    cv2.imshow("vigil", render(frame.image, snap, stack.calib, am, t, ctx, reader.fps))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            stack.close()
            if pipeline is not None and pipeline.body is not None:
                pipeline.body.close()
            pub.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
