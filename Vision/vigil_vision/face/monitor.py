"""Rulare live CV1: cameră -> față -> ochi + cap -> calibrare -> avertizări. [CV1]

    python -m vigil_vision.face.monitor --show            # calibrare 30 s
    python -m vigil_vision.face.monitor --show --calib 5  # calibrare scurtă la test
    python -m vigil_vision.face.monitor --show --mute     # fără sunete
    python -m vigil_vision.face.monitor --show --min-sound 2  # sunet doar la nivel >= 2

Fereastra: camera (neschimbată) + coloană cu detalii în dreapta + bandă de avertizări jos.
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
from ..perception.face import FaceDetector, Idx
from .calibration import Calibrator
from .eyes import EyeAnalyzer
from .head_pose import HeadPoseAnalyzer
from .alerts import AlertManager
from .yawn import YawnAnalyzer

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


def render(frame_img, res, eyes, head, yawn, calib, am, t):
    """Construiește imaginea finală: camera neschimbată + panou + avertizări."""
    cam = frame_img.copy()
    h, w = cam.shape[:2]
    level = am.level if calib.done else 0

    # pe cameră doar punctele ochilor, nasul și o margine în culoarea nivelului
    if res is not None:
        for i in Idx.LEFT_EYE + Idx.RIGHT_EYE:
            x, y = res.px(i).astype(int)
            cv2.circle(cam, (int(x), int(y)), 2, (0, 0, 255) if eyes.eyes_closed else (0, 255, 0), -1)
        nx, ny = res.px(Idx.NOSE_TIP).astype(int)
        cv2.circle(cam, (int(nx), int(ny)), 3, (255, 0, 0), -1)
    cv2.rectangle(cam, (0, 0), (w - 1, h - 1), COLORS[level], 4)

    H = max(h, 440)
    canvas = np.full((H + ALERTS_H, w + PANEL_W, 3), BG, np.uint8)
    canvas[:h, :w] = cam
    cv2.line(canvas, (w, 0), (w, H), LINE, 1)

    # ----- antet -----
    p = Panel(canvas, w, PANEL_W)
    text(canvas, "VIGIL - SOFER", p.x0, 30, 0.6, TEXT, 2)
    if not calib.done:
        badge, bcol = f"CALIBRARE {calib.progress(t) * 100:.0f}%", ORANGE
    elif level == 0:
        badge, bcol = "OK", GREEN
    else:
        badge, bcol = f"NIVEL {level}", COLORS[level]
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.5, 2)
    cv2.rectangle(canvas, (p.x1 - bw - 12, 12), (p.x1, 38), bcol, -1)
    text(canvas, badge, p.x1 - 6, 31, 0.5, TEXT, 2, right=True)
    p.y = 40
    if not calib.done:
        bar(canvas, p.x0, p.y + 10, p.x1 - p.x0, calib.progress(t), ORANGE)
        p.y += 10

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
    p.row("EAR", f(eyes.ear, ".3f"))

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

    # ----- gură -----
    p.section("GURA")
    p.row("Cascat", f"DA {yawn.open_sec:.1f}s" if yawn.yawning else "nu", ORANGE if yawn.yawning else TEXT)
    p.row("Cascate (5 min)", f"{yawn.yawns_recent}   (total {yawn.yawn_count})",
          RED if yawn.yawns_recent >= 3 else TEXT)

    # ----- bandă de avertizări -----
    y0 = H
    cv2.rectangle(canvas, (0, y0), (w + PANEL_W, y0 + ALERTS_H), (255, 255, 255), -1)
    cv2.line(canvas, (0, y0), (w + PANEL_W, y0), LINE, 1)
    text(canvas, "AVERTIZARI", 16, y0 + 26, 0.5, MUTED, 2)
    if not calib.done:
        text(canvas, "Calibrare: priviti drept inainte, normal, fara sa zambiti.", 16, y0 + 60, 0.55)
    elif not am.active:
        text(canvas, "Nicio avertizare - totul in regula.", 16, y0 + 60, 0.55, GREEN, 2)
    else:
        y = y0 + 40
        for a in am.active[:3]:
            cv2.rectangle(canvas, (12, y), (w + PANEL_W - 12, y + 26), COLORS[a.level], -1)
            text(canvas, f"NIVEL {a.level}   {a.message}", 22, y + 18, 0.5, TEXT, 1)
            text(canvas, f"{a.active_sec:.0f}s", w + PANEL_W - 22, y + 18, 0.5, TEXT, 1, right=True)
            y += 30
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description="Vigil CV1 live: ochi, privire, avertizări")
    p.add_argument("--index", default=0, type=int)
    p.add_argument("--show", action="store_true")
    p.add_argument("--calib", default=30.0, type=float, help="secunde de calibrare (0 = fără)")
    p.add_argument("--mute", action="store_true", help="fără sunete")
    p.add_argument("--min-sound", default=MIN_SOUND_LEVEL, type=int, help="nivelul minim cu sunet (1-3)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    eyes, head, wm = EyeAnalyzer(), HeadPoseAnalyzer(), AlertManager()
    yawn = YawnAnalyzer()
    calib = Calibrator(duration_sec=args.calib, min_samples=1 if args.calib == 0 else 30)

    with Camera(index=args.index) as cam, FaceDetector() as face:
        try:
            for frame in cam.reader(CV1):
                t = frame.timestamp
                res = face.process(frame)
                es = eyes.update(res, t)
                hs = head.update(res, t)
                ys = yawn.update(res, t)

                if not calib.done:
                    if calib.update(es, hs, t):
                        calib.apply(eyes, head)
                        print(f"\nCalibrare gata: {calib.result.to_dict()}")
                else:
                    dets = {**eyes.detections(), **yawn.detections()}
                    for w in wm.update(es, hs, dets, t):
                        print(f"{time.strftime('%H:%M:%S')} [nivel {w.level}] {w.message}")
                        if not args.mute and w.level >= args.min_sound:
                            beep(w.level)

                if args.show:
                    cv2.imshow("vigil CV1", render(frame.image, res, es, hs, ys, calib, wm, t))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
