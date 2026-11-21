"""Aplicația FAȚĂ (CV1): o cameră doar pentru față. [CV1]

    python -m vigil_vision.run_face --show --calib 5
    python -m vigil_vision.run_face --show --index 0     # altă cameră decât face_camera din config.yaml

Rulează: ochi (EAR, PERCLOS, clipit), privire, căscat, pierderea cunoștinței + avertizările feței.
Camera: secțiunea `face_camera` din config.yaml (ce lipsește se ia din `camera`).
Corpul rulează separat: python -m vigil_vision.run_body
(Pasul următor: fiecare aplicație publică pe MQTT, iar mesajele se combină pentru ML1.)
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2

from .camera import CV1, Camera
from .config import camera_cfg, load_config
from .face.alerts import AlertManager, face_rules
from .face.monitor import MIN_SOUND_LEVEL, beep, render
from .face.stack import FaceStack

CAMERA_HELP = (
    "\nNu pot deschide camera {index!r} pentru {role}.\n"
    "  - Pe laptop e de obicei o singură cameră (index 0). Pentru a doua: cameră USB,\n"
    "    telefonul ca webcam (apare ca index 1) sau un URL de cameră IP în config.yaml.\n"
    "  - Schimbă indexul în config.yaml ({role}_camera: index: ...) sau rulează cu --index N.\n"
    "  - O cameră poate fi folosită de un singur program odată."
)


def open_camera(cam_cfg: dict, role: str) -> Optional[Camera]:
    """Pornește camera sau explică ce să faci dacă nu se poate deschide."""
    try:
        return Camera(**cam_cfg).start()
    except RuntimeError:
        print(CAMERA_HELP.format(index=cam_cfg["index"], role=role))
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Vigil FAȚĂ: o cameră doar pentru față")
    p.add_argument("--index", type=int, help="indexul camerei (implicit face_camera din config.yaml)")
    p.add_argument("--config", help="cale config.yaml (implicit Vision/config.yaml)")
    p.add_argument("--show", action="store_true", help="fereastra cu camera, panoul și avertizările")
    p.add_argument("--calib", default=30.0, type=float, help="secunde de calibrare (0 = fără)")
    p.add_argument("--mute", action="store_true", help="fără sunete")
    p.add_argument("--min-sound", default=MIN_SOUND_LEVEL, type=int, help="nivelul minim cu sunet (1-3)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cam_cfg = camera_cfg(cfg, "face")
    if args.index is not None:
        cam_cfg["index"] = args.index
    cfg.setdefault("face", {})["calib_sec"] = args.calib

    print(f"FAȚĂ: camera {cam_cfg['index']} (q = ieșire)")
    cam = open_camera(cam_cfg, "face")
    if cam is None:
        return

    am = AlertManager(face_rules())
    stack = FaceStack(cfg)
    reader = cam.reader(CV1)
    try:
        for frame in reader:
            t = frame.timestamp
            stack.process(frame)
            snap = stack.last
            if stack.calib.done:
                for a in am.update(snap.eyes, snap.head, stack.detections(), t):
                    print(f"{time.strftime('%H:%M:%S')} [FAȚĂ nivel {a.level}] {a.message}")
                    if not args.mute and a.level >= args.min_sound:
                        beep(a.level)
            if args.show:
                img = render(frame.image, snap, stack.calib, am, t, ctx=None, fps=reader.fps,
                             title="VIGIL - FATA", body=False)
                cv2.imshow("vigil - FATA", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        stack.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
