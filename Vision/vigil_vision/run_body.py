"""Aplicația CORP (CV2): o cameră doar pentru corp. [CV1 + CV2]

    python -m vigil_vision.run_body --show
    python -m vigil_vision.run_body --show --index 1     # altă cameră decât body_camera din config.yaml

Rulează pipeline-ul CV2 (Pose + Hands + YOLO): mâna pe piept, respirația, telefonul,
postura / prăbușirea + avertizările corpului. Fără față: filtrul "mână la față" din
chest.py folosește gura/nasul din Pose, iar "telefon + privire" devine "telefon în mână".
Camera: secțiunea `body_camera` din config.yaml (ce lipsește se ia din `camera`).
Fața rulează separat: python -m vigil_vision.run_face
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from .camera import CV2
from .config import camera_cfg, load_config
from .face.alerts import AlertManager, body_rules
from .face.eyes import EyeState
from .face.head_pose import HeadState
from .face.monitor import MIN_SOUND_LEVEL, _NoPublish, beep, render
from .run_face import open_camera


def main() -> None:
    p = argparse.ArgumentParser(description="Vigil CORP: o cameră doar pentru corp")
    p.add_argument("--index", type=int, help="indexul camerei (implicit body_camera din config.yaml)")
    p.add_argument("--config", help="cale config.yaml (implicit Vision/config.yaml)")
    p.add_argument("--show", action="store_true", help="fereastra cu camera, panoul și avertizările")
    p.add_argument("--mute", action="store_true", help="fără sunete")
    p.add_argument("--min-sound", default=MIN_SOUND_LEVEL, type=int, help="nivelul minim cu sunet (1-3)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cam_cfg = camera_cfg(cfg, "body")
    if args.index is not None:
        cam_cfg["index"] = args.index
    cfg["camera"] = cam_cfg                       # pipeline-ul (geometria, oglinda) citește cfg["camera"]

    print(f"CORP: camera {cam_cfg['index']} (q = ieșire)")
    cam = open_camera(cam_cfg, "body")
    if cam is None:
        return

    from .pipeline import Pipeline
    am = AlertManager(body_rules())
    no_face_eyes, no_face_head = EyeState(), HeadState()   # regulile de corp nu folosesc fața
    pipeline = Pipeline(cfg, publisher=_NoPublish(), camera=cam)
    pipeline.face = None                          # fără față în această aplicație
    reader = cam.reader(CV2)
    try:
        for frame in reader:
            t = frame.timestamp
            ctx = pipeline.process(frame)
            dets = {n: (d.active, d.confidence) for n, d in ctx.detections.items()}
            for a in am.update(no_face_eyes, no_face_head, dets, t):
                print(f"{time.strftime('%H:%M:%S')} [CORP nivel {a.level}] {a.message}")
                if not args.mute and a.level >= args.min_sound:
                    beep(a.level)
            if args.show:
                img = render(frame.image, None, None, am, t, ctx=ctx, fps=reader.fps,
                             title="VIGIL - CORP", body=True)
                cv2.imshow("vigil - CORP", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if pipeline.body is not None:
            pipeline.body.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
