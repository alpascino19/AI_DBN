"""Aplicația FAȚĂ (CV1): o cameră doar pentru față. [CV1]

    python -m vigil_vision.run_face --show --calib 5
    python -m vigil_vision.run_face --show --index 0     # altă cameră decât face_camera din config.yaml
    python -m vigil_vision.run_face --show --publish --mqtt-host 172.22.118.134   # -> vigil/vision/face

Rulează: ochi (EAR, PERCLOS, clipit), privire, căscat, pierderea cunoștinței + avertizările feței.
Camera: secțiunea `face_camera` din config.yaml (ce lipsește se ia din `camera`).
Corpul rulează separat: python -m vigil_vision.run_body
Cu --publish trimite pe MQTT topicul `vigil/vision/face`; vigil_vision.merge (pe PC-ul cu
brokerul) îl combină cu `vigil/vision/body` într-un singur `vigil/vision` pentru ML1.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2

from .camera import CV1, Camera
from .config import camera_cfg, load_config
from .context import FaceOutput
from .events.tracker import EventTracker
from .face.alerts import AlertManager, face_rules
from .face.monitor import MIN_SOUND_LEVEL, beep, render
from .face.stack import FaceStack
from .panel import face_panel
from .schemas import VisionMessage

FACE_TOPIC = "vigil/vision/face"

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


def face_message(out: FaceOutput, events: list, fps: float) -> VisionMessage:
    """Mesajul vigil/vision cu DOAR câmpurile feței (corpul vine de la run_body)."""
    s = out.signals
    return VisionMessage(
        ear=s.get("ear"), perclos=s.get("perclos"), blink_rate=s.get("blink_rate"),
        gaze_on_road=s.get("gaze_on_road"), head_drop=s.get("head_drop"),
        events=events, landmarks=out.landmarks, ts=time.time(), camera_on=True,
        face_detected=out.detected, body_detected=False, fps=fps,
    )


def add_mqtt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--publish", action="store_true", help="trimite pe MQTT (vigil/vision/face sau /body)")
    p.add_argument("--mqtt-host", help="IP-ul brokerului (implicit mqtt.host din config.yaml)")


def main() -> None:
    p = argparse.ArgumentParser(description="Vigil FAȚĂ: o cameră doar pentru față")
    p.add_argument("--index", type=int, help="indexul camerei (implicit face_camera din config.yaml)")
    p.add_argument("--config", help="cale config.yaml (implicit Vision/config.yaml)")
    p.add_argument("--show", action="store_true", help="fereastra cu camera, panoul și avertizările")
    p.add_argument("--calib", default=30.0, type=float, help="secunde de calibrare (0 = fără)")
    p.add_argument("--mute", action="store_true", help="fără sunete")
    p.add_argument("--min-sound", default=MIN_SOUND_LEVEL, type=int, help="nivelul minim cu sunet (1-3)")
    add_mqtt_args(p)
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
    pub, tracker = None, EventTracker(cfg)
    if args.publish:
        from .publisher import Publisher
        pub = Publisher(cfg, topic=FACE_TOPIC, host=args.mqtt_host)
    period, next_pub = 1.0 / max(1, cfg.get("publish_hz", 10)), 0.0
    reader = cam.reader(CV1)
    try:
        for frame in reader:
            t = frame.timestamp
            out = stack.process(frame)
            snap = stack.last
            events = tracker.update(out.detections, t)
            if pub is not None and t >= next_pub:
                pub.publish({**face_message(out, events, reader.fps).to_dict(),
                             "face_panel": face_panel(snap, stack.calib, am, t, reader.fps)})  # pentru dashboard
                next_pub = t + period
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
        if pub is not None:
            pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
