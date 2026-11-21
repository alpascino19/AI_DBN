"""Imaginea live a camerei pentru dashboard. [CV1 + CV2]

Pornită implicit când camera publică pe MQTT (--publish); --no-video o oprește (doar cifre). Fiecare cameră
trimite JPEG-uri (VIDEO_WIDTH px lățime, până la VIDEO_FPS cadre/s, calitate VIDEO_QUALITY) pe
vigil/video/face sau vigil/video/body: același cadru (deja oglindit) din care se calculează punctele, deci
punctele desenate de dashboard cad exact pe față / corp.
* Fir separat (VideoSender.start): imaginea are cititorul ei de cadre, la viteza camerei (30 cadre/s), nu la
  viteza procesării. Altfel camera corpului (Pose + Hands + YOLO, 7-11 cadre/s) n-ar putea trimite 20 cadre/s.
* „Live”, nu „în urmă”: dacă rețeaua nu a apucat să trimită cadrul precedent, cel nou se sare
  (Publisher.publish_bytes), în loc să se strângă o coadă de cadre vechi.
* QoS 0, fără retain: nimic nu rămâne în broker.
Mărime: ~30 KB pe cadru la 640 px, deci ~0.6 MB/s pe cameră la 20 cadre/s (prin hotspot: ~10 Mbit/s pentru
ambele camere). Pe un hotspot slab: --video-fps 10 sau --video-width 480; eticheta LIVE din dashboard arată
câte cadre/s ajung de fapt.
"""
from __future__ import annotations

import argparse
import logging
import threading

import cv2

VIDEO_WIDTH = 640
VIDEO_FPS = 20.0
VIDEO_QUALITY = 60

log = logging.getLogger(__name__)


def add_video_args(p: argparse.ArgumentParser) -> None:
    """Opțiunile comune: --no-video, --video-fps, --video-width (--video rămâne acceptat, e implicit)."""
    p.add_argument("--video", dest="video", action="store_true", default=True,
                   help="imaginea live a camerei în dashboard (implicit pornită cu --publish)")
    p.add_argument("--no-video", dest="video", action="store_false", help="fără imagine: doar cifre")
    p.add_argument("--video-fps", type=float, default=VIDEO_FPS, help=f"cadre/s trimise (implicit {VIDEO_FPS:g})")
    p.add_argument("--video-width", type=int, default=VIDEO_WIDTH, help=f"lățimea în pixeli (implicit {VIDEO_WIDTH})")


def encode_jpeg(image, width: int = VIDEO_WIDTH, quality: int = VIDEO_QUALITY) -> bytes:
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class VideoSender:
    """Trimite cel mult `fps` cadre pe secundă, prin Publisher.publish_bytes."""

    def __init__(self, pub, topic: str, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH,
                 quality: int = VIDEO_QUALITY) -> None:
        self.pub, self.topic = pub, topic
        self.period, self.width, self.quality = 1.0 / max(0.5, fps), width, quality
        self._next = 0.0
        self.sent = 0

    @classmethod
    def from_args(cls, pub, topic: str, args) -> "VideoSender":
        return cls(pub, topic, fps=args.video_fps, width=args.video_width)

    def maybe_send(self, image, t: float) -> bool:
        if t < self._next:
            return False
        # programul ține ritmul mediu (nu „t + perioadă”, care pierde câte o fracțiune la fiecare cadru și
        # dădea ~10 cadre/s în loc de 12); după o pauză lungă (cameră oprită) pornește din nou de la t
        self._next = self._next + self.period if t - self._next < self.period else t + self.period
        data = encode_jpeg(image, self.width, self.quality)
        if data and self.pub.publish_bytes(self.topic, data):
            self.sent += 1
            return True
        return False

    def start(self, camera, reader: str = "video") -> "VideoSender":
        """Firul imaginii: cititor propriu de cadre (camera.reader), independent de procesare."""
        frames = camera.reader(reader)

        def run() -> None:
            try:
                for frame in frames:          # se oprește odată cu camera
                    self.maybe_send(frame.image, frame.timestamp)
            except Exception as e:            # noqa: BLE001 (imaginea nu are voie să oprească detecția)
                log.warning("video %s oprit: %r", self.topic, e)

        threading.Thread(target=run, name=f"video-{self.topic.rsplit('/', 1)[-1]}", daemon=True).start()
        log.info("video live pe %s: până la %.0f cadre/s, %d px", self.topic, 1 / self.period, self.width)
        return self
