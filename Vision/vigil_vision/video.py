"""Imaginea camerei pentru dashboard (opțional, --video). [CV1 + CV2]

Fără --video, din cameră ies doar cifre (privacy mode). Cu --video, fiecare cameră trimite un JPEG mic
(VIDEO_WIDTH px lățime, VIDEO_FPS cadre/s, calitate VIDEO_QUALITY) pe vigil/video/face sau vigil/video/body:
același cadru (deja oglindit) din care se calculează punctele, deci punctele desenate de dashboard cad exact
pe față / corp. QoS 0, fără retain: un cadru pierdut nu se retrimite, nimic nu rămâne în broker.
Mărime tipică: 20-30 KB pe cadru, ~100-150 KB/s pe cameră.
"""
from __future__ import annotations

import cv2

VIDEO_WIDTH = 480
VIDEO_FPS = 5.0
VIDEO_QUALITY = 60


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

    def maybe_send(self, image, t: float) -> bool:
        if t < self._next:
            return False
        self._next = t + self.period
        data = encode_jpeg(image, self.width, self.quality)
        if data and self.pub.publish_bytes(self.topic, data):
            self.sent += 1
            return True
        return False
