"""YOLO (clasa cell phone), rulat la fiecare N cadre. [CV2]

Între rulări întoarce ultimele cutii (cât timp nu sunt mai vechi de max_age_sec),
ca analizorul de telefon să aibă date la fiecare cadru.
"""
from __future__ import annotations

import logging

from ..camera import Frame
from ..config import model_path
from ..context import BBox

log = logging.getLogger(__name__)

PHONE_LABEL = "cell phone"   # clasa COCO 67


class ObjectDetector:
    def __init__(self, cfg: dict) -> None:
        from ultralytics import YOLO

        yc = cfg["yolo"]
        p = model_path(cfg, "yolo")
        # dacă nu e în models/, ultralytics îl descarcă după nume
        self._model = YOLO(str(p) if p.exists() else cfg["models"]["yolo"])
        names = self._model.names
        self._phone_ids = [i for i, n in names.items() if n == PHONE_LABEL]
        self.every_n = max(1, int(yc["every_n_frames"]))
        self.imgsz = yc["imgsz"]
        self.conf = yc["conf"]
        self.max_age = yc["max_age_sec"]
        self._count = 0
        self._last: list[BBox] = []
        self._last_t = -1e9

    def process(self, frame: Frame) -> list[BBox]:
        self._count += 1
        if self._count % self.every_n == 1 or self.every_n == 1:
            self._last = self._detect(frame)
            self._last_t = frame.timestamp
        elif frame.timestamp - self._last_t > self.max_age:
            self._last = []
        return self._last

    def _detect(self, frame: Frame) -> list[BBox]:
        res = self._model.predict(frame.image, imgsz=self.imgsz, conf=self.conf,
                                  classes=self._phone_ids, verbose=False)[0]
        out = []
        w, h = frame.width, frame.height
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(BBox(PHONE_LABEL, float(b.conf[0]), x1 / w, y1 / h, x2 / w, y2 / h))
        return out
