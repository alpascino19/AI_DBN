"""MediaPipe Pose + Hands. [CV2]

Folosește MediaPipe Tasks (PoseLandmarker, HandLandmarker) în modul VIDEO.
Modelele .task stau în Vision/models/ și se descarcă singure dacă lipsesc.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

from ..camera import Frame
from ..config import model_path
from ..context import Hand, Landmark

log = logging.getLogger(__name__)

MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
}


def ensure_model(path: Path, url: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Descarc %s ...", path.name)
        urllib.request.urlretrieve(url, path)
    return path


class BodyPerception:
    """process(frame) -> (pose | None, [Hand, ...])"""

    def __init__(self, cfg: dict) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        pc = cfg["perception"]
        pose_path = ensure_model(model_path(cfg, "pose"), MODEL_URLS["pose"])
        hand_path = ensure_model(model_path(cfg, "hand"), MODEL_URLS["hand"])

        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=pc["pose_min_conf"],
            min_pose_presence_confidence=pc["pose_min_conf"],
            min_tracking_confidence=pc["pose_min_conf"],
        ))
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=pc["num_hands"],
            min_hand_detection_confidence=pc["hand_min_conf"],
            min_hand_presence_confidence=pc["hand_min_conf"],
            min_tracking_confidence=pc["hand_min_conf"],
        ))
        self._last_ts = -1

    def process(self, frame: Frame) -> tuple[Optional[list[Landmark]], list[Hand]]:
        # MediaPipe VIDEO cere timestamp-uri strict crescătoare
        ts = max(frame.timestamp_ms, self._last_ts + 1)
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame.rgb)

        pose = None
        pr = self._pose.detect_for_video(img, ts)
        if pr.pose_landmarks:
            pose = [Landmark(l.x, l.y, l.z, getattr(l, "visibility", 1.0) or 0.0)
                    for l in pr.pose_landmarks[0]]

        hands: list[Hand] = []
        hr = self._hands.detect_for_video(img, ts)
        for i, lms in enumerate(hr.hand_landmarks or []):
            cat = hr.handedness[i][0] if hr.handedness and hr.handedness[i] else None
            hands.append(Hand(
                landmarks=[Landmark(l.x, l.y, l.z) for l in lms],
                handedness=cat.category_name if cat else "Unknown",
                score=cat.score if cat else 1.0,
            ))
        return pose, hands

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
