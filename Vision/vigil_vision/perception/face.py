"""MediaPipe FaceLandmarker: landmarks, blendshapes, matricea capului. [CV1]

Strat de "percepție" subțire: rulează modelul O SINGURĂ DATĂ pe cadru și
întoarce un `FaceResult`. Nu calculează EAR / PERCLOS / poziția capului;
asta fac analizorii din `face/*`, care citesc doar din `FaceResult`.

Utilizare:
    with Camera() as cam, FaceDetector() as face:
        for frame in cam.reader(CV1):
            res = face.process(frame)     # FaceResult sau None (fără față)
            if res:
                res.landmarks             # (478, 3) normalizate [0..1]
                res.points_px             # (478, 2) în pixeli
                res.blend("eyeBlinkLeft") # 0..1
                res.matrix                # 4x4, poziția capului

Modelul `face_landmarker.task` (~3.7 MB) stă în Vision/models/.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..camera import Frame

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


# ---------- indici utili din cele 478 de puncte (pentru face/*) ----------
class Idx:
    # 6 puncte pe ochi, în ordinea pentru EAR: colț ext., sus1, sus2, colț int., jos2, jos1
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)
    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    RIGHT_IRIS = 468   # centrul irisului (doar cu modelul cu iris: 478 de puncte)
    LEFT_IRIS = 473
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LIP_TOP = 13
    LIP_BOTTOM = 14


@dataclass
class FaceResult:
    """Rezultatul pentru o față dintr-un cadru."""

    frame_id: int
    timestamp: float                 # time.monotonic() al cadrului
    width: int
    height: int
    landmarks: np.ndarray            # (N, 3) float32, x/y normalizate, z relativ
    blendshapes: dict[str, float] = field(default_factory=dict)
    matrix: Optional[np.ndarray] = None  # (4, 4) transformarea capului

    @property
    def points_px(self) -> np.ndarray:
        """(N, 2) coordonate în pixeli."""
        return self.landmarks[:, :2] * np.array([self.width, self.height], dtype=np.float32)

    def px(self, idx: int) -> np.ndarray:
        """Un singur punct în pixeli."""
        return self.points_px[idx]

    def blend(self, name: str, default: float = 0.0) -> float:
        """Scorul unui blendshape, ex. 'eyeBlinkLeft', 'jawOpen'."""
        return self.blendshapes.get(name, default)


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Descarcă modelul o singură dată, dacă lipsește."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Descarc modelul FaceLandmarker în %s ...", path)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class FaceDetector:
    """Wrapper peste MediaPipe FaceLandmarker în modul VIDEO."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        num_faces: int = 1,
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
        download: bool = True,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = ensure_model(Path(model_path)) if download else Path(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection,
            min_face_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame: Frame) -> Optional[FaceResult]:
        """Rulează modelul pe un cadru. Întoarce None dacă nu e nicio față."""
        # În modul VIDEO timestamp-urile trebuie să fie strict crescătoare.
        ts_ms = max(frame.timestamp_ms, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame.rgb)
        res = self._landmarker.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            return None

        lms = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)
        blend = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        matrix = np.asarray(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None

        return FaceResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            landmarks=lms,
            blendshapes=blend,
            matrix=matrix,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------- test rapid: python -m vigil_vision.perception.face [--show] ----------
if __name__ == "__main__":
    import argparse
    import time

    import cv2

    from ..camera import CV1, Camera

    p = argparse.ArgumentParser(description="Test FaceLandmarker pe cameră")
    p.add_argument("--index", default=0, type=int)
    p.add_argument("--show", action="store_true", help="fereastră cu punctele feței (q = ieșire)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    with Camera(index=args.index) as cam, FaceDetector() as face:
        t0, n = time.monotonic(), 0
        try:
            for frame in cam.reader(CV1):
                res = face.process(frame)
                n += 1
                fps = n / (time.monotonic() - t0)
                if res is None:
                    info = "fără față"
                else:
                    info = (f"{len(res.landmarks)} pct | blinkL {res.blend('eyeBlinkLeft'):.2f} "
                            f"blinkR {res.blend('eyeBlinkRight'):.2f} | jawOpen {res.blend('jawOpen'):.2f}")
                print(f"{fps:5.1f} fps | {info}          ", end="\r")

                if args.show:
                    img = frame.image.copy()  # nu desenăm pe cadrul partajat
                    if res is not None:
                        for x, y in res.points_px.astype(int):
                            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
                    cv2.imshow("vigil face", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
    cv2.destroyAllWindows()
