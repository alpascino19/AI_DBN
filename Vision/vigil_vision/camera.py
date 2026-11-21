"""Thread de captură; un singur proces deschide camera. [CV2]

Structura:
    Camera          - deschide camera o singură dată, citește într-un thread
                      separat și păstrează DOAR ultimul cadru (fără coadă,
                      deci fără latență acumulată).
    Frame           - un cadru + metadate (id, timestamp, dimensiuni).
    FrameReader     - "abonamentul" unui consumator (CV1 sau CV2). Fiecare
                      cititor primește fiecare cadru nou cel mult o dată,
                      independent de ceilalți: dacă CV2 (YOLO) e lent, CV1
                      (Face Mesh) nu e încetinit.

Utilizare:
    with Camera(index=0) as cam:
        cv1 = cam.reader("cv1")      # fața și ochii
        cv2_ = cam.reader("cv2")     # corp, mâini, obiecte
        frame = cv1.read(timeout=1.0)
        if frame is not None:
            frame.image  # BGR (OpenCV)
            frame.rgb    # RGB (MediaPipe)

Privacy: cadrele stau doar în memorie; nimic nu e salvat sau trimis.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Numele consumatorilor cunoscuți (doar convenție, poți crea și alții).
CV1 = "cv1"  # fața și ochii: perception/face.py, face/*
CV2 = "cv2"  # corp, mâini, obiecte: perception/body.py, perception/objects.py, body/*


@dataclass(frozen=True)
class Frame:
    """Un cadru capturat. `image` e BGR și NU trebuie modificat in-place
    (e partajat între CV1 și CV2) — fă `.copy()` dacă desenezi pe el."""

    image: np.ndarray
    frame_id: int
    timestamp: float  # time.monotonic() la captură
    width: int
    height: int
    _rgb_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def rgb(self) -> np.ndarray:
        """Versiunea RGB pentru MediaPipe, calculată o singură dată per cadru."""
        if "rgb" not in self._rgb_cache:
            self._rgb_cache["rgb"] = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self._rgb_cache["rgb"]

    @property
    def timestamp_ms(self) -> int:
        """Timestamp în ms, util pentru MediaPipe în modul VIDEO/LIVE_STREAM."""
        return int(self.timestamp * 1000)


class FrameReader:
    """Cititorul unui consumator. Ține minte ultimul cadru văzut."""

    def __init__(self, camera: "Camera", name: str) -> None:
        self._camera = camera
        self.name = name
        self._last_id = -1
        self._count = 0
        self._t0 = time.monotonic()

    def read(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Așteaptă un cadru mai nou decât ultimul citit de acest consumator.
        Întoarce None la timeout sau dacă camera s-a oprit."""
        frame = self._camera._wait_newer(self._last_id, timeout)
        if frame is not None:
            self._last_id = frame.frame_id
            self._count += 1
        return frame

    @property
    def fps(self) -> float:
        """Câte cadre pe secundă procesează efectiv acest consumator."""
        dt = time.monotonic() - self._t0
        return self._count / dt if dt > 0 else 0.0

    def __iter__(self):
        """for frame in reader: ...  (se oprește când se oprește camera)."""
        while self._camera.running:
            frame = self.read(timeout=1.0)
            if frame is not None:
                yield frame


class Camera:
    """Sursa unică de cadre. Un singur thread citește din cameră."""

    def __init__(
        self,
        index: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._next_id = 0
        self._readers: dict[str, FrameReader] = {}
        self.capture_fps = 0.0

    # ---------- ciclul de viață ----------
    def start(self) -> "Camera":
        if self.running:
            return self
        self._stop.clear()
        self._open()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        log.info("Camera %s pornită (%dx%d @ %d fps)", self.index, self.width, self.height, self.fps)
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()
        log.info("Camera oprită")

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    # ---------- API pentru consumatori ----------
    def reader(self, name: str) -> FrameReader:
        """Creează (sau întoarce) cititorul pentru un consumator, ex. 'cv1', 'cv2'."""
        if name not in self._readers:
            self._readers[name] = FrameReader(self, name)
        return self._readers[name]

    def latest(self) -> Optional[Frame]:
        """Ultimul cadru, fără așteptare (ex. pentru dashboard)."""
        with self._cond:
            return self._latest

    # ---------- intern ----------
    def _open(self) -> None:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" and isinstance(self.index, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # cât mai puțină latență
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Nu pot deschide camera {self.index!r}")
        self._cap = cap

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        t_last = time.monotonic()
        while not self._stop.is_set():
            ok, img = (self._cap.read() if self._cap is not None else (False, None))
            if not ok or img is None:
                log.warning("Cadru pierdut; reîncerc conectarea la cameră...")
                self._release()
                if self._stop.wait(self.reconnect_delay):
                    break
                try:
                    self._open()
                except RuntimeError as e:
                    log.warning("%s", e)
                continue

            if self.mirror:
                img = cv2.flip(img, 1)
            now = time.monotonic()
            h, w = img.shape[:2]
            frame = Frame(image=img, frame_id=self._next_id, timestamp=now, width=w, height=h)
            self._next_id += 1

            dt = now - t_last
            t_last = now
            if dt > 0:
                self.capture_fps = 0.9 * self.capture_fps + 0.1 * (1.0 / dt)

            with self._cond:
                self._latest = frame
                self._cond.notify_all()

    def _wait_newer(self, last_id: int, timeout: Optional[float]) -> Optional[Frame]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.frame_id > last_id:
                    return self._latest
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
        return None


# ---------- test rapid: python -m vigil_vision.camera [--show] ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test cameră: doi consumatori (CV1, CV2) pe aceeași cameră")
    p.add_argument("--index", default=0, type=int)
    p.add_argument("--show", action="store_true", help="afișează fereastra de preview (doar local)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    def consumer(reader: FrameReader, work_sec: float) -> None:
        for _frame in reader:
            time.sleep(work_sec)  # simulează procesarea (CV2 cu YOLO e mai lent)

    with Camera(index=args.index) as cam:
        threading.Thread(target=consumer, args=(cam.reader(CV1), 0.01), daemon=True).start()
        threading.Thread(target=consumer, args=(cam.reader(CV2), 0.08), daemon=True).start()
        try:
            while True:
                f = cam.latest()
                if args.show and f is not None:
                    cv2.imshow("vigil camera", f.image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(1.0)
                print(f"camera {cam.capture_fps:5.1f} fps | "
                      f"cv1 {cam.reader(CV1).fps:5.1f} fps | cv2 {cam.reader(CV2).fps:5.1f} fps", end="\r")
        except KeyboardInterrupt:
            pass
    cv2.destroyAllWindows()
