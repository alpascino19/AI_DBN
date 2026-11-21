"""Bucla: cadru -> perception -> analizori -> tracker -> publish. [CV2]

Structura (un singur proces, o singură cameră):

    Camera ──► reader("cv1") ──► FaceWorker (thread)  ──► ultimul FaceOutput ─┐
           └─► reader("cv2") ──► Pipeline.run (bucla principală)              │
                                   BodyPerception (Pose + Hands) + PoseSmoother │
                                   ObjectDetector (YOLO, la N cadre)           │
                                   + FaceOutput de la CV1  ◄───────────────────┘
                                   analizori: geometry → chest → breathing → phone → slump
                                   EventTracker → VisionMessage → Publisher (vigil/vision)

Hook pentru CV1: modulul `vigil_vision.face` (face/__init__.py) trebuie să ofere
    def create(cfg) -> obiect cu .process(frame: Frame) -> context.FaceOutput
FaceOutput.signals: ear, perclos, blink_rate, gaze_on_road, head_drop, jaw_open
FaceOutput.detections: ex. {"eyes_closed": Detection(...)} -> trec prin același tracker.
Până există hook-ul, pipeline-ul merge doar cu partea de corp.
"""
from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, Callable, Optional

from .body.breathing import BreathingAnalyzer
from .body.chest import ChestAnalyzer
from .body.geometry import BodyGeometry
from .body.phone import PhoneAnalyzer
from .body.slump import SlumpAnalyzer
from .camera import CV1, CV2, Camera, Frame
from .context import FaceOutput, FrameContext
from .events.tracker import EventTracker
from .publisher import Publisher
from .schemas import VisionMessage
from .smoothing import PoseSmoother

log = logging.getLogger(__name__)

FACE_MAX_AGE = 0.5      # s; un FaceOutput mai vechi e ignorat
CAMERA_LOST = 2.0       # s fără cadre -> camera_on = False


def _try_build(factory: Callable[[], Any], what: str) -> Any:
    try:
        return factory()
    except ImportError as e:
        log.warning("%s dezactivat: lipsește pachetul (%s)", what, e.name or e)
    except Exception as e:                                            # noqa: BLE001
        log.warning("%s dezactivat: %s", what, e)
    return None


def _head_drop(face_value: Optional[bool], ctx: FrameContext) -> Optional[bool]:
    slump = ctx.detections.get("head_on_wheel")
    body = bool(slump and slump.active)
    if face_value is None:
        return body if ctx.pose is not None or body else None
    return bool(face_value) or body


class FaceWorker:
    """Rulează stiva CV1 pe reader-ul 'cv1', în thread propriu."""

    def __init__(self, stack: Any, camera: Camera) -> None:
        self._stack = stack
        self._reader = camera.reader(CV1)
        self._lock = threading.Lock()
        self._latest: Optional[FaceOutput] = None
        self._thread = threading.Thread(target=self._loop, name="cv1-face", daemon=True)

    @classmethod
    def from_hook(cls, cfg: dict, camera: Camera) -> Optional["FaceWorker"]:
        try:
            mod = importlib.import_module("vigil_vision.face")
            create = getattr(mod, "create")
        except (ImportError, AttributeError):
            log.info("CV1: vigil_vision.face.create(cfg) încă nu există -> fără semnale de față")
            return None
        stack = _try_build(lambda: create(cfg), "CV1 (față)")
        return cls(stack, camera) if stack is not None else None

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        for frame in self._reader:
            try:
                out = self._stack.process(frame)
            except Exception:                                         # noqa: BLE001
                log.exception("CV1 process() a eșuat")
                continue
            if out is not None:
                out.timestamp = out.timestamp or frame.timestamp
                with self._lock:
                    self._latest = out

    def latest(self, now: float) -> Optional[FaceOutput]:
        with self._lock:
            out = self._latest
        if out is None or now - out.timestamp > FACE_MAX_AGE:
            return None
        return out

    @property
    def fps(self) -> float:
        return self._reader.fps


class Pipeline:
    def __init__(self, cfg: dict, publisher: Optional[Publisher] = None,
                 show: bool = False, camera: Optional[Camera] = None) -> None:
        self.cfg = cfg
        self.show = show
        self.camera = camera or Camera(**cfg["camera"])
        self.publisher = publisher or Publisher(cfg)

        from .perception.body import BodyPerception
        from .perception.objects import ObjectDetector
        self.body = _try_build(lambda: BodyPerception(cfg), "MediaPipe Pose/Hands")
        self.objects = _try_build(lambda: ObjectDetector(cfg), "YOLO (telefon)")

        # ordinea contează: breathing folosește zona pieptului (flow) și pune
        # "breathing" pe evenimentul hand_on_chest
        breathing = [BreathingAnalyzer(cfg)] if cfg.get("breathing", {}).get("enabled", False) else []
        self.analyzers = [BodyGeometry(cfg), ChestAnalyzer(cfg), *breathing, PhoneAnalyzer(cfg),
                          SlumpAnalyzer(cfg)]
        self.tracker = EventTracker(cfg)
        sm = cfg.get("smoothing", {})
        self.smoother = PoseSmoother(sm.get("min_cutoff", 1.5), sm.get("beta", 0.5))
        self.face: Optional[FaceWorker] = None
        self._period = 1.0 / max(1, cfg["publish_hz"])
        self._stop = threading.Event()

    # ---------- un cadru ----------
    def process(self, frame: Frame) -> FrameContext:
        ctx = FrameContext(frame=frame, t=frame.timestamp)
        if self.body is not None:
            pose, ctx.hands = self.body.process(frame)
            ctx.pose = self.smoother(pose, frame.timestamp)   # fără tremurat
        if self.objects is not None:
            ctx.objects = self.objects.process(frame)
        if self.face is not None:
            ctx.face = self.face.latest(frame.timestamp)
            if ctx.face is not None:
                ctx.detections.update(ctx.face.detections)
        for a in self.analyzers:
            a.update(ctx)
        ctx.events = self.tracker.update(ctx.detections, ctx.t)
        return ctx

    def build_message(self, ctx: Optional[FrameContext], fps: float = 0.0) -> VisionMessage:
        if ctx is None:                                    # camera oprită / pierdută
            return VisionMessage(ts=time.time(), camera_on=False,
                                 events=self.tracker.active_events())
        fs = ctx.face.signals if ctx.face else {}
        body = []
        if ctx.pose:
            body = [[round(l.x, 3), round(l.y, 3)] for l in ctx.pose[:25]]
        return VisionMessage(
            ear=fs.get("ear"), perclos=fs.get("perclos"), blink_rate=fs.get("blink_rate"),
            gaze_on_road=fs.get("gaze_on_road"),
            # head_drop: de la CV1 (fața) SAU de la CV2 (corpul prăbușit pe volan)
            head_drop=_head_drop(fs.get("head_drop"), ctx),
            phone=ctx.signals.get("phone", 0.0),
            breathing=ctx.signals.get("breathing"),
            posture=ctx.signals.get("slump"),
            events=ctx.events,
            landmarks=ctx.face.landmarks if ctx.face else [],
            body_landmarks=body,
            ts=time.time(), camera_on=True,
            face_detected=bool(ctx.face and ctx.face.detected),
            body_detected=ctx.pose is not None,
            fps=fps,
        )

    # ---------- bucla principală ----------
    def run(self) -> None:
        self.camera.start()
        self.face = FaceWorker.from_hook(self.cfg, self.camera)
        if self.face is not None:
            self.face.start()
        reader = self.camera.reader(CV2)
        next_pub, last_frame_t = 0.0, time.monotonic()
        log.info("Pipeline pornit (Ctrl+C pentru oprire%s)", ", q în fereastră" if self.show else "")
        try:
            while not self._stop.is_set():
                frame = reader.read(timeout=0.5)
                now = time.monotonic()
                if frame is None:
                    if now - last_frame_t > CAMERA_LOST and now >= next_pub:
                        self.publisher.publish(self.build_message(None))
                        next_pub = now + self._period
                    continue
                last_frame_t = now
                ctx = self.process(frame)
                if now >= next_pub:
                    self.publisher.publish(self.build_message(ctx, reader.fps))
                    next_pub = now + self._period
                if self.show and not self._show(ctx, reader.fps):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _show(self, ctx: FrameContext, fps: float) -> bool:
        import cv2
        from .debug_view import draw
        cv2.imshow("vigil - CV2 debug", draw(ctx, fps))
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.camera.stop()
        for part in (self.body,):
            if part is not None and hasattr(part, "close"):
                part.close()
        self.publisher.close()
        if self.show:
            import cv2
            cv2.destroyAllWindows()
