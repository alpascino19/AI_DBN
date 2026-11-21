"""Căscat: gura larg deschisă, susținut. [CV1]

Două semnale pentru "gura deschisă":
    MAR  (mouth aspect ratio) = distanța buze sus-jos / lățimea gurii
    jawOpen (blendshape MediaPipe, 0..1)
Căscat = gura larg deschisă cel puțin `min_sec` (vorbitul o deschide doar scurt)
și cel mult `max_sec` (altfel e altceva: respirație grea, gura căscată de leșin).

Utilizare:
    yawn = YawnAnalyzer()
    st = yawn.update(face_result, frame.timestamp)
    st.yawning, st.yawn_count, st.yawns_recent
    yawn.detections()   # {"yawning": (True, 0.8), "frequent_yawning": (False, 0.0)}
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..perception.face import FaceResult, Idx


@dataclass
class YawnConfig:
    mar_open: float = 0.55         # MAR peste prag = gura larg deschisă
    jaw_open: float = 0.40     # sau jawOpen peste prag
    close_ratio: float = 0.7       # se închide sub prag * ratio (histerezis)
    min_sec: float = 1.5           # sub = vorbit / râs
    max_sec: float = 8.0           # peste = nu e căscat
    gap_sec: float = 0.4           # o scurtă închidere (mâna la gură) nu rupe căscatul
    window_sec: float = 300.0      # căscate numărate în ultimele 5 minute
    frequent_count: int = 3        # atâtea în fereastră = oboseală


@dataclass
class YawnState:
    face_visible: bool = False
    mar: Optional[float] = None
    jaw_open: Optional[float] = None
    mouth_open: bool = False
    open_sec: float = 0.0          # de cât timp e gura larg deschisă acum
    yawning: bool = False          # căscat în desfășurare (open_sec >= min_sec)
    yawn_count: int = 0            # total de la pornire
    yawns_recent: int = 0          # în ultimele window_sec
    last_yawn_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR din punctele în pixeli: |buza sus - buza jos| / |colț stâng - colț drept|."""
    width = np.linalg.norm(pts[Idx.MOUTH_LEFT] - pts[Idx.MOUTH_RIGHT])
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(pts[Idx.LIP_TOP] - pts[Idx.LIP_BOTTOM]) / width)


class YawnAnalyzer:
    def __init__(self, config: Optional[YawnConfig] = None) -> None:
        self.cfg = config or YawnConfig()
        self.state = YawnState()
        self._open_since: Optional[float] = None
        self._closed_at: Optional[float] = None   # pentru gap_sec
        self._yawns: deque[float] = deque()

    def update(self, face: Optional[FaceResult], t: float) -> YawnState:
        s = self.state
        if face is None:
            s.face_visible = False
            s.mar = s.jaw_open = None
            # mâna la gură poate ascunde fața: nu închidem imediat căscatul
            return self._step(None, None, t)
        s.face_visible = True
        return self._step(mouth_aspect_ratio(face.points_px), face.blend("jawOpen", 0.0), t)

    def update_values(self, mar: float, jaw: float, t: float) -> YawnState:
        """Ca update(), dar direct cu valori (pentru teste / date false)."""
        self.state.face_visible = True
        return self._step(mar, jaw, t)

    def detections(self) -> dict[str, tuple[bool, float]]:
        s, c = self.state, self.cfg
        conf = min(1.0, 0.6 + 0.1 * (s.open_sec - c.min_sec)) if s.yawning else 0.0
        freq = s.yawns_recent >= c.frequent_count
        return {
            "yawning": (s.yawning, round(conf, 2)),
            "frequent_yawning": (freq, min(1.0, 0.5 + 0.15 * s.yawns_recent) if freq else 0.0),
        }

    # ---------- intern ----------
    def _step(self, mar: Optional[float], jaw: Optional[float], t: float) -> YawnState:
        s, c = self.state, self.cfg
        if mar is not None:
            s.mar, s.jaw_open = round(mar, 3), round(jaw or 0.0, 3)
            wide = mar > c.mar_open or (jaw or 0) > c.jaw_open
            closed = mar < c.mar_open * c.close_ratio and (jaw or 0) < c.jaw_open * c.close_ratio
        else:
            wide, closed = False, False   # fără față: nici deschis, nici închis sigur

        if wide:
            self._closed_at = None
            if self._open_since is None:
                self._open_since = t
            s.mouth_open = True
        elif s.mouth_open:
            # închis clar sau fără față prea mult -> se termină episodul
            if self._closed_at is None:
                self._closed_at = t
            if (closed or mar is None) and t - self._closed_at >= c.gap_sec:
                self._finish(self._closed_at)

        s.open_sec = 0.0 if self._open_since is None else round(t - self._open_since, 2)
        s.yawning = s.mouth_open and c.min_sec <= s.open_sec <= c.max_sec

        while self._yawns and self._yawns[0] < t - c.window_sec:
            self._yawns.popleft()
        s.yawns_recent = len(self._yawns)
        return s

    def _finish(self, t_end: float) -> None:
        s, c = self.state, self.cfg
        dur = t_end - (self._open_since if self._open_since is not None else t_end)
        if c.min_sec <= dur <= c.max_sec:
            s.yawn_count += 1
            s.last_yawn_sec = round(dur, 2)
            self._yawns.append(t_end)
        s.mouth_open = False
        s.yawning = False
        self._open_since = None
        self._closed_at = None
