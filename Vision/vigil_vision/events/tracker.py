"""Detecții per cadru -> evenimente cu durată și încredere. [CV2]

    tracker = EventTracker(cfg)
    events = tracker.update(ctx.detections, ctx.t)   # -> [Event, ...]

Reguli:
    - un eveniment începe la primul cadru activ;
    - pauze mai scurte de gap_sec (mâna alunecă o clipă, YOLO pierde telefonul)
      NU îl întrerup;
    - apare în mesaj doar după min_duration_sec (ex. hand_on_chest >= 2 s),
      ca o atingere scurtă să nu ajungă la LLM;
    - încrederea = media netezită a încrederilor × cât de constant a fost activ,
      crescând puțin cu durata.
Funcționează pentru orice nume, deci CV1 îl poate folosi pentru eyes_closed etc.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..context import Detection
from ..schemas import Event

log = logging.getLogger(__name__)


@dataclass
class _State:
    start: float
    last_active: float
    conf: float
    n_active: int = 1
    n_total: int = 1
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.last_active - self.start


class EventTracker:
    def __init__(self, cfg: dict) -> None:
        e = cfg["events"]
        self.gap = e["gap_sec"]
        self.min_dur = dict(e["min_duration_sec"])
        self._states: dict[str, _State] = {}
        self.ended: list[dict] = []           # ultimele evenimente încheiate (dashboard / log)

    def _min_duration(self, name: str) -> float:
        return self.min_dur.get(name, self.min_dur.get("default", 1.0))

    def update(self, detections: dict[str, Detection], t: float) -> list[Event]:
        for name, det in detections.items():
            st = self._states.get(name)
            if det.active:
                if st is None:
                    self._states[name] = _State(start=t, last_active=t, conf=det.confidence,
                                                attrs=dict(det.attrs))
                    continue
                st.last_active = t
                st.conf = 0.8 * st.conf + 0.2 * det.confidence
                st.n_active += 1
                st.attrs.update(det.attrs)
            if st is not None:
                st.n_total += 1

        # închide evenimentele inactive de prea mult timp
        for name in list(self._states):
            st = self._states[name]
            if t - st.last_active > self.gap:
                del self._states[name]
                if st.duration >= self._min_duration(name):
                    info = {"event": name, "duration_sec": round(st.duration, 1), "ended_at": t}
                    self.ended = (self.ended + [info])[-20:]
                    log.info("Eveniment încheiat: %s (%.1f s)", name, st.duration)

        return self.active_events()

    def active_events(self) -> list[Event]:
        out = []
        for name, st in self._states.items():
            md = self._min_duration(name)
            if st.duration < md:
                continue
            steady = (st.n_active / max(st.n_total, 1)) ** 0.5
            ramp = 0.7 + 0.3 * min(1.0, st.duration / (2 * md))
            conf = max(0.0, min(1.0, st.conf * steady * ramp))
            out.append(Event(name, st.duration, conf, dict(st.attrs)))
        out.sort(key=lambda e: -e.duration_sec)
        return out

    def reset(self) -> None:
        self._states.clear()
