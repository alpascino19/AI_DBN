"""Serviciul ML1, independent de transport (MQTT, HTTP de la serverul Spring Boot, teste).

    svc = Ml1Service(llm=OllamaClient())       # sau llm=None: doar reguli
    svc.ingest_vision(msg)                       # un cadru vigil/vision (orice formă acceptată)
    svc.ingest_response(resp)                    # răspunsul șoferului (de la ML2)
    svc.decisions_since(seq)                     # deciziile noi
    svc.subscribe() / unsubscribe(q)             # flux de decizii (pentru SSE)
    svc.status()                                 # scor, încredere, calibrare, statistici

Realitatea unui flux live (cameră reală + server) tratată aici, nu în fiecare transport:
  * timestamp lipsă -> ora serverului; ms / ISO-8601 -> secunde (cv_adapter.normalize_ts);
  * mesaje vechi (ts mai mic decât ultimul cu > STALE_SEC) sau duplicate -> ignorate și numărate;
  * pauză lungă în flux (> GAP_RESET_SEC, ex. camera repornită) -> episodul curent se închide curat;
  * thread-safe: un singur lacăt în jurul motorului; deciziile LLM asincrone ajung tot aici.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

from .cv_adapter import _snake_keys, normalize_ts, unwrap
from .engine import DecisionEngine
from . import profile

STALE_SEC = 0.5        # un cadru mai vechi decât ultimul cu atât e ignorat (rețea / reordonare)
GAP_RESET_SEC = 30.0   # pauză mai lungă în flux: episod nou (calibrarea se păstrează)
MAX_DECISIONS = 500


class Ml1Service:
    def __init__(self, llm=None, model_name: str | None = None, engine: DecisionEngine | None = None):
        self.llm = llm
        self.model_name = model_name or getattr(llm, "model", None) or "reguli"
        # Două lacăte, mereu în aceeași ordine (_lock -> motor); deciziile (care pot veni din thread-ul
        # LLM, cu lacătul motorului luat) folosesc doar _dlock -> fără blocaj circular.
        self._lock = threading.RLock()
        self._dlock = threading.Lock()
        self._listeners: list = []
        self._subs: list[queue.Queue] = []
        self._decisions: deque = deque(maxlen=MAX_DECISIONS)
        self._seq = 0
        self.counters = {"vision": 0, "responses": 0, "stale": 0, "duplicate": 0, "invalid": 0, "gap_resets": 0}
        self._new_engine(engine)

    # ------------------------------------------------------------- motor
    def _new_engine(self, engine: DecisionEngine | None = None) -> None:
        self.engine = engine or DecisionEngine(llm_fn=self.llm, sync=self.llm is None)
        self.engine.publish_fn = self._on_decision
        self._last_ts: float | None = None
        self._last_wall = time.time()

    def reset(self) -> dict:
        """Călătorie nouă / alt șofer: motor nou, calibrare de la zero."""
        with self._lock:
            self._new_engine()
        return {"ok": True}

    def add_listener(self, fn) -> None:
        """fn(decision) la fiecare decizie nouă (ex. puntea MQTT)."""
        self._listeners.append(fn)

    def _on_decision(self, d: dict) -> None:
        with self._dlock:
            self._seq += 1
            item = {"seq": self._seq, **_public(d)}
            self._decisions.append(item)
            subs = list(self._subs)
        for fn in list(self._listeners):
            try:
                fn(item)
            except Exception:  # un consumator stricat nu oprește motorul
                pass
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    # ------------------------------------------------------------ intrări
    def ingest_vision(self, msg: Any) -> dict:
        msg = _snake_keys(unwrap(msg))
        if not isinstance(msg, dict):
            self.counters["invalid"] += 1
            return {"accepted": False, "reason": "mesaj invalid"}
        ts = normalize_ts(msg.get("ts")) if "ts" in msg else None
        with self._lock:
            if ts is None:
                ts = time.time()
            if self._last_ts is not None:
                if ts == self._last_ts:
                    self.counters["duplicate"] += 1
                    return {"accepted": False, "reason": "duplicat"}
                if ts < self._last_ts - STALE_SEC:
                    self.counters["stale"] += 1
                    return {"accepted": False, "reason": "mesaj vechi"}
                if ts - self._last_ts > GAP_RESET_SEC:
                    self.counters["gap_resets"] += 1
                    self.engine._reset_episode()
            self._last_ts = max(ts, self._last_ts or ts)
            self._last_wall = time.time()
            self.counters["vision"] += 1
            with self._dlock:
                before = self._seq
            self.engine.on_vision({**msg, "ts": ts})
        with self._dlock:
            new = [d for d in self._decisions if d["seq"] > before]
        return {"accepted": True, "ts": ts, "decisions": new}

    def ingest_many(self, msgs: list) -> dict:
        out = [self.ingest_vision(m) for m in msgs]
        return {"accepted": sum(1 for r in out if r.get("accepted")), "received": len(out),
                "decisions": [d for r in out for d in r.get("decisions", [])]}

    def ingest_response(self, resp: Any) -> dict:
        resp = _snake_keys(unwrap(resp))
        if not isinstance(resp, dict) or "answered" not in resp:
            self.counters["invalid"] += 1
            return {"accepted": False, "reason": "lipsește câmpul answered"}
        ts = normalize_ts(resp.get("ts")) if resp.get("ts") is not None else None
        with self._lock:
            self.counters["responses"] += 1
            self.engine.on_driver_response(resp, ts=ts if ts is not None else self._last_ts)
        return {"accepted": True}

    # ------------------------------------------------------------ ieșiri
    def decisions_since(self, seq: int = 0, limit: int = 100) -> list[dict]:
        with self._dlock:
            return [d for d in self._decisions if d["seq"] > seq][:limit]

    def latest_decision(self) -> dict | None:
        with self._dlock:
            return self._decisions[-1] if self._decisions else None

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._dlock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._dlock:
            if q in self._subs:
                self._subs.remove(q)

    def status(self) -> dict:
        with self._dlock:
            last = self._decisions[-1] if self._decisions else None
            seq = self._seq
        with self._lock:
            r = self.engine.last_result
            return {
                "ok": True, "model": self.model_name, "profile": profile.ACTIVE,
                "last_ts": self._last_ts, "seconds_since_last_frame": round(time.time() - self._last_wall, 2),
                "a": None if r is None else round(r.a, 3),
                "confidence": None if r is None else round(r.confidence, 3),
                "critical": None if r is None else r.critical,
                "camera_on": None if r is None else r.camera_on,
                "severity": last["severity"] if last else 0,
                "calibration": self.engine.baseline.to_dict(),
                "engine": dict(self.engine.stats), "counters": dict(self.counters),
                "decisions": seq,
            }


def _public(d: dict) -> dict:
    keep = ("ts", "a", "severity", "actions", "reason", "source", "latency_ms", "safety_notes", "fallback_reason")
    return {k: d[k] for k in keep if k in d}
