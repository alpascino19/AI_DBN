"""Motorul de decizie ML 1: vigil/vision (+ driver_response) -> vigil/decision.

Optimizări pentru rulare pe controllerul din mașină:
  * LLM-ul e apelat DOAR la schimbarea situației (semnătură: severitate din reguli,
    evenimente active, treaptă de scor, starea răspunsului), max 1/s, + heartbeat 15 s.
    La 5 Hz pe 30 s de somnolență: ~10 apeluri în loc de 150.
  * Calea rapidă: severitatea 3 e publicată IMEDIAT din reguli; LLM-ul doar nuanțează
    după. Siguranța nu așteaptă niciodată după model.
  * Răspunsurile LLM întârziate (situația s-a schimbat între timp) sunt aruncate.
  * 112 o singură dată; „Sunteți bine?” nu se repetă cât așteptăm răspunsul;
    tăcere > ask_timeout = „fără răspuns” (chiar dacă STT-ul a căzut).
  * Memorie constantă: istoricul scorului e eșantionat la 1 Hz.

    python -m ml1.engine tests/data/collapse.jsonl            # doar reguli
    python -m ml1.engine tests/data/collapse.jsonl --ollama   # cu LLM local
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .actions import PHRASES
from .calibration import DriverBaseline
from .cv_adapter import normalize_vision
from .fusion import AttentionFusion
from .reports import build_trip_report
from .rules import answer_status, fallback_decision, rule_severity, wants_112
from .validate import LLM_TIMEOUT_SEC, decide


class _NoBaseline:
    """Calibrare dezactivată (pentru comparații în evaluare)."""
    ready = False

    def update(self, msg, ts):
        return False

    def to_dict(self):
        return {"ready": False, "source": "off"}


class DecisionEngine:
    def __init__(self, llm_fn: Callable[[dict], Any] | None = None,
                 publish: Callable[[dict], None] | None = None, sync: bool = True,
                 min_gap: float = 1.0, heartbeat: float = 15.0, ask_timeout: float = 8.0,
                 llm_timeout: float = LLM_TIMEOUT_SEC, baseline_ear: float | None = None,
                 calibrate: bool = True):
        self.fusion = AttentionFusion(baseline_ear=baseline_ear)
        self.llm_fn = llm_fn
        self.publish_fn = publish or (lambda d: None)
        self.sync = sync
        self.min_gap, self.heartbeat, self.ask_timeout, self.llm_timeout = min_gap, heartbeat, ask_timeout, llm_timeout
        self._pool = None if sync else ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml1-llm")
        self._lock = threading.RLock()

        self._busy = False
        self._last_sig: tuple | None = None
        self._last_call_ts = -math.inf
        self._last_pub: dict = {"severity": 0, "ts": -math.inf, "tools": frozenset()}
        self._latest_rule_sev = 0
        self._asked_ts: float | None = None
        self._asked_q: str | None = None
        self._calm_since: float | None = None
        self._lower_since: float | None = None
        self._resp: tuple[float, dict] | None = None
        self._called_112 = False
        self._last_fast_reason: str | None = None
        self.baseline = DriverBaseline() if calibrate else _NoBaseline()
        self._timeline: dict[str, dict] = {}
        self._a_hist: deque = deque()
        self._last_frame_ts: float | None = None

        self.last_result = None
        self.decisions: list[dict] = []
        self.a_samples: list[tuple[float, float]] = []
        self.event_totals: dict[str, float] = defaultdict(float)
        self.stats = {"frames": 0, "llm_calls": 0, "llm_discarded": 0, "fast_path": 0}

    # ------------------------------------------------------------ intrări --
    def on_driver_response(self, resp: dict, ts: float | None = None) -> None:
        with self._lock:
            t = ts if ts is not None else (self._last_frame_ts or 0.0)
            self._resp = (t, resp)
            self.fusion.update_driver_response(resp, ts=t)

    def on_vision(self, msg: dict) -> list[dict]:
        """Procesează un cadru. Întoarce deciziile publicate acum (în modul sync include și LLM-ul)."""
        with self._lock:
            out: list[dict] = []
            msg = normalize_vision(msg)  # forma reală CV (phone_use, head_on_wheel, ...) -> forma ML1
            r = self.fusion.update(msg)
            self.last_result = r  # pentru dashboard / debugging
            ts = r.ts
            dt = 0.0 if self._last_frame_ts is None else max(0.0, ts - self._last_frame_ts)
            self._last_frame_ts = ts
            self.stats["frames"] += 1
            self._track(msg, ts, dt, r.a)
            if self.baseline.update(msg, ts):  # calibrarea tocmai s-a încheiat
                self.fusion.baseline_ear = self.baseline.ear
                self.fusion.baseline_blink = self.baseline.blink
                self.fusion.perclos_offset = self.baseline.perclos_offset

            ctx = self._context(r.to_dict(), msg, ts)
            rule_sev, rule_reason = rule_severity(ctx)
            self._latest_rule_sev = rule_sev
            if rule_sev < self._last_pub["severity"]:
                self._lower_since = self._lower_since if self._lower_since is not None else ts
            else:
                self._lower_since = None

            # 1. cale rapidă: severitate 3 / 112 nu așteaptă LLM-ul. Se declanșează și când:
            #    * apare un semn grav NOU în același episod (ex. oboseală -> prăbușire),
            #    * episodul e la severitate 3 dar „Sunteți bine?” nu a fost pus încă (ex. întrebarea
            #      a fost blocată cât se aștepta răspunsul la „Doriți o pauză?”). Fără asta,
            #      un șofer prăbușit după o alertă de oboseală nu ajungea niciodată la 112.
            if rule_sev == 3:
                new_sign = rule_reason != self._last_fast_reason
                need_check = (not self._called_112 and not self._waiting_answer(ts)
                              and self._asked_q != PHRASES["ask_ok"])
                if (self._last_pub["severity"] < 3 or new_sign or need_check
                        or (wants_112(ctx) and not self._called_112)):
                    self.stats["fast_path"] += 1
                    self._last_fast_reason = rule_reason
                    out += self._publish(fallback_decision(ctx, "cale rapidă (reguli, severitate 3)"), ts, dedupe=True)

            # 2. revenire la normal (histerezis: 5 s de calm continuu, fără pâlpâire)
            calm = rule_sev == 0 and r.a >= 0.85
            self._calm_since = (self._calm_since if self._calm_since is not None else ts) if calm else None
            if calm and ts - self._calm_since >= 5.0 and self._last_pub["severity"] > 0:
                out += self._publish(fallback_decision(ctx, "revenire la normal"), ts)
                self._reset_episode()

            # 3. LLM doar la schimbare
            active = frozenset(e["event"] for e in self._active_events(msg))
            sig = (rule_sev, active, round(r.a * 4), answer_status(ctx["driver_response"]))
            interesting = rule_sev > 0 or r.a < 0.85
            changed = sig != self._last_sig
            due = interesting and (changed or ts - self._last_call_ts >= self.heartbeat)
            if due and not self._busy and ts - self._last_call_ts >= self.min_gap:
                self._last_sig, self._last_call_ts = sig, ts
                if self.llm_fn is None:
                    up = rule_sev > self._last_pub["severity"]
                    down = 0 < rule_sev < self._last_pub["severity"] and self._can_downgrade(ts)
                    if up or down:
                        out += self._publish(fallback_decision(ctx, "fără LLM"), ts)
                else:
                    self._busy = True
                    self.stats["llm_calls"] += 1
                    if self._pool is None:
                        out += self._llm_job(ctx, ts)
                    else:
                        self._pool.submit(self._llm_job, ctx, ts)
            return out

    # ------------------------------------------------------------ interne --
    @staticmethod
    def _active_events(msg: Any) -> list[dict]:
        evs = msg.get("events") if isinstance(msg, dict) else None
        return [e for e in evs if isinstance(e, dict) and isinstance(e.get("event"), str)
                and isinstance(e.get("confidence"), (int, float)) and e["confidence"] >= 0.5] if isinstance(evs, list) else []

    def _track(self, msg: dict, ts: float, dt: float, a: float) -> None:
        for e in self._active_events(msg):
            tl = self._timeline.setdefault(e["event"], {"duration_sec": 0.0, "confidence": 0.0, "breathing": None})
            tl["duration_sec"] = max(tl["duration_sec"], float(e.get("duration_sec") or 0))
            tl["confidence"] = max(tl["confidence"], float(e["confidence"]))
            tl["breathing"] = tl["breathing"] or e.get("breathing")
            self.event_totals[e["event"]] += dt
        self._a_hist.append((ts, a))
        while self._a_hist and ts - self._a_hist[0][0] > 10.0:
            self._a_hist.popleft()
        if not self.a_samples or ts - self.a_samples[-1][0] >= 1.0:
            self.a_samples.append((ts, round(a, 3)))

    def _context(self, fusion: dict, vision: dict, ts: float) -> dict:
        resp = None
        if self._asked_ts is not None:
            if self._resp and self._resp[0] >= self._asked_ts:
                resp = {**self._resp[1], "question": self._asked_q}
            elif ts - self._asked_ts >= self.ask_timeout:
                resp = {"answered": False, "latency_sec": None, "text": None, "timeout": True, "question": self._asked_q}
        vis = {k: v for k, v in vision.items() if k not in ("landmarks", "body_landmarks")}
        if self.baseline.ready and vis.get("perclos") is not None:
            vis["perclos_cal"] = self.baseline.calibrated_perclos(vis["perclos"])
        return {
            "fusion": fusion,
            "vision": vis,
            "driver_response": resp,
            "timeline": {k: dict(v) for k, v in self._timeline.items()},
            "a_trend": self._a_hist[0][1] if self._a_hist and ts - self._a_hist[0][0] >= 5 else None,
        }

    def _llm_job(self, ctx: dict, frame_ts: float) -> list[dict]:
        dec = decide(ctx, self.llm_fn, self.llm_timeout)
        with self._lock:
            self._busy = False
            latest = self._latest_rule_sev
            floor = 3 if latest == 3 else max(0, latest - 1)
            stale = frame_ts < self._last_pub["ts"] or dec["severity"] < floor
            if dec["severity"] < self._last_pub["severity"] and not self._can_downgrade(frame_ts):
                stale = True
            if stale:
                self.stats["llm_discarded"] += 1
                return []
            return self._publish(dec, frame_ts, dedupe=True)

    def _waiting_answer(self, ts: float) -> bool:
        return (self._asked_ts is not None and not (self._resp and self._resp[0] >= self._asked_ts)
                and ts - self._asked_ts < self.ask_timeout)

    def _can_downgrade(self, ts: float) -> bool:
        """Coborâm severitatea doar după 5 s la nivel mai mic și niciodată în timpul unei întrebări."""
        return (not self._waiting_answer(ts) and self._lower_since is not None
                and ts - self._lower_since >= 5.0)

    def _publish(self, dec: dict, ts: float, dedupe: bool = False) -> list[dict]:
        acts = dec["actions"]
        if self._called_112:  # 112 o singură dată; după apel nu mai punem întrebări
            acts = [a for a in acts if a["tool"] not in ("call_112", "ask_driver")]
        if self._waiting_answer(ts):
            acts = [a for a in acts if a["tool"] != "ask_driver"]
        tools = frozenset(a["tool"] for a in acts)
        if dedupe and dec["severity"] == self._last_pub["severity"] and (
            dec["severity"] == 0 or (tools <= self._last_pub["tools"] and ts - self._last_pub["ts"] < 10.0)
        ):
            return []  # nimic nou pentru șofer: nu repetăm alarma / mesajul
        dec = {**dec, "actions": acts, "ts": ts}
        if any(a["tool"] == "call_112" for a in acts):
            self._called_112 = True
        for a in acts:
            if a["tool"] == "ask_driver":
                self._asked_ts, self._asked_q = ts, a["args"]["question"]
        self._last_pub = {"severity": dec["severity"], "ts": ts, "tools": frozenset(a["tool"] for a in acts)}
        self.decisions.append(dec)
        self.publish_fn(dec)
        return [dec]

    def _reset_episode(self) -> None:
        self._timeline.clear()
        self._asked_ts = self._asked_q = None
        self._resp = None
        self._called_112 = False
        self._last_fast_reason = None

    def trip_log(self) -> dict:
        return {"a_samples": list(self.a_samples), "decisions": list(self.decisions), "events": dict(self.event_totals)}


# ----------------------------------------------------------------- CLI ----

def replay(path: Path, engine: DecisionEngine) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if "driver_response" in m:
            engine.on_driver_response(m["driver_response"], ts=m.get("ts"))
        else:
            out += engine.on_vision(m)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--ollama", action="store_true", help="folosește LLM-ul local (altfel doar reguli)")
    p.add_argument("--model", default=None)
    p.add_argument("--report", action="store_true", help="afișează textul complet al raportului 112")
    args = p.parse_args()

    llm = None
    if args.ollama:
        from .llm_client import DEFAULT_MODEL, OllamaClient
        llm = OllamaClient(model=args.model or DEFAULT_MODEL)
        print(f"warmup {llm.model}: {llm.warmup():.1f}s")

    eng = DecisionEngine(llm_fn=llm)
    for d in replay(args.file, eng):
        tools = ", ".join(a["tool"] for a in d["actions"]) or "-"
        print(f"t={d['ts']:5.1f}  sev={d['severity']}  {d['source']:<5}  [{tools}]  {d['reason']}")
        for a in d["actions"]:
            if a["tool"] == "call_112" and args.report:
                print("    " + a["args"]["report"].replace("\n", "\n    "))
    print(f"\n{eng.stats}\n")
    print(build_trip_report(eng.trip_log()))


if __name__ == "__main__":
    main()
