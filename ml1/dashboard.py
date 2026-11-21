"""Dashboard local pentru ML 1: corp (mock) + grafice, în timp real.

    python -m ml1.dashboard                # pornește serverul și deschide browserul
    python -m ml1.dashboard --port 8800 --no-open
    python -m ml1.dashboard --autostart demo --speed 2 --llm

Doar biblioteca standard. Serverul rulează scenariul prin același `DecisionEngine`
din producție (fuziune -> reguli -> LLM -> validare) și trimite cadrele către pagină.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import DecisionEngine
from .cv_adapter import cv_available, cv_demo_messages
from .fake_vision import SCENARIOS as ML1_SCENARIOS, generate as ml1_generate

# Scenariile ML1 + demo-ul echipei CV (forma reală a mesajelor), dacă pachetul Vision e prezent.
SCENARIOS = dict(ML1_SCENARIOS)
if cv_available():
    def cv_demo(rng=None):
        """Demo-ul echipei CV (Vision/vigil_vision/fake): mesaje exact ca de la cameră, fără răspunsuri vocale."""
    SCENARIOS["cv_demo"] = cv_demo


def generate(name: str) -> list[dict]:
    return cv_demo_messages() if name == "cv_demo" else ml1_generate(name)
from .fusion import WEIGHTS
from .mock_pose import PoseModel
from .reports import build_trip_report

WEB = Path(__file__).resolve().parent / "web"
SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clean_events(evs):
    out = []
    for e in evs if isinstance(evs, list) else []:
        if isinstance(e, dict) and isinstance(e.get("event"), str):
            out.append({"event": e["event"], "duration_sec": _num(e.get("duration_sec")) or 0.0,
                        "confidence": _num(e.get("confidence")) or 0.0,
                        **({"breathing": e["breathing"]} if isinstance(e.get("breathing"), str) else {})})
    return out


def _scenario_info() -> list[dict]:
    info = []
    for name, fn in SCENARIOS.items():
        msgs = generate(name)
        ts = [m["ts"] for m in msgs if isinstance(m.get("ts"), (int, float))]
        doc = (fn.__doc__ or "").strip().splitlines()
        info.append({"name": name, "duration": round(max(ts) if ts else 0, 1), "about": doc[0] if doc else ""})
    return info


class Runner:
    """Rulează un scenariu în fundal; păstrează cadrele pentru dashboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.run_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reset("idle")
        self.info = _scenario_info()

    def reset(self, status="idle", note=""):
        self.status, self.note = status, note
        self.frames: list[dict] = []
        self.scenario, self.speed, self.llm_on, self.duration = "", 1.0, False, 0.0
        self.stats: dict = {}
        self.trip_report = ""
        self.model = ""

    # ---------------------------------------------------------------- control
    def start(self, scenario: str, speed: float, use_llm: bool) -> dict:
        if scenario not in SCENARIOS:
            return {"ok": False, "error": f"scenariu necunoscut: {scenario}"}
        self.stop()
        with self.lock:
            self.run_id += 1
            rid = self.run_id
            self.reset("starting")
            self.scenario, self.speed, self.llm_on = scenario, speed, use_llm
            self.duration = next(i["duration"] for i in self.info if i["name"] == scenario)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(rid, scenario, speed, use_llm, self._stop), daemon=True)
        self._thread.start()
        return {"ok": True, "run": rid}

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self.lock:
            if self.status in ("starting", "warming", "running"):
                self.status = "stopped"

    def snapshot(self, since: int) -> dict:
        with self.lock:
            return {
                "run": self.run_id, "status": self.status, "note": self.note, "scenario": self.scenario,
                "speed": self.speed, "llm": self.llm_on, "model": self.model, "duration": self.duration,
                "total": len(self.frames), "since": since, "frames": self.frames[since:],
                "stats": self.stats, "trip_report": self.trip_report, "weights": WEIGHTS,
            }

    # ------------------------------------------------------------------ run
    def _set(self, rid, **kw):
        with self.lock:
            if rid == self.run_id:
                for k, v in kw.items():
                    setattr(self, k, v)

    def _run(self, rid: int, scenario: str, speed: float, use_llm: bool, stop: threading.Event) -> None:
        try:
            llm = None
            if use_llm:
                from .llm_client import OllamaClient
                client = OllamaClient()
                if client.available():
                    self._set(rid, status="warming", note=f"încarc {client.model}…", model=client.model)
                    client.warmup()
                    llm = client
                    self._set(rid, note="")
                else:
                    self._set(rid, note="Ollama indisponibil: rulez doar cu reguli fixe", model="(reguli)")
            pending: list[dict] = []
            plock = threading.Lock()

            def on_decision(d):
                with plock:
                    pending.append(d)

            eng = DecisionEngine(llm_fn=llm, publish=on_decision, sync=llm is None)
            pose = PoseModel()
            msgs = generate(scenario)
            self._set(rid, status="running")

            t0 = time.monotonic()
            first = next((m["ts"] for m in msgs if _num(m.get("ts")) is not None), 0.0)
            last_ts = first
            answers: list[dict] = []
            for m in msgs:
                if stop.is_set():
                    return
                ts = _num(m.get("ts"))
                ts = last_ts + 0.2 if ts is None else ts
                due = (ts - first) / speed
                while time.monotonic() - t0 < due:
                    if stop.wait(0.01):
                        return
                last_ts = ts

                if "driver_response" in m:
                    eng.on_driver_response(m["driver_response"], ts=ts)
                    answers.append({"ts": ts, **m["driver_response"]})
                    continue

                msg = {**m, "ts": ts}
                eng.on_vision(msg)
                r = eng.last_result.to_dict()
                with plock:
                    decs, pending[:] = list(pending), []
                frame = {
                    "i": len(self.frames), "ts": ts, "a": r["a"], "a_raw": r["a_raw"], "conf": r["confidence"],
                    "critical": r["critical"], "camera_on": r["camera_on"], "capped_by": r["capped_by"],
                    "contrib": r["contributions"],
                    "vision": {
                        "ear": _num(msg.get("ear")), "perclos": _num(msg.get("perclos")),
                        "blink": _num(msg.get("blink_rate")), "phone": _num(msg.get("phone")),
                        "gaze_on_road": msg.get("gaze_on_road") if isinstance(msg.get("gaze_on_road"), bool) else None,
                        "head_drop": msg.get("head_drop") if isinstance(msg.get("head_drop"), bool) else None,
                        "events": _clean_events(msg.get("events")),
                    },
                    "pose": pose.update(msg, ts),
                    "decisions": decs, "answers": answers,
                }
                answers = []
                with self.lock:
                    if rid != self.run_id:
                        return
                    self.frames.append(frame)
                    self.stats = dict(eng.stats)

            if llm is not None:  # așteptăm un eventual răspuns LLM încă în zbor
                deadline = time.monotonic() + 3.0
                while eng._busy and time.monotonic() < deadline:
                    time.sleep(0.05)
            with plock:
                late = list(pending)
            with self.lock:
                if rid == self.run_id:
                    if late and self.frames:
                        self.frames.append({**self.frames[-1], "i": len(self.frames), "decisions": late, "answers": []})
                    self.stats = dict(eng.stats)
                    self.trip_report = build_trip_report(eng.trip_log())
                    self.status = "done"
        except Exception as e:  # nu lăsăm thread-ul să moară în tăcere
            self._set(rid, status="error", note=f"{type(e).__name__}: {e}")


RUNNER = Runner()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # liniște în consolă
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/scenarios":
            self._json({"scenarios": RUNNER.info, "speeds": SPEEDS})
        elif u.path == "/api/start":
            try:
                speed = float(q.get("speed", 1))
            except ValueError:
                speed = 1.0
            speed = min(SPEEDS, key=lambda s: abs(s - speed))
            self._json(RUNNER.start(q.get("scenario", "demo"), speed, q.get("llm") == "1"))
        elif u.path == "/api/stop":
            RUNNER.stop()
            self._json({"ok": True})
        elif u.path == "/api/state":
            try:
                since = max(0, int(q.get("since", 0)))
            except ValueError:
                since = 0
            self._json(RUNNER.snapshot(since))
        else:
            self._send(404, b"not found", "text/plain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--autostart", choices=list(SCENARIOS), help="pornește direct un scenariu")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--llm", action="store_true", help="folosește LLM-ul local (Ollama)")
    args = p.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Vigil dashboard: {url}   (Ctrl+C pentru oprire)")
    if args.autostart:
        RUNNER.start(args.autostart, args.speed, args.llm)
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        server.server_close()


if __name__ == "__main__":
    main()
