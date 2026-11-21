"""Combină cele două camere într-un singur vigil/vision, formatul pe care îl așteaptă ML1, ML2 și dashboard-ul.

    vigil/vision/face  (run_face --publish) ─┐
                                              ├─► merge ──► vigil/vision ──► ML1, dashboard
    vigil/vision/body  (run_body --publish) ─┘

    cd Vision
    python -m vigil_vision.merge                       # broker local (PC-ul central)
    python -m vigil_vision.merge --broker 172.22.118.134

Pornit automat de START/run_vigil.py cu --cameras two / remote. Propunerea: docs/vision-doua-camere.md.

De la fiecare cameră se ia doar ce vede ea bine: fața dă ochii, privirea, scheletul feței; corpul dă telefonul,
respirația, postura, scheletul corpului. Evenimentele ambelor se adună, `head_drop` e adevărat dacă îl vede
oricare. O cameră care tace mai mult de STALE_SEC e tratată ca pierdută: câmpurile ei lipsesc, iar
`face_detected` / `body_detected` devin false (ML1 scade încrederea, nu inventează valori). Nu publică nimic
până nu a sosit primul mesaj de la o cameră.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time

STALE_SEC = 1.0
PUBLISH_HZ = 10
FACE_KEYS = ("ear", "perclos", "blink_rate", "gaze_on_road", "landmarks", "face_detected")
BODY_KEYS = ("phone", "breathing", "posture", "body_landmarks", "body_detected")
TOPIC_IN = "vigil/vision/+"        # nu prinde vigil/vision (fără sufix): nu există buclă
TOPIC_OUT = "vigil/vision"

log = logging.getLogger("merge")


def merge_events(*lists: list) -> list:
    """Evenimentele ambelor camere; același nume în ambele -> cel mai lung, cu cea mai mare încredere."""
    best: dict[str, dict] = {}
    for events in lists:
        for e in events or []:
            if not isinstance(e, dict) or "event" not in e:
                continue
            old = best.get(e["event"])
            if old is None:
                best[e["event"]] = dict(e)
            else:
                merged = dict(e if e.get("duration_sec", 0) >= old.get("duration_sec", 0) else old)
                merged["confidence"] = max(e.get("confidence", 0), old.get("confidence", 0))
                best[e["event"]] = merged
    return list(best.values())


def merge(latest: dict, now: float, wall: float | None = None) -> dict:
    """latest: {"face": (momentul primirii, mesaj), "body": (...)}; now: același ceas ca momentul primirii."""
    fresh = {}
    for role in ("face", "body"):
        t, m = latest.get(role, (0.0, None))
        if m is not None and now - t <= STALE_SEC:
            fresh[role] = m
    out: dict = {"ts": round(time.time() if wall is None else wall, 3), "camera_on": bool(fresh),
                 "face_detected": False, "body_detected": False, "head_drop": False, "events": []}
    for role, keys in (("face", FACE_KEYS), ("body", BODY_KEYS)):
        m = fresh.get(role)
        if m is not None:
            out.update({k: m[k] for k in keys if k in m})
    out["head_drop"] = any(bool(m.get("head_drop")) for m in fresh.values())
    out["events"] = merge_events(*(m.get("events") for m in fresh.values()))
    fps = [m["fps"] for m in fresh.values() if isinstance(m.get("fps"), (int, float))]
    if fps:
        out["fps"] = round(min(fps), 1)
    return out


def main() -> None:
    import paho.mqtt.client as mqtt

    ap = argparse.ArgumentParser(description="Combină vigil/vision/face + /body în vigil/vision")
    ap.add_argument("--broker", default=os.environ.get("VIGIL_BROKER", "localhost"))
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    latest: dict = {}
    lock = threading.Lock()
    seen = {"any": False}

    def on_connect(client, userdata, flags, rc, *a):
        client.subscribe(TOPIC_IN)
        log.info("conectat la %s:%d, ascult %s", args.broker, args.port, TOPIC_IN)

    def on_message(client, userdata, msg):
        role = msg.topic.rsplit("/", 1)[-1]
        if role not in ("face", "body"):
            return
        try:
            data = json.loads(msg.payload)
        except ValueError:
            return                                    # mesaj stricat: ignorat
        if not isinstance(data, dict):
            return
        with lock:
            if role not in latest:
                log.info("prima cameră %s a răspuns", role)
            latest[role] = (time.monotonic(), data)
            seen["any"] = True

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-merge")
    else:
        client = mqtt.Client(client_id="vigil-merge")
    client.on_connect, client.on_message = on_connect, on_message
    client.reconnect_delay_set(1, 10)
    client.connect_async(args.broker, args.port, keepalive=30)
    client.loop_start()
    state: dict[str, bool] = {}                          # doar camerele care au trimis cel puțin o dată
    try:
        while True:
            time.sleep(1.0 / PUBLISH_HZ)
            with lock:
                if not seen["any"]:
                    continue
                snapshot = dict(latest)
            out = merge(snapshot, time.monotonic())
            for role in ("face", "body"):                # un rând în log când o cameră cade / revine
                if role not in snapshot:
                    continue
                alive = time.monotonic() - snapshot[role][0] <= STALE_SEC
                if alive != state.get(role, True):
                    log.warning("camera %s: %s", role, "revenită" if alive else "nu mai trimite (câmpurile ei lipsesc)")
                    state[role] = alive
            client.publish(TOPIC_OUT, json.dumps(out, separators=(",", ":")), qos=0)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
