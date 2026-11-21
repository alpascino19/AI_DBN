"""Puntea MQTT a ML1: legătura cu echipele CV și ML2.

    pip install paho-mqtt                 # o singură dată
    python -m ml1.mqtt_bridge             # broker local (localhost:1883)
    python -m ml1.mqtt_bridge --llm --host 192.168.1.20

Topicuri (contractele din TASKS.md):
    vigil/vision           CV  -> ML1   cadrele camerei (schemas.VisionMessage)
    vigil/driver_response  ML2 -> ML1   răspunsul la „Sunteți bine?”
    vigil/decision         ML1 -> ML2, dashboard   deciziile (severitate + acțiuni)
    vigil/ml1/status       ML1 -> dashboard        scor a, încredere, calibrare (1 Hz)

`Bridge` conține toată logica și nu depinde de paho (testabilă fără broker);
`run_mqtt` doar leagă `Bridge` de un client paho.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Callable

from .engine import DecisionEngine
from .service import Ml1Service

log = logging.getLogger("ml1.mqtt")

TOPIC_VISION = "vigil/vision"
TOPIC_RESPONSE = "vigil/driver_response"
TOPIC_DECISION = "vigil/decision"
TOPIC_STATUS = "vigil/ml1/status"


def _decision_payload(d: dict) -> dict:
    """Doar câmpurile din contractul vigil/decision (+ diagnostic util pentru dashboard)."""
    keep = ("a", "severity", "actions", "reason", "source", "ts", "latency_ms",
            "safety_notes", "fallback_reason")
    return {k: d[k] for k in keep if k in d}


class Bridge:
    def __init__(self, target: DecisionEngine | Ml1Service, send: Callable[[str, dict], None], status_hz: float = 1.0):
        self.svc = target if isinstance(target, Ml1Service) else Ml1Service(engine=target)
        self.engine = self.svc.engine
        self.send = send
        self.status_period = 1.0 / status_hz if status_hz > 0 else float("inf")
        self._last_status = -float("inf")
        self.received = {"vision": 0, "response": 0, "invalid": 0}
        self.svc.add_listener(lambda d: self.send(TOPIC_DECISION, _decision_payload(d)))

    def on_message(self, topic: str, payload: bytes | str) -> None:
        try:
            msg = json.loads(payload)
        except (ValueError, TypeError):
            self.received["invalid"] += 1
            log.warning("JSON invalid pe %s", topic)
            return
        if not isinstance(msg, dict):
            self.received["invalid"] += 1
            return
        if topic == TOPIC_VISION:
            self.received["vision"] += 1
            res = self.svc.ingest_vision(msg)   # deciziile pleacă prin listener
            if res.get("accepted"):
                self._maybe_status(res["ts"])
        elif topic == TOPIC_RESPONSE:
            self.received["response"] += 1
            self.svc.ingest_response(msg)

    def _maybe_status(self, ts: float) -> None:
        if ts - self._last_status < self.status_period:
            return
        self._last_status = ts
        r = self.engine.last_result
        if r is None:
            return
        self.send(TOPIC_STATUS, {
            "ts": ts, "a": round(r.a, 3), "confidence": round(r.confidence, 3), "critical": r.critical,
            "camera_on": r.camera_on, "capped_by": r.capped_by, "calibration": self.engine.baseline.to_dict(),
            "stats": dict(self.engine.stats), "received": dict(self.received), "counters": dict(self.svc.counters),
        })


def run_mqtt(host: str, port: int, use_llm: bool) -> None:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise SystemExit("Lipsește paho-mqtt: pip install paho-mqtt")

    llm = None
    if use_llm:
        from .llm_client import OllamaClient
        client_llm = OllamaClient()
        if client_llm.available():
            log.info("warmup %s: %.1fs", client_llm.model, client_llm.warmup())
            llm = client_llm
        else:
            log.warning("Ollama indisponibil: rulez doar cu reguli fixe")

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-ml1")
    else:
        client = mqtt.Client(client_id="vigil-ml1")

    def send(topic: str, obj: dict) -> None:
        client.publish(topic, json.dumps(obj, ensure_ascii=False), qos=1 if topic == TOPIC_DECISION else 0)

    bridge = Bridge(Ml1Service(llm=llm), send)

    def on_connect(c, userdata, flags, rc, *args):
        log.info("conectat la %s:%d", host, port)
        c.subscribe([(TOPIC_VISION, 0), (TOPIC_RESPONSE, 1)])

    client.on_connect = on_connect
    client.on_message = lambda c, u, m: bridge.on_message(m.topic, m.payload)
    client.reconnect_delay_set(1, 10)
    client.connect(host, port, keepalive=30)
    log.info("ML1 ascultă pe %s și %s", TOPIC_VISION, TOPIC_RESPONSE)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--llm", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_mqtt(args.host, args.port, args.llm)


if __name__ == "__main__":
    main()
