"""Publică pe MQTT vigil/vision (sau pe consolă). [CV2]

    pub = Publisher(cfg)          # se conectează în fundal, se reconectează singur
    pub.publish(msg)              # VisionMessage sau dict
Fără broker (sau --no-mqtt): afișează un rezumat în consolă (max 1/s,
plus imediat când se schimbă lista de evenimente). Landmark-urile nu se afișează.

Două laptopuri (docs/vision-doua-camere.md): fiecare publică pe topicul lui
(Publisher(cfg, topic="vigil/vision/face") / ".../body"), iar client_id-ul MQTT e
unic per laptop + topic, ca brokerul să nu-i deconecteze reciproc.
"""
from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any, Union

from .schemas import VisionMessage

log = logging.getLogger(__name__)


class Publisher:
    def __init__(self, cfg: dict, enabled: bool | None = None, echo: bool = False,
                 topic: str | None = None, host: str | None = None) -> None:
        m = cfg["mqtt"]
        self.topic = topic or m["topic"]
        # unic per laptop + topic: același nume = brokerul îl dă afară pe celălalt ("session taken over")
        self.client_id = f"vigil-vision-{socket.gethostname()}-{self.topic.rsplit('/', 1)[-1]}"
        self.print_fallback = m.get("print_fallback", True)
        self.echo = echo
        self.connected = False
        self._client = None
        self._last_print = 0.0
        self._last_events: tuple = ()
        self._pending: dict = {}        # topic -> ultimul MQTTMessageInfo binar (cadre video)
        if m["enabled"] if enabled is None else enabled:
            self._connect(host or m["host"], int(m["port"]))

    def _connect(self, host: str, port: int) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.warning("paho-mqtt nu e instalat -> doar consolă")
            return
        try:
            if hasattr(mqtt, "CallbackAPIVersion"):                # paho-mqtt 2.x
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
            else:                                                   # paho-mqtt 1.x
                client = mqtt.Client(client_id=self.client_id)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.reconnect_delay_set(1, 10)
            client.connect_async(host, port, keepalive=30)
            client.loop_start()
            self._client = client
            log.info("MQTT -> %s:%d topic %s (client %s)", host, port, self.topic, self.client_id)
        except Exception as e:                                      # noqa: BLE001
            log.warning("Nu mă pot conecta la MQTT (%s) -> consolă", e)

    def _on_connect(self, client, userdata, flags, rc, *args) -> None:
        self.connected = (rc == 0) if isinstance(rc, int) else not getattr(rc, "is_failure", False)
        log.info("MQTT %s", "conectat" if self.connected else f"refuzat ({rc})")

    def _on_disconnect(self, client, userdata, *args) -> None:
        if self.connected:
            log.warning("MQTT deconectat, reîncerc...")
        self.connected = False

    def publish(self, msg: Union[VisionMessage, dict[str, Any]]) -> None:
        data = msg.to_dict() if isinstance(msg, VisionMessage) else msg
        sent = False
        if self._client is not None and self.connected:
            self._client.publish(self.topic, json.dumps(data, separators=(",", ":")), qos=0)
            sent = True
        if self.echo or (not sent and self.print_fallback):
            self._print(data)

    def publish_bytes(self, topic: str, data: bytes) -> bool:
        """Date binare (cadrele JPEG pentru dashboard, video.py) pe alt topic; QoS 0, fără retain.
        Dacă cadrul precedent de pe același topic nu a plecat încă (rețea lentă), îl sărim pe cel nou: imaginea
        rămâne actuală în loc să rămână tot mai în urmă. Fără broker nu se afișează nimic în consolă."""
        if self._client is None or not self.connected:
            return False
        last = self._pending.get(topic)
        if last is not None and not last.is_published():
            return False
        self._pending[topic] = self._client.publish(topic, data, qos=0)
        return True

    def _print(self, data: dict) -> None:
        events = tuple((e["event"], round(e["duration_sec"])) for e in data.get("events", []))
        names = tuple(e[0] for e in events)
        now = time.monotonic()
        if now - self._last_print < 1.0 and names == tuple(e[0] for e in self._last_events):
            return
        self._last_print, self._last_events = now, events
        short = {k: v for k, v in data.items() if k not in ("landmarks", "body_landmarks")}
        print(f"[{self.topic}] {json.dumps(short, ensure_ascii=False)}", flush=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
