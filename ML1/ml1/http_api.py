"""API HTTP al ML1 pentru serverul Spring Boot (și orice alt client). Doar biblioteca standard.

    python -m ml1.http_api                         # http://127.0.0.1:8090, doar reguli
    python -m ml1.http_api --llm --port 8090       # cu LLM-ul local
    set ML1_API_TOKEN=secret                       # opțional: cere "Authorization: Bearer secret"

Rute (JSON, UTF-8). Contractul complet + exemple Java: docs/API.md
    POST /api/v1/vision              un cadru vigil/vision SAU o listă de cadre -> deciziile noi
    POST /api/v1/driver-response     răspunsul șoferului (ML2)
    GET  /api/v1/decisions?since=N   deciziile cu seq > N
    GET  /api/v1/decisions/latest    ultima decizie (sau null)
    GET  /api/v1/decisions/stream    Server-Sent Events: `event: decision` la fiecare decizie nouă
    GET  /api/v1/status              scor, încredere, calibrare, contoare
    POST /api/v1/reset               călătorie nouă / alt șofer (calibrare de la zero)
    GET  /api/v1/health              {"ok": true}
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .service import Ml1Service

MAX_BODY = 2 * 1024 * 1024      # 2 MB: un cadru cu landmark-uri are ~30 KB
SSE_KEEPALIVE_SEC = 15.0


def make_handler(svc: Ml1Service, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        # ---------------------------------------------------------- utilitare
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            if self.headers.get("Authorization", "") == f"Bearer {token}":
                return True
            self._json({"error": "neautorizat"}, 401)
            return False

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return None, ("corp lipsă sau prea mare" if n <= 0 else "corp prea mare")
            try:
                return json.loads(self.rfile.read(n).decode("utf-8")), None
            except (ValueError, UnicodeDecodeError):
                return None, "JSON invalid"

        # ---------------------------------------------------------- rute
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            if not self._authorized():
                return
            path = urlparse(self.path).path
            if path == "/api/v1/reset":
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(min(n, MAX_BODY))
                return self._json(svc.reset())
            body, err = self._body()
            if err:
                return self._json({"error": err}, 400)
            if path == "/api/v1/vision":
                return self._json(svc.ingest_many(body) if isinstance(body, list) else svc.ingest_vision(body))
            if path == "/api/v1/driver-response":
                res = svc.ingest_response(body)
                return self._json(res, 200 if res.get("accepted") else 400)
            self._json({"error": "rută necunoscută"}, 404)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/v1/health":
                return self._json({"ok": True})
            if not self._authorized():
                return
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/v1/status":
                return self._json(svc.status())
            if u.path == "/api/v1/decisions":
                try:
                    since = int(q.get("since", 0))
                except ValueError:
                    since = 0
                return self._json({"decisions": svc.decisions_since(since)})
            if u.path == "/api/v1/decisions/latest":
                return self._json({"decision": svc.latest_decision()})
            if u.path == "/api/v1/decisions/stream":
                return self._stream()
            self._json({"error": "rută necunoscută"}, 404)

        def _stream(self):
            sub = svc.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(b": ml1 stream\n\n")
                self.wfile.flush()
                while True:
                    try:
                        d = sub.get(timeout=SSE_KEEPALIVE_SEC)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    data = json.dumps(d, ensure_ascii=False, default=str)
                    self.wfile.write(f"id: {d['seq']}\nevent: decision\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                svc.unsubscribe(sub)
                self.close_connection = True

    return Handler


def serve(svc: Ml1Service, host: str = "127.0.0.1", port: int = 8090, token: str | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(svc, token))
    server.daemon_threads = True
    return server


def attach_mqtt(svc: Ml1Service, broker: str, port: int = 1883):
    """Cadrele vin prin HTTP (serverul Spring), dar ML2 ascultă pe MQTT: deciziile pleacă pe vigil/decision,
    iar răspunsurile șoferului (vigil/driver_response) intră înapoi în același serviciu."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("Lipsește paho-mqtt: deciziile NU ajung la ML2 (pip install paho-mqtt)")
        return None
    from .mqtt_bridge import TOPIC_DECISION, TOPIC_RESPONSE, _decision_payload
    client = (mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-ml1-http")
              if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client(client_id="vigil-ml1-http"))
    client.on_connect = lambda c, *a: c.subscribe(TOPIC_RESPONSE, qos=1)
    client.on_message = lambda c, u, m: svc.ingest_response(json.loads(m.payload))
    client.reconnect_delay_set(1, 10)
    client.connect_async(broker, port, keepalive=30)
    client.loop_start()
    svc.add_listener(lambda d: client.publish(TOPIC_DECISION, json.dumps(_decision_payload(d), ensure_ascii=False), qos=1))
    print(f"decizii -> MQTT {broker}:{port} {TOPIC_DECISION}")
    return client


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 ca serverul Spring să ajungă din rețea")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--llm", action="store_true")
    p.add_argument("--mqtt-out", action="store_true",
                   help="publică deciziile și pe MQTT (vigil/decision) pentru ML2; răspunsurile ML2 vin înapoi")
    p.add_argument("--broker", default="localhost")
    args = p.parse_args()

    llm = None
    if args.llm:
        from .llm_client import OllamaClient
        c = OllamaClient()
        if c.available():
            print(f"warmup {c.model}: {c.warmup():.1f}s")
            llm = c
        else:
            print("Ollama indisponibil: doar reguli fixe")
    svc = Ml1Service(llm=llm)
    if args.mqtt_out:
        attach_mqtt(svc, args.broker)
    token = os.environ.get("ML1_API_TOKEN") or None
    server = serve(svc, args.host, args.port, token)
    print(f"ML1 API: http://{args.host}:{args.port}/api/v1  (token: {'da' if token else 'nu'})  Ctrl+C oprește")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
