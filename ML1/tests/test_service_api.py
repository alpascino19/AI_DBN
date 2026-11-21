"""Serviciul ML1 + API-ul HTTP pentru serverul Spring Boot (rulat pe un port local real)."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from ml1.cv_adapter import cv_available, cv_demo_messages
from ml1.http_api import serve
from ml1.service import Ml1Service

BASE_TS = 1_763_720_000.0


def frame(t, **over):
    m = {"ts": BASE_TS + t, "camera_on": True, "face_detected": True, "ear": 0.30, "perclos": 0.03,
         "blink_rate": 15, "gaze_on_road": True, "head_drop": False, "phone": 0.0, "events": []}
    m.update(over)
    return m


class TestService(unittest.TestCase):
    def test_spring_style_payload(self):
        """camelCase (Jackson), timestamp în ms, plic {"type","payload"}."""
        svc = Ml1Service()
        for i in range(60):
            t = i / 5
            body = {"type": "vision", "payload": {
                "ts": int((BASE_TS + t) * 1000), "cameraOn": True, "faceDetected": True, "ear": 0.3, "perclos": 0.03,
                "gazeOnRoad": False, "phone": 0.85,
                "events": [{"event": "phone_use", "durationSec": max(0.0, t - 1), "confidence": 0.85}]}}
            svc.ingest_vision(body)
        sev = [d["severity"] for d in svc.decisions_since(0)]
        self.assertIn(1, sev)
        self.assertEqual(svc.counters["vision"], 60)
        self.assertAlmostEqual(svc.status()["last_ts"], BASE_TS + 59 / 5, places=2)

    def test_duplicates_and_stale_frames_are_ignored(self):
        svc = Ml1Service()
        self.assertTrue(svc.ingest_vision(frame(10))["accepted"])
        self.assertFalse(svc.ingest_vision(frame(10))["accepted"])          # duplicat
        self.assertFalse(svc.ingest_vision(frame(8))["accepted"])           # vechi cu 2 s
        self.assertTrue(svc.ingest_vision(frame(9.8))["accepted"])          # reordonare mică acceptată
        self.assertEqual((svc.counters["duplicate"], svc.counters["stale"]), (1, 1))

    def test_missing_timestamp_uses_server_clock(self):
        svc = Ml1Service()
        m = frame(0)
        del m["ts"]
        res = svc.ingest_vision(m)
        self.assertTrue(res["accepted"])
        self.assertAlmostEqual(res["ts"], time.time(), delta=2)

    def test_long_gap_closes_episode(self):
        svc = Ml1Service()
        svc.ingest_vision(frame(0))
        svc.ingest_vision(frame(100))
        self.assertEqual(svc.counters["gap_resets"], 1)

    def test_invalid_inputs(self):
        svc = Ml1Service()
        self.assertFalse(svc.ingest_vision("nu e json")["accepted"])
        self.assertFalse(svc.ingest_response({"text": "fără answered"})["accepted"])
        self.assertTrue(svc.ingest_response({"answered": True, "latencySec": 1.2, "text": "da"})["accepted"])

    def test_reset_starts_new_calibration(self):
        svc = Ml1Service()
        for i in range(200):
            svc.ingest_vision(frame(i / 5))
        self.assertTrue(svc.status()["calibration"]["ready"])
        svc.reset()
        self.assertFalse(svc.status()["calibration"]["ready"])

    def test_listener_and_subscriber_receive_decisions(self):
        svc = Ml1Service()
        got = []
        svc.add_listener(got.append)
        q = svc.subscribe()
        for i in range(60):
            svc.ingest_vision(frame(i / 5, events=[{"event": "head_on_wheel", "duration_sec": 2 + i / 5, "confidence": 0.9}]))
        self.assertTrue(got)
        self.assertEqual(q.get_nowait()["seq"], got[0]["seq"])


class TestHttpApi(unittest.TestCase):
    def setUp(self):
        self.svc = Ml1Service()
        self.server = serve(self.svc, "127.0.0.1", 0, token="secret")
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None, auth=True):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method,
                                     data=None if body is None else json.dumps(body).encode())
        req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", "Bearer secret")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_health_is_public_rest_needs_token(self):
        self.assertEqual(self.call("GET", "/api/v1/health", auth=False)[0], 200)
        self.assertEqual(self.call("GET", "/api/v1/status", auth=False)[0], 401)
        self.assertEqual(self.call("GET", "/api/v1/status")[0], 200)

    def test_vision_batch_and_decisions(self):
        frames = [frame(i / 5, events=[{"event": "head_on_wheel", "duration_sec": 2 + i / 5, "confidence": 0.9}])
                  for i in range(30)]
        code, res = self.call("POST", "/api/v1/vision", frames)
        self.assertEqual(code, 200)
        self.assertEqual(res["accepted"], 30)
        self.assertEqual(res["decisions"][0]["severity"], 3)
        code, res = self.call("GET", "/api/v1/decisions?since=0")
        self.assertTrue(res["decisions"])
        self.assertEqual(self.call("GET", "/api/v1/decisions/latest")[1]["decision"]["severity"], 3)

    def test_bad_requests(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1/vision", method="POST", data=b"{nope")
        req.add_header("Authorization", "Bearer secret")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        cm.exception.close()
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(self.call("POST", "/api/v1/driver-response", {"text": "x"})[0], 400)
        self.assertEqual(self.call("GET", "/api/v1/nope")[0], 404)

    def test_sse_stream_delivers_decisions(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/v1/decisions/stream")
        req.add_header("Authorization", "Bearer secret")
        resp = urllib.request.urlopen(req, timeout=5)
        self.assertIn("text/event-stream", resp.headers["Content-Type"])
        resp.readline()  # comentariul inițial
        resp.readline()

        def feed():
            time.sleep(0.2)
            for i in range(20):
                self.svc.ingest_vision(frame(i / 5, events=[{"event": "head_on_wheel", "duration_sec": 2 + i / 5,
                                                              "confidence": 0.9}]))
        threading.Thread(target=feed, daemon=True).start()
        lines = [resp.readline().decode().strip() for _ in range(3)]
        resp.close()
        self.assertTrue(lines[0].startswith("id: "))
        self.assertEqual(lines[1], "event: decision")
        self.assertEqual(json.loads(lines[2][len("data: "):])["severity"], 3)

    def test_reset_endpoint(self):
        self.assertEqual(self.call("POST", "/api/v1/reset")[1], {"ok": True})

    @unittest.skipUnless(cv_available(), "pachetul Vision lipsește")
    def test_cv_demo_over_http_reaches_112(self):
        """Demo-ul echipei CV trimis prin HTTP (cum va face serverul Spring): 112 o singură dată."""
        msgs = cv_demo_messages()
        for m in msgs:
            m["ts"] += BASE_TS
        for i in range(0, len(msgs), 25):
            self.assertEqual(self.call("POST", "/api/v1/vision", msgs[i:i + 25])[0], 200)
        decs = self.call("GET", "/api/v1/decisions?since=0")[1]["decisions"]
        tools = [a["tool"] for d in decs for a in d["actions"]]
        self.assertEqual(tools.count("call_112"), 1)


if __name__ == "__main__":
    unittest.main()
