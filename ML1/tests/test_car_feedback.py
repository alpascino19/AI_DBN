"""ECU -> ML1: starea reală a mașinii (vigil/car/state, vigil/car/event, publicate de ECU_Arduino) intră în
vigil/ml1/status și în raportul 112."""

import json
import unittest
from pathlib import Path

from ml1.engine import DecisionEngine
from ml1.mqtt_bridge import TOPIC_CAR_EVENT, TOPIC_CAR_STATE, TOPIC_STATUS, TOPIC_VISION, Bridge
from ml1.reports import build_112_report

DATA = Path(__file__).resolve().parent / "data"

STATE = {"speed_kmh": 72.4, "target_kmh": 0, "braking": True, "hazard": True, "stopped": False,
         "emergency": True, "alert": 3}
EVENT = {"event": "emergency_braking_started", "speed_kmh": 70.0, "target_kmh": 0, "decel_mps2": 2.5,
         "hazard": True, "emergency": True, "id": "ml2-3"}


class TestCarFeedback(unittest.TestCase):
    def test_bridge_keeps_car_state_and_last_event(self):
        br = Bridge(DecisionEngine(), lambda t, o: None)
        br.on_message(TOPIC_CAR_STATE, json.dumps(STATE))
        br.on_message(TOPIC_CAR_EVENT, json.dumps(EVENT))
        car = br.engine.car
        self.assertEqual(car["speed_kmh"], 70.0)          # evenimentul e mai nou decât starea
        self.assertTrue(car["braking"] and car["hazard"] and car["emergency"])
        self.assertEqual(car["last_event"], "emergency_braking_started")
        self.assertEqual(car["last_id"], "ml2-3")         # comanda ML2 la care răspunde ECU
        self.assertEqual(br.received["car"], 2)

    def test_bad_car_messages_ignored(self):
        br = Bridge(DecisionEngine(), lambda t, o: None)
        br.on_message(TOPIC_CAR_STATE, "{x")
        br.on_message(TOPIC_CAR_STATE, json.dumps([1, 2]))
        self.assertIsNone(br.engine.car)

    def test_status_includes_car(self):
        sent = []
        br = Bridge(DecisionEngine(), lambda t, o: sent.append((t, o)))
        br.on_message(TOPIC_CAR_STATE, json.dumps(STATE))
        frame = next(l for l in (DATA / "calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
        br.on_message(TOPIC_VISION, frame)
        status = [o for t, o in sent if t == TOPIC_STATUS]
        self.assertTrue(status, "puntea publică vigil/ml1/status")
        self.assertEqual(status[-1]["car"]["speed_kmh"], 72.4)

    def test_context_carries_car_for_reports(self):
        e = DecisionEngine()
        self.assertIsNone(e._context({}, {}, 1.0)["car"])
        e.on_car(STATE)
        self.assertEqual(e._context({}, {}, 1.0)["car"]["speed_kmh"], 72.4)

    def test_112_report_uses_real_car_state(self):
        stopped = {"speed_kmh": 0.0, "stopped": True, "braking": False, "hazard": True}
        self.assertIn("Vehicul (ECU): 0 km/h, oprit, avarii pornite.",
                      build_112_report({"fusion": {"a": 0.1}, "car": stopped}, now=0))
        self.assertIn("Vehicul (ECU): 72 km/h, frânează până la oprire, avarii pornite.",
                      build_112_report({"fusion": {"a": 0.1}, "car": STATE}, now=0))
        # fără ECU conectat: textul de dinainte
        self.assertIn("Vehicul: avarii pornite, manevră de oprire în siguranță.",
                      build_112_report({"fusion": {"a": 0.1}}, now=0))


if __name__ == "__main__":
    unittest.main()
