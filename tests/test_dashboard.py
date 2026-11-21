"""Teste pentru scheletul mock și rularea în timp real a dashboard-ului."""

import json
import time
import unittest

from ml1.dashboard import Runner
from ml1.fake_vision import SCENARIOS, generate
from ml1.mock_pose import PoseModel


def run_pose(frames, until=None):
    pm, last = PoseModel(), None
    for m in frames:
        if "driver_response" in m:
            continue
        last = pm.update(m, m.get("ts", 0.0))
        if until is not None and m.get("ts", 0) >= until:
            break
    return last


class TestPose(unittest.TestCase):
    def test_normal_posture_has_no_flags(self):
        p = run_pose(generate("calibration"), until=5)
        self.assertFalse(any(p["flags"].values()))
        self.assertGreater(p["eyes"]["open"], 0.9)

    def test_phone_raises_hand_to_face(self):
        calm, phone = run_pose(generate("phone"), until=3), run_pose(generate("phone"), until=11)
        self.assertTrue(phone["flags"]["phone"])
        self.assertLess(phone["wr_r"][1], calm["wr_r"][1] - 0.3, "mâna urcă spre față")
        self.assertIsNotNone(phone["phone"])
        self.assertTrue(phone["flags"]["gaze_off"])

    def test_hand_on_chest_and_heavy_breathing(self):
        p = run_pose(generate("chest"), until=10)
        self.assertTrue(p["flags"]["chest"] and p["flags"]["heavy"])
        self.assertLess(abs(p["wr_r"][0] - p["chest"][0]), 0.08, "încheietura ajunge la piept")
        self.assertGreater(p["mouth"], 0.3)

    def test_head_drop_and_closed_eyes(self):
        calm, col = run_pose(generate("collapse"), until=1), run_pose(generate("collapse"), until=12)
        self.assertTrue(col["flags"]["drop"] and col["flags"]["eyes_closed"])
        self.assertGreater(col["head"][1], calm["head"][1] + 0.1)
        self.assertGreater(col["tilt"], 0.3)

    def test_heavy_breathing_moves_shoulders_more(self):
        def amp(name, until):
            pm, ys = PoseModel(), []
            for m in generate(name):
                if "driver_response" not in m and 5 <= m.get("ts", 0) <= until:
                    ys.append(pm.update(m, m["ts"])["sh_l"][1])
                elif "driver_response" not in m:
                    pm.update(m, m["ts"])
            return max(ys) - min(ys)
        self.assertGreater(amp("chest", 14), 2 * amp("calibration", 14))

    def test_garbage_input_never_breaks(self):
        pm = PoseModel()
        for m in generate("noisy"):
            p = pm.update(m, m.get("ts", 0.0))
            json.dumps(p, allow_nan=False)

    def test_coordinates_stay_inside_canvas(self):
        for name in ("demo", "collapse", "phone"):
            pm = PoseModel()
            for m in generate(name):
                if "driver_response" in m:
                    continue
                p = pm.update(m, m["ts"])
                for k in ("neck", "sh_l", "sh_r", "el_l", "el_r", "wr_l", "wr_r", "chest"):
                    self.assertTrue(0 <= p[k][0] <= 1 and 0 <= p[k][1] <= 1, (name, k, p[k]))


class TestRunner(unittest.TestCase):
    def finish(self, r, timeout=20):
        end = time.time() + timeout
        while time.time() < end and r.status not in ("done", "error", "stopped"):
            time.sleep(0.05)

    def test_all_scenarios_run_and_serialize(self):
        for name in SCENARIOS:
            if name == "demo":
                continue  # 138 s de date: testat mai jos cu viteză mare
            with self.subTest(name):
                r = Runner()
                r.start(name, 8.0, False)
                self.finish(r)
                snap = r.snapshot(0)
                self.assertEqual(snap["status"], "done", snap["note"])
                json.dumps(snap, allow_nan=False)
                self.assertGreater(snap["total"], 40)
                self.assertTrue(all(0 <= f["a"] <= 1 for f in snap["frames"]))
                self.assertIn("RAPORT POST-CURSĂ", snap["trip_report"])

    def test_demo_full_flow_and_incremental_state(self):
        r = Runner()
        r.start("demo", 8.0, False)
        self.finish(r, 40)
        snap = r.snapshot(0)
        self.assertEqual(snap["status"], "done", snap["note"])
        decs = [d for f in snap["frames"] for d in f["decisions"]]
        self.assertEqual(sum(1 for d in decs for a in d["actions"] if a["tool"] == "call_112"), 2)
        self.assertEqual({d["severity"] for d in decs}, {0, 1, 2, 3})
        tail = r.snapshot(snap["total"] - 5)
        self.assertEqual(len(tail["frames"]), 5, "`since` întoarce doar cadrele noi")

    def test_stop_and_restart(self):
        r = Runner()
        r.start("demo", 1.0, False)
        time.sleep(0.5)
        r.stop()
        self.assertEqual(r.status, "stopped")
        first = r.run_id
        r.start("glance", 8.0, False)
        self.assertGreater(r.run_id, first)
        self.finish(r)
        self.assertEqual(r.snapshot(0)["scenario"], "glance")

    def test_unknown_scenario(self):
        self.assertFalse(Runner().start("nu_exista", 1.0, False)["ok"])


if __name__ == "__main__":
    unittest.main()
