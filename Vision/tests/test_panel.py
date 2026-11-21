"""Panoul ferestrelor FATA / CORP trimis dashboard-ului (vigil_vision.panel). Rulare din Vision/:
python -m unittest discover -s tests"""
import json
import unittest
from types import SimpleNamespace as NS

import numpy as np

from vigil_vision.face.eyes import EyeState
from vigil_vision.face.head_pose import HeadState
from vigil_vision.face.unconscious import UnconsciousState
from vigil_vision.face.yawn import YawnState
from vigil_vision.merge import merge
from vigil_vision.panel import body_panel, face_panel

ALERT = NS(level=2, message="Ochi inchisi 1.6s", active_sec=np.float32(1.6))


class TestFacePanel(unittest.TestCase):
    def snap(self):
        eyes = EyeState()
        eyes.closure = np.float32(0.734)          # valorile reale vin des ca tipuri numpy
        eyes.perclos = np.float64(0.12)
        return NS(eyes=eyes, head=HeadState(), yawn=YawnState(), unconscious=UnconsciousState())

    def test_serializable_with_numpy_values(self):
        am = NS(level=2, active=[ALERT])
        p = face_panel(self.snap(), NS(done=True), am, 10.0, fps=29.7)
        json.dumps(p)
        self.assertEqual(p["eyes"]["closure"], 0.73)
        self.assertEqual(p["level"], 2)
        self.assertEqual(p["fps"], 30)
        self.assertEqual(p["alerts"], [{"level": 2, "message": "Ochi inchisi 1.6s", "sec": 2.0}])

    def test_calibration_hides_alerts_and_reports_progress(self):
        calib = NS(done=False, progress=lambda t: 0.4)
        p = face_panel(self.snap(), calib, NS(level=3, active=[ALERT]), 2.0)
        self.assertEqual((p["level"], p["calibration"], p["alerts"]), (0, 0.4, []))

    def test_no_face_snapshot(self):
        self.assertIsNone(face_panel(None, None, NS(level=0, active=[]), 0.0))


class TestBodyPanel(unittest.TestCase):
    def test_fields(self):
        ctx = NS(pose=[object()], detections={"hand_on_chest": NS(active=True, attrs={}),
                                             "phone_use": NS(active=False, attrs={"usage": "call"})},
                 signals={"chest_src": "hands", "chest_hand_scale": np.float32(0.81234),
                          "breathing": {"state": "heavy", "score": 0.6, "rate_bpm": 31.6},
                          "phone": np.float64(0.03), "slump": {"state": "upright"},
                          "chest_zone": (0.2, 0.5, 0.8, 0.9)},
                 geom=NS(sx=0.5, sy=0.55))
        p = body_panel(ctx, NS(level=3, active=[ALERT]), fps=11.2)
        json.dumps(p)
        self.assertTrue(p["chest"]["on"])
        self.assertEqual(p["hand_scale"], 0.81)
        self.assertEqual(p["breathing"]["rate_bpm"], 32)
        self.assertFalse(p["phone"]["active"])
        self.assertEqual(p["chest_zone"], [0.2, 0.5, 0.8, 0.9])
        self.assertEqual(p["neck"], [0.5, 0.55])
        self.assertEqual(p["fps"], 11)


class TestMergeKeepsPanels(unittest.TestCase):
    def test_each_panel_comes_from_its_camera(self):
        face = {"ear": 0.27, "face_detected": True, "face_panel": {"level": 0}, "body_panel": {"x": "greșit"}}
        body = {"phone": 0.1, "body_detected": True, "body_panel": {"level": 1}, "face_panel": {"x": "greșit"}}
        m = merge({"face": (10.0, face), "body": (10.0, body)}, 10.1, wall=1.0)
        self.assertEqual(m["face_panel"], {"level": 0})
        self.assertEqual(m["body_panel"], {"level": 1})


if __name__ == "__main__":
    unittest.main()
