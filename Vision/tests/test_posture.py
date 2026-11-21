"""Postura (body/slump.py): căzut într-o parte, aplecat prelungit, ieșit din poziție; respirația oprită.
Rulare din Vision/: python -m unittest tests.test_posture"""
import math
import unittest

from vigil_vision.body.geometry import BodyGeometry
from vigil_vision.body.slump import SlumpAnalyzer
from vigil_vision.context import Landmark, P
from tests.test_cv2 import CFG, ctx, pose


def body(nose=(0.5, 0.35), ly=0.6, ry=0.6, sw=0.3):
    p = pose(shoulder_y=0.6, nose=nose)
    p[P.SHOULDER_L] = Landmark(0.5 + sw / 2, ly, 0, 0.99)
    p[P.SHOULDER_R] = Landmark(0.5 - sw / 2, ry, 0, 0.99)
    return p


class TestPosture(unittest.TestCase):
    def setUp(self):
        self.geo, self.sa, self.t = BodyGeometry(CFG), SlumpAnalyzer(CFG), 0.0
        self.play(6, lambda s: body())                       # calibrare: drept

    def play(self, sec, f):
        c = None
        for _ in range(int(sec * 20)):
            self.t += 0.05
            c = ctx(self.t, pose=f(self.t))
            self.geo.update(c)
            self.sa.update(c)
        return c

    def on(self, c, name):
        return c.detections[name].active

    def test_upright_is_ok(self):
        c = self.play(3, lambda s: body())
        for name in ("slumped_side", "bad_posture", "out_of_position", "head_on_wheel"):
            self.assertFalse(self.on(c, name), name)
        self.assertEqual(c.signals["slump"]["state"], "upright")

    def test_fallen_to_side_and_still(self):
        c = self.play(5, lambda s: body(nose=(0.32, 0.40), ly=0.66, ry=0.54))
        self.assertTrue(self.on(c, "slumped_side"), c.signals["slump"])
        self.assertEqual(c.signals["slump"]["direction"], "side")
        self.assertEqual(c.signals["slump"]["state"], "slumped")

    def test_leaning_to_side_while_moving_is_not_fallen(self):
        c = self.play(5, lambda s: body(nose=(0.32 + 0.08 * math.sin(4 * s), 0.40), ly=0.66, ry=0.54))
        self.assertFalse(self.on(c, "slumped_side"))

    def test_slight_lean_is_bad_posture(self):
        c = self.play(3, lambda s: body(nose=(0.58, 0.36)))
        self.assertTrue(self.on(c, "bad_posture"), c.signals["slump"])
        self.assertFalse(self.on(c, "slumped_side"))

    def test_out_of_position(self):
        c = self.play(1, lambda s: None)                      # nu se mai vede nimeni
        self.assertTrue(self.on(c, "out_of_position"))
        self.assertEqual(c.signals["slump"]["out"], "lost")
        self.play(3, lambda s: body())
        c = self.play(6, lambda s: body(nose=(0.5, 0.47), sw=0.15))   # mult mai departe de cameră
        self.assertTrue(self.on(c, "out_of_position"), c.signals["slump"])


class TestBreathingOff(unittest.TestCase):
    def test_pipeline_has_no_breathing_by_default(self):
        from vigil_vision.body.breathing import BreathingAnalyzer
        from vigil_vision.config import load_config
        cfg = load_config()
        self.assertFalse(cfg["breathing"]["enabled"])

        names = [type(a).__name__ for a in _analyzers(cfg)]
        self.assertNotIn(BreathingAnalyzer.__name__, names)
        cfg["breathing"]["enabled"] = True
        self.assertIn(BreathingAnalyzer.__name__, [type(a).__name__ for a in _analyzers(cfg)])


def _analyzers(cfg):
    """Analizorii pipeline-ului fără să pornească MediaPipe / YOLO / camera."""
    import vigil_vision.pipeline as pl
    orig = pl._try_build
    pl._try_build = lambda f, what: None
    try:
        class Cam:
            def __init__(self, **kw): pass
        class Pub:
            def __init__(self, *a, **kw): pass
        return pl.Pipeline(cfg, publisher=Pub(), camera=Cam()).analyzers
    finally:
        pl._try_build = orig


if __name__ == "__main__":
    unittest.main()
