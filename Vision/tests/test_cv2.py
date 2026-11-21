"""Teste CV2 cu date sintetice (fără cameră, MediaPipe sau YOLO).

    python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from vigil_vision.body.breathing import BreathingAnalyzer
from vigil_vision.body.chest import ChestAnalyzer
from vigil_vision.body.phone import PhoneAnalyzer
from vigil_vision.camera import Frame
from vigil_vision.config import load_config
from vigil_vision.context import BBox, Detection, FrameContext, Hand, Landmark, P
from vigil_vision.events.tracker import EventTracker

CFG = load_config()
# testele de bază folosesc un corp văzut din față, fără oglindire
CFG["view"]["mode"] = "front"
CFG["camera"]["mirror"] = False
CFG_34 = load_config()                       # cameră la ~45°, în dreapta șoferului
CFG_34["view"].update(mode="three_quarter", camera_side="right")
CFG_34["camera"]["mirror"] = False
IMG = np.zeros((480, 640, 3), np.uint8)


def pose(shoulder_y=0.6, wrist=(0.2, 0.95), nose=(0.5, 0.35)):
    lms = [Landmark(0.5, 0.5, 0, 0.0) for _ in range(33)]
    lms[P.NOSE] = Landmark(*nose, 0, 0.99)
    lms[P.SHOULDER_L] = Landmark(0.65, shoulder_y, 0, 0.99)
    lms[P.SHOULDER_R] = Landmark(0.35, shoulder_y, 0, 0.99)
    for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
        lms[w] = Landmark(*wrist, 0, 0.9)
        lms[i] = Landmark(*wrist, 0, 0.9)
    return lms


def hand_at(x, y):
    return Hand([Landmark(x, y) for _ in range(21)], "Right", 0.95)


def ctx(t=0.0, **kw):
    c = FrameContext(Frame(IMG, int(t * 30), t, 640, 480), t)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestTracker(unittest.TestCase):
    def test_duration_gap_and_min_duration(self):
        tr = EventTracker(CFG)
        on = {"hand_on_chest": Detection("hand_on_chest", True, 0.9)}
        off = {"hand_on_chest": Detection("hand_on_chest", False)}
        t = 0.0
        while t < 1.5:                                  # sub min 2 s -> nimic
            self.assertEqual(tr.update(on, t), [])
            t += 0.1
        tr.update(off, t + 0.5)                         # pauză 0.5 s < gap 1 s
        t += 0.5
        while t < 4.0:
            ev = tr.update(on, t)
            t += 0.1
        self.assertEqual(ev[0].event, "hand_on_chest")
        self.assertAlmostEqual(ev[0].duration_sec, 3.9, delta=0.15)
        self.assertGreater(ev[0].confidence, 0.6)
        self.assertEqual(tr.update(off, t + 1.5), [])   # pauză > gap -> se încheie
        self.assertEqual(tr.ended[-1]["event"], "hand_on_chest")


class TestChest(unittest.TestCase):
    def test_hand_on_chest(self):
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_at(0.6, 0.72)])
        a.update(c)
        d = c.detections["hand_on_chest"]
        self.assertTrue(d.active)
        self.assertEqual(d.attrs["side"], "left")
        self.assertGreater(d.confidence, 0.6)

    def test_hand_on_wheel_is_not_chest(self):
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_at(0.3, 0.97)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)

    def test_phone_held_at_chest_is_not_chest(self):
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_at(0.5, 0.7)],
                objects=[BBox("cell phone", 0.8, 0.45, 0.62, 0.55, 0.78)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)

    def test_no_pose(self):
        a = ChestAnalyzer(CFG)
        c = ctx()
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)


class TestPhone(unittest.TestCase):
    box = BBox("cell phone", 0.8, 0.40, 0.50, 0.50, 0.65)

    def test_phone_in_hand_looking(self):
        a = PhoneAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_at(0.45, 0.6)], objects=[self.box],
                signals={"gaze_on_road": False})
        a.update(c)
        d = c.detections["phone_use"]
        self.assertTrue(d.active)
        self.assertTrue(d.attrs["in_hand"])
        self.assertEqual(d.attrs["usage"], "looking")
        self.assertGreater(c.signals["phone"], 0.6)

    def test_phone_in_mount_not_used(self):
        a = PhoneAnalyzer(CFG)
        c = ctx(pose=pose(), objects=[BBox("cell phone", 0.8, 0.85, 0.85, 0.95, 0.98)])
        a.update(c)
        self.assertFalse(c.detections["phone_use"].active)
        self.assertLess(c.signals["phone"], CFG["phone"]["min_score"])


class TestBreathing(unittest.TestCase):
    def run_signal(self, a, t0, dur, rate_bpm, amp, jaw=0.0, fps=20):
        n = int(dur * fps)
        for k in range(n):
            t = t0 + k / fps
            y = 0.6 + amp * math.sin(2 * math.pi * rate_bpm / 60 * t)
            c = ctx(t, pose=pose(shoulder_y=y), signals={"jaw_open": jaw})
            a.update(c)
        return c

    def test_calibrate_then_heavy(self):
        a = BreathingAnalyzer(CFG)
        c = self.run_signal(a, 0, 35, 14, 0.004)          # calibrare: respirație normală
        self.assertEqual(c.signals["breathing"]["state"], "normal")
        self.assertTrue(c.signals["breathing"]["calibrated"])
        c = self.run_signal(a, 35, 12, 32, 0.015, jaw=0.6)  # rapidă, amplă, gura deschisă
        b = c.signals["breathing"]
        self.assertEqual(b["state"], "heavy", b)
        self.assertAlmostEqual(b["rate_bpm"], 32, delta=5)
        self.assertTrue(c.detections["heavy_breathing"].active)


class TestJitter(unittest.TestCase):
    """MediaPipe tremură: zona pieptului trebuie să stea pe loc și respirația
    calmă să nu devină 'heavy' (bug-ul raportat la breathing)."""

    def test_zone_stable_and_no_false_heavy(self):
        import random
        from vigil_vision.smoothing import PoseSmoother
        random.seed(0)
        a, ch, sm = BreathingAnalyzer(CFG), ChestAnalyzer(CFG), PoseSmoother()
        heights, heavy = [], 0
        for k in range(70 * 20):                       # 70 s calm, 20 fps
            t = k / 20
            p = pose(shoulder_y=0.6 + 0.002 * math.sin(2 * math.pi * 14 / 60 * t))
            for i in (P.SHOULDER_L, P.SHOULDER_R, P.NOSE, P.HIP_L, P.HIP_R):
                p[i].x += random.gauss(0, 0.006)
                p[i].y += random.gauss(0, 0.006)
            p[P.HIP_L].visibility = p[P.HIP_R].visibility = random.choice([0.45, 0.55, 0.85])
            c = ctx(t, pose=sm(p, t))
            a.update(c)
            ch.update(c)
            z = c.signals["chest_zone"]
            heights.append(z[3] - z[1])
            heavy += c.signals["breathing"]["state"] == "heavy"
        h = np.array(heights[40:])
        self.assertLess(np.abs(np.diff(h)).max(), 0.005)   # înainte: ~0.047
        self.assertEqual(heavy, 0)

    def test_body_sway_is_not_heavy(self):
        a = BreathingAnalyzer(CFG)
        for k in range(35 * 20):
            t = k / 20
            a.update(ctx(t, pose=pose(shoulder_y=0.6 + 0.002 * math.sin(2 * math.pi * 0.23 * t))))
        for k in range(15 * 20):                        # tot corpul se mișcă (și nasul)
            t = 35 + k / 20
            d = 0.012 * math.sin(2 * math.pi * 0.4 * t)
            c = ctx(t, pose=pose(shoulder_y=0.6 + d, nose=(0.5, 0.35 + d)))
            a.update(c)
            self.assertNotEqual(c.signals["breathing"]["state"], "heavy")


class TestBreathingFlow(unittest.TestCase):
    """Optical flow pe un video sintetic: tricou cu textură care se mișcă."""

    @classmethod
    def setUpClass(cls):
        import cv2
        cls.cv2 = cv2
        rng = np.random.default_rng(0)
        tex = lambda s: cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (0, 0), s)
        cls.bg, cls.shirt, cls.head, cls.rng = tex(3), tex(2.5), tex(2), rng

    def frame(self, t, chest_dy, head_dy=0.0):
        cv2 = self.cv2
        sh = lambda im, dy: cv2.warpAffine(im, np.float32([[1, 0, 0], [0, 1, dy]]), (640, 480),
                                           borderMode=cv2.BORDER_REFLECT)
        img = self.bg.copy()
        img[264:, 160:480] = sh(self.shirt, chest_dy)[264:, 160:480]
        img[96:230, 256:384] = sh(self.head, head_dy)[96:230, 256:384]
        img = np.clip(img + self.rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        return Frame(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), int(t * 20), t, 640, 480)

    def run_phase(self, ch, br, t0, sec, chest_fn, head_fn=lambda t: 0.0):
        states = []
        for k in range(int(sec * 20)):
            t = t0 + k / 20
            c = FrameContext(self.frame(t, chest_fn(t), head_fn(t)), t, pose=pose())
            ch.update(c)
            br.update(c)
            states.append(c.signals["breathing"]["state"])
        return states, c.signals["breathing"]

    def test_flow_heavy_and_sway(self):
        ch, br = ChestAnalyzer(CFG), BreathingAnalyzer(CFG)
        normal = lambda t: 1.0 * math.sin(2 * math.pi * 14 / 60 * t)
        st, b = self.run_phase(ch, br, 0, 32, normal)
        self.assertEqual(b["source"], "flow")
        self.assertNotIn("heavy", st)
        st, b = self.run_phase(ch, br, 32, 10, lambda t: 4 * math.sin(2 * math.pi * 0.5 * t))
        self.assertEqual(b["state"], "heavy", b)
        self.assertAlmostEqual(b["rate_bpm"], 30, delta=4)
        self.run_phase(ch, br, 42, 8, normal)
        sway = lambda t: 5 * math.sin(2 * math.pi * 0.4 * t)
        st, b = self.run_phase(ch, br, 50, 10, sway, head_fn=sway)   # tot corpul se mișcă
        self.assertNotIn("heavy", st[60:], b)


class TestSlump(unittest.TestCase):
    """Capul pe volan: prăbușit + nemișcat -> head_on_wheel; aplecat dar în mișcare -> nu."""

    def body(self, nose_y, sh_y=0.6, sw=0.3):
        p = pose(shoulder_y=sh_y, nose=(0.5, nose_y))
        p[P.SHOULDER_L].x, p[P.SHOULDER_R].x = 0.5 + sw / 2, 0.5 - sw / 2
        return p

    def run_phase(self, sa, tr, t0, sec, f):
        hits = []
        for k in range(int(sec * 20)):
            t = t0 + k / 20
            p = f(k / 20)
            c = ctx(t, pose=p)
            sa.update(c)
            hits.append(any(e.event == "head_on_wheel" for e in tr.update(c.detections, t)))
        return hits, c

    def test_collapse_moving_and_pose_lost(self):
        from vigil_vision.body.slump import SlumpAnalyzer
        sa, tr = SlumpAnalyzer(CFG), EventTracker(CFG)
        hits, _ = self.run_phase(sa, tr, 0, 31, lambda t: self.body(0.35))       # calibrare
        self.assertFalse(any(hits))
        self.assertTrue(sa.base_gap)
        # se apleacă după ceva de jos, dar se mișcă -> NU
        hits, _ = self.run_phase(sa, tr, 31, 6, lambda t: self.body(
            0.55 + 0.06 * math.sin(2 * math.pi * 0.7 * t), sh_y=0.63 + 0.03 * math.sin(2 * math.pi * 0.7 * t)))
        self.assertFalse(any(hits))
        self.run_phase(sa, tr, 37, 3, lambda t: self.body(0.35))
        # cade pe volan și rămâne nemișcat -> DA, în câteva secunde
        hits, c = self.run_phase(sa, tr, 40, 8, lambda t: self.body(0.35 + 0.29 * min(t / 1.5, 1), 0.63, 0.35))
        self.assertTrue(hits[-1])
        self.assertLess(hits.index(True) / 20, 7.0)
        self.assertTrue(c.detections["head_on_wheel"].active)
        # Pose pierde persoana -> starea se păstrează
        hits, c = self.run_phase(sa, tr, 48, 6, lambda t: None)
        self.assertTrue(all(hits))
        self.assertTrue(c.detections["head_on_wheel"].attrs["pose_lost"])
        # își revine
        hits, c = self.run_phase(sa, tr, 54, 5, lambda t: self.body(0.35))
        self.assertFalse(hits[-1])


def pose_34(shoulder_y=0.6, nose=(0.62, 0.36), wrist=(0.3, 0.95), lean=0.0):
    """Șofer văzut la ~45° din dreapta lui (imagine normală): privește spre dreapta
    imaginii; umărul drept (apropiat, z mic) e în stânga, cel stâng e parțial ascuns."""
    p = pose(shoulder_y=shoulder_y, nose=nose, wrist=wrist)
    p[P.SHOULDER_R] = Landmark(0.42, shoulder_y + 0.01, -0.2, 0.99)   # apropiat
    p[P.SHOULDER_L] = Landmark(0.58, shoulder_y - 0.01, 0.1, 0.6)     # depărtat, parțial ascuns
    return p


class TestThreeQuarter(unittest.TestCase):
    """Camera la 45° în dreapta șoferului: umerii par apropiați pe x."""

    def test_geometry(self):
        from vigil_vision.body.geometry import BodyGeometry
        c = ctx(pose=pose_34())
        BodyGeometry(CFG_34).update(c)
        g = c.geom
        self.assertIsNotNone(g)
        self.assertEqual(g.facing, 1)                 # nasul e în dreapta umerilor
        self.assertEqual(g.near, P.SHOULDER_R)        # z mai mic
        self.assertGreater(g.unit, 0.15)              # nu se prăbușește ca |dx| (0.16)

    def test_chest_zone_extends_forward(self):
        a = ChestAnalyzer(CFG_34)
        c = ctx(pose=pose_34(), hands=[hand_at(0.62, 0.72)])   # mâna pe piept, în față
        a.update(c)
        x1, y1, x2, y2 = c.signals["chest_zone"]
        self.assertGreater(x2 - 0.5, 0.5 - x1)        # zona se întinde mai mult în față
        self.assertGreater(y2 - y1, 0.12)             # înălțimea nu se mai prăbușește
        d = c.detections["hand_on_chest"]
        self.assertTrue(d.active)

    def test_hand_on_wheel_not_chest(self):
        a = ChestAnalyzer(CFG_34)
        c = ctx(pose=pose_34(), hands=[hand_at(0.85, 0.92)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)

    def test_slump_three_quarter(self):
        from vigil_vision.body.slump import SlumpAnalyzer
        cfg = load_config()
        cfg["view"].update(mode="three_quarter", camera_side="right")
        cfg["camera"]["mirror"] = False
        sa, tr = SlumpAnalyzer(cfg), EventTracker(cfg)
        for k in range(31 * 20):
            t = k / 20
            c = ctx(t, pose=pose_34())
            sa.update(c)
            tr.update(c.detections, t)
        hits = []
        for k in range(8 * 20):                       # capul cade în față și în jos, apoi nemișcat
            t = 31 + k / 20
            a = min(k / 30, 1.0)
            c = ctx(t, pose=pose_34(nose=(0.62 + 0.12 * a, 0.36 + 0.26 * a), shoulder_y=0.6 + 0.04 * a))
            sa.update(c)
            hits.append(any(e.event == "head_on_wheel" for e in tr.update(c.detections, t)))
        self.assertTrue(hits[-1], c.signals["slump"])


class TestPipeline(unittest.TestCase):
    def test_process_and_message(self):
        import json
        from vigil_vision.pipeline import Pipeline

        class StubBody:
            def process(self, frame):
                return pose(), [hand_at(0.6, 0.72)]

        class StubPub:
            def publish(self, m): pass
            def close(self): pass

        pl = Pipeline(CFG, publisher=StubPub(), camera=object())
        pl.body, pl.objects = StubBody(), None
        for k in range(90):                                 # 3 s la 30 fps
            c = pl.process(Frame(IMG, k, k / 30, 640, 480))
        msg = pl.build_message(c, 30).to_dict()
        json.dumps(msg)
        self.assertEqual(msg["events"][0]["event"], "hand_on_chest")
        self.assertTrue(msg["body_detected"])
        self.assertEqual(len(msg["body_landmarks"]), 25)
        self.assertFalse(pl.build_message(None).to_dict()["camera_on"])


if __name__ == "__main__":
    unittest.main()
