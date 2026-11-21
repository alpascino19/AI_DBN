"""Filtrele împotriva alarmelor false pentru mâna pe piept (body/chest.py)."""
import copy
import unittest

from vigil_vision.body.chest import ChestAnalyzer
from vigil_vision.context import FaceOutput, Hand, Landmark, P

from tests.test_cv2 import CFG, ctx, hand_at, pose

CHIN, MOUTH = (0.5, 0.48), (0.5, 0.44)   # pose(): nas (0.5, 0.35), umeri la y = 0.6, unit ≈ 0.3


def hand_on_chin():
    """Degetele pe bărbie, palma mai jos (la gât / începutul pieptului)."""
    lms = [Landmark(0.5, 0.72) for _ in range(21)]          # încheietura și palma jos
    for i in (5, 9, 13, 17):
        lms[i] = Landmark(0.5, 0.66)                         # bazele degetelor (centrul palmei intră în zonă)
    for i in (4, 8, 12, 16, 20):
        lms[i] = Landmark(0.5, 0.49)                         # vârfurile pe bărbie
    return Hand(lms, "Right", 0.95)


def face(chin=CHIN, mouth=MOUTH):
    return FaceOutput(signals={"chin": chin, "mouth": mouth}, detected=True)


class TestChestFilters(unittest.TestCase):
    def test_hand_on_chin_is_not_chest(self):
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_on_chin()], face=face())
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
        self.assertEqual(c.signals["chest_reject"], "face")

    def test_hand_on_chin_without_cv1_uses_pose_mouth(self):
        p = pose()
        p[P.MOUTH_L] = Landmark(0.53, 0.44, 0, 0.9)
        p[P.MOUTH_R] = Landmark(0.47, 0.44, 0, 0.9)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[hand_on_chin()])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)

    def test_hand_on_heart_still_detected_with_face_known(self):
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[hand_at(0.6, 0.75)], face=face())
        a.update(c)
        self.assertTrue(c.detections["hand_on_chest"].active)

    def test_hand_held_on_chest_stays_active(self):
        a = ChestAnalyzer(CFG)
        for i in range(30):                                  # 1 s, mâna stă
            c = ctx(t=i / 30, pose=pose(), hands=[hand_at(0.6, 0.75)])
            a.update(c)
            self.assertTrue(c.detections["hand_on_chest"].active)

    def test_hand_passing_through_is_not_chest(self):
        a = ChestAnalyzer(CFG)
        actives = []
        for i in range(30):                                  # mâna traversează zona în 1 s
            x = 0.2 + 0.6 * i / 29
            c = ctx(t=i / 30, pose=pose(), hands=[hand_at(x, 0.75)])
            a.update(c)
            actives.append(c.detections["hand_on_chest"].active)
        self.assertLessEqual(sum(actives[3:]), 1)            # (primul cadru n-are istoric)

    def test_hand_mostly_outside_zone(self):
        lms = [Landmark(0.6, 1.05) for _ in range(21)]       # degetele în afara zonei...
        for i in (0, 5, 9, 13, 17):
            lms[i] = Landmark(0.6, 0.75)                     # ...doar palma în zonă (5/21 puncte)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=pose(), hands=[Hand(lms, "Right", 0.9)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
        self.assertEqual(c.signals["chest_reject"], "partial")

    def test_hand_in_the_air_in_front_of_chest(self):
        p = pose(wrist=(0.6, 0.78))
        p[P.WRIST_L] = Landmark(0.6, 0.78, -0.5, 0.95)      # mult mai aproape de cameră decât umerii
        p[P.WRIST_R] = Landmark(0.6, 0.78, -0.5, 0.95)
        # implicit filtrul e oprit (z-ul din Pose e zgomotos): mâna e acceptată, adâncimea e raportată
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[hand_at(0.6, 0.75)])
        a.update(c)
        self.assertTrue(c.detections["hand_on_chest"].active)
        self.assertGreater(c.signals["chest_depth"], 1.0)
        # pornit explicit din config -> respinsă
        cfg = copy.deepcopy(CFG)
        cfg["chest"]["depth_max"] = 1.0
        a = ChestAnalyzer(cfg)
        c = ctx(pose=p, hands=[hand_at(0.6, 0.75)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
        self.assertEqual(c.signals["chest_reject"], "depth")


if __name__ == "__main__":
    unittest.main()


def spread_hand(cx, cy, palm):
    """Mână cu palma de lungime `palm` (încheietura jos, degetele sus), centrată în (cx, cy)."""
    lms = [Landmark(cx, cy) for _ in range(21)]
    lms[0] = Landmark(cx, cy + palm / 2)                     # încheietura
    lms[9] = Landmark(cx, cy - palm / 2)                     # baza mijlociului
    lms[5] = Landmark(cx - palm * 0.4, cy - palm / 2)
    lms[17] = Landmark(cx + palm * 0.4, cy - palm / 2 + 0.01)
    lms[13] = Landmark(cx + palm * 0.2, cy - palm / 2)
    return Hand(lms, "Right", 0.95)


class TestHandNearCamera(unittest.TestCase):
    def test_normal_size_hand_on_chest(self):
        a = ChestAnalyzer(CFG)                               # unit ≈ 0.3 -> palma 0.09 = 0.3 × unit
        c = ctx(pose=pose(), hands=[spread_hand(0.6, 0.75, 0.09)])
        a.update(c)
        self.assertTrue(c.detections["hand_on_chest"].active)
        self.assertLess(c.signals["chest_hand_scale"], 0.45)

    def test_big_hand_close_to_camera_is_rejected(self):
        a = ChestAnalyzer(CFG)                               # palma 0.2 = 0.67 × unit -> lângă cameră
        c = ctx(pose=pose(), hands=[spread_hand(0.55, 0.74, 0.2)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
        self.assertEqual(c.signals["chest_reject"], "close")


class TestPoseDoesNotOverrideHands(unittest.TestCase):
    def test_close_hand_rejected_even_if_pose_wrist_in_zone(self):
        """Hands vede mâna mare (lângă cameră) -> respinsă; Pose (grosier) nu o mai poate „salva”."""
        p = pose()
        for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
            p[w] = Landmark(0.56, 0.80, 0, 0.9)
            p[i] = Landmark(0.57, 0.74, 0, 0.9)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[spread_hand(0.55, 0.74, 0.2)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
        self.assertEqual(c.signals["chest_reject"], "close")

    def test_pose_still_used_when_hands_sees_nothing(self):
        p = pose()
        for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
            p[w] = Landmark(0.6, 0.78, 0, 0.9)
            p[i] = Landmark(0.6, 0.72, 0, 0.9)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[])
        a.update(c)
        self.assertTrue(c.detections["hand_on_chest"].active)


class TestOtherArmPose(unittest.TestCase):
    def test_other_arm_pose_wrist_does_not_trigger_when_hands_sees_a_hand(self):
        """Mâna mare lângă cameră + încheietura celuilalt braț (ghicită de Pose) în zonă -> nu."""
        p = pose()
        p[P.WRIST_L] = Landmark(0.62, 0.84, 0, 0.7)
        p[P.INDEX_L] = Landmark(0.62, 0.80, 0, 0.7)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[spread_hand(0.45, 0.74, 0.2)])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)

    def test_pose_points_outside_image_are_ignored(self):
        p = pose()
        for w, i in ((P.WRIST_L, P.INDEX_L), (P.WRIST_R, P.INDEX_R)):
            p[w] = Landmark(0.6, 1.02, 0, 0.9)                 # sub marginea imaginii
            p[i] = Landmark(0.6, 0.86, 0, 0.9)
        a = ChestAnalyzer(CFG)
        c = ctx(pose=p, hands=[])
        a.update(c)
        self.assertFalse(c.detections["hand_on_chest"].active)
