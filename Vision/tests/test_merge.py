"""Teste pentru vigil_vision.merge (combinarea camerelor față + corp). Rulare din Vision/: python -m unittest discover -s tests"""
import unittest

from vigil_vision.merge import STALE_SEC, merge

FACE = {"ear": 0.27, "perclos": 0.04, "blink_rate": 16.0, "gaze_on_road": True, "head_drop": False, "phone": 0.0,
        "events": [{"event": "yawn", "duration_sec": 2.0, "confidence": 0.7}], "landmarks": [[0.1, 0.2]],
        "face_detected": True, "body_detected": False, "fps": 24.0}
BODY = {"ear": None, "perclos": None, "phone": 0.8, "head_drop": False, "breathing": {"state": "heavy"},
        "posture": {"state": "upright"}, "body_landmarks": [[0.5, 0.5]], "face_detected": False,
        "body_detected": True, "fps": 15.0,
        "events": [{"event": "hand_on_chest", "duration_sec": 4.0, "confidence": 0.8}]}


class TestMerge(unittest.TestCase):
    def both(self, now=10.0, face=FACE, body=BODY):
        return merge({"face": (now, face), "body": (now, body)}, now + 0.1, wall=123.0)

    def test_each_camera_gives_its_own_fields(self):
        m = self.both()
        self.assertEqual(m["ear"], 0.27)                 # nu None-ul din mesajul corpului
        self.assertEqual(m["phone"], 0.8)                # nu 0.0 din mesajul feței
        self.assertTrue(m["face_detected"] and m["body_detected"])
        self.assertEqual(m["landmarks"], [[0.1, 0.2]])
        self.assertEqual(m["body_landmarks"], [[0.5, 0.5]])
        self.assertEqual(m["breathing"], {"state": "heavy"})
        self.assertEqual(m["fps"], 15.0)
        self.assertTrue(m["camera_on"])

    def test_events_from_both_are_listed(self):
        names = sorted(e["event"] for e in self.both()["events"])
        self.assertEqual(names, ["hand_on_chest", "yawn"])

    def test_same_event_in_both_keeps_longest(self):
        f = {**FACE, "events": [{"event": "head_on_wheel", "duration_sec": 2.0, "confidence": 0.6}]}
        b = {**BODY, "events": [{"event": "head_on_wheel", "duration_sec": 5.0, "confidence": 0.5}]}
        events = self.both(face=f, body=b)["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["duration_sec"], events[0]["confidence"]), (5.0, 0.6))

    def test_head_drop_if_either_sees_it(self):
        self.assertTrue(self.both(body={**BODY, "head_drop": True})["head_drop"])
        self.assertTrue(self.both(face={**FACE, "head_drop": True})["head_drop"])
        self.assertFalse(self.both()["head_drop"])

    def test_silent_body_camera_drops_only_body_fields(self):
        m = merge({"face": (10.0, FACE), "body": (10.0 - STALE_SEC - 1, BODY)}, 10.1)
        self.assertEqual(m["ear"], 0.27)
        self.assertNotIn("phone", m)
        self.assertNotIn("breathing", m)
        self.assertFalse(m["body_detected"])
        self.assertTrue(m["camera_on"])                  # fața merge
        self.assertEqual([e["event"] for e in m["events"]], ["yawn"])

    def test_silent_face_camera_means_face_not_detected(self):
        m = merge({"face": (0.0, FACE), "body": (10.0, BODY)}, 10.1)
        self.assertNotIn("ear", m)
        self.assertFalse(m["face_detected"])             # ML1: fața lipsă, nu valori inventate
        self.assertEqual(m["phone"], 0.8)

    def test_both_silent_or_missing(self):
        m = merge({"face": (0.0, FACE)}, 50.0)
        self.assertFalse(m["camera_on"])
        self.assertEqual(m["events"], [])
        self.assertFalse(m["head_drop"])
        self.assertFalse(merge({}, 1.0)["camera_on"])

    def test_malformed_events_are_ignored(self):
        m = self.both(face={**FACE, "events": [None, "x", {"nume": 1}]})
        self.assertEqual([e["event"] for e in m["events"]], ["hand_on_chest"])


if __name__ == "__main__":
    unittest.main()
