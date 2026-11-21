"""Ce trimite acum Vision de pe laptopurile 2 și 3 (merge.py, postura din body/slump.py, respirația oprită)
și cum le folosesc regulile, fuziunea și matricea de probabilitate. Plus pauza întrebării ML1 cât vorbește Virgil."""

import unittest

from ml1 import probability as prob
from ml1.actions import PHRASES
from ml1.cv_adapter import normalize_vision
from ml1.engine import DecisionEngine
from ml1.fusion import AttentionFusion
from ml1.mqtt_bridge import TOPIC_ASSISTANT_REPLY, Bridge
from ml1.rules import fallback_decision, rule_severity
from ml1.validate import decide


def ev(name, dur=4.0, conf=0.85, **attrs):
    return {"event": name, "duration_sec": dur, "confidence": conf, **attrs}


def ctx(a=0.8, answer=None, **vision):
    return {"fusion": {"a": a}, "vision": {"events": [], **vision}, "driver_response": answer}


def merged(t, face=True, body=True, **extra):
    """Un mesaj vigil/vision ca de la merge.py, șofer atent."""
    m = {"ts": t, "camera_on": face or body, "cameras": {"face": face, "body": body},
         "face_detected": face, "body_detected": body, "head_drop": False, "events": []}
    if face:
        m.update({"ear": 0.29, "perclos": 0.03, "blink_rate": 15.0, "gaze_on_road": True})
    if body:
        m.update({"phone": 0.02, "breathing": None, "posture": {"state": "upright", "still_sec": 0.0}})
    m.update(extra)
    return m


class TestTwoLaptops(unittest.TestCase):
    def test_face_laptop_offline_means_face_unknown_not_missing(self):
        m = normalize_vision(merged(1.0, face=False))
        self.assertIsNone(m["face_detected"])
        self.assertIs(m["face_camera"], False)
        self.assertIs(m["body_camera"], True)
        self.assertNotIn("cameras", m)
        # camera merge, dar fața nu se vede: rămâne „lipsă” (șoferul nu privește drumul)
        self.assertIs(normalize_vision({**merged(1.0), "face_detected": False})["face_detected"], False)

    def test_long_face_dropout_is_not_inattention(self):
        def run(with_cameras):
            f = AttentionFusion()
            for i in range(5 * 60):
                t = i / 5
                m = merged(t, face=not 20 <= t < 55)
                if not with_cameras:
                    m.pop("cameras")
                r = f.update(normalize_vision(m))
                if t == 50.0:
                    return r
        self.assertNotIn("gaze", run(True).contributions)          # necunoscut, nu 0
        self.assertEqual(run(False).contributions["gaze"]["score"], 0.0)   # vechiul comportament
        self.assertGreater(run(True).a, 0.9)

    def test_camera_coverage(self):
        f = AttentionFusion()
        both = f.update(normalize_vision(merged(1.0)))
        face = AttentionFusion().update(normalize_vision(merged(1.0, body=False)))
        body = AttentionFusion().update(normalize_vision(merged(1.0, face=False)))
        off = AttentionFusion().update({"ts": 1.0, "camera_on": False})
        self.assertGreater(both.camera_cov, 0.9)
        self.assertGreater(face.camera_cov, body.camera_cov)
        self.assertGreater(body.camera_cov, prob.MIN_COV)            # corpul singur rămâne o observație
        self.assertEqual(off.camera_cov, 0.0)
        self.assertIn("camera_cov", both.to_dict())


class TestPostureEvents(unittest.TestCase):
    def test_slumped_side_is_emergency(self):
        self.assertEqual(rule_severity(ctx(events=[ev("slumped_side", 3)]))[0], 3)
        self.assertNotEqual(rule_severity(ctx(events=[ev("slumped_side", 1)]))[0], 3)
        posture = normalize_vision({"posture": {"state": "slumped", "direction": "side", "still_sec": 4}})
        self.assertEqual(posture["posture_direction"], "side")

    def test_bad_posture_and_out_of_position_are_reminders(self):
        d = fallback_decision(ctx(events=[ev("bad_posture", 12, 0.6)]))
        self.assertEqual((d["severity"], d["actions"][0]["args"]["text"]), (1, PHRASES["remind_posture"]))
        d = fallback_decision(ctx(events=[ev("out_of_position", 4, 0.7)]))
        self.assertEqual((d["severity"], d["actions"][0]["args"]["text"]), (1, PHRASES["remind_position"]))
        # telefonul în mână are prioritate la memento
        d = fallback_decision(ctx(phone=0.9, events=[ev("bad_posture", 12, 0.6)]))
        self.assertEqual(d["actions"][0]["args"]["text"], PHRASES["remind_phone"])

    def test_llm_posture_reminder_is_kept_and_wrong_one_replaced(self):
        c = ctx(events=[ev("bad_posture", 12, 0.6)])
        d = decide(c, lambda _: '{"severity": 1, "actions": ["remind_road"], "reason": "postură"}', 5.0)
        self.assertEqual(d["severity"], 1)                            # nu coborât la 0 (semnul există)
        self.assertEqual([a["args"]["text"] for a in d["actions"]], [PHRASES["remind_posture"]])
        d = decide(ctx(), lambda _: '{"severity": 0, "actions": ["remind_posture"], "reason": "x"}', 5.0)
        self.assertEqual(d["actions"], [])                            # fără semn: reminderul dispare


class TestChestWithoutBreathing(unittest.TestCase):
    def test_hand_on_chest_needs_longer_hold_when_breathing_is_off(self):
        chest = lambda dur: [ev("hand_on_chest", dur, 0.8)]
        self.assertNotEqual(rule_severity(ctx(events=chest(4)))[0], 3)            # breathing None (Vision implicit)
        sev, reason = rule_severity(ctx(events=chest(6.5)))
        self.assertEqual(sev, 3)
        self.assertIn("nu e măsurată", reason)
        # respirația măsurată și normală: mâna singură nu e urgență
        self.assertNotEqual(rule_severity(ctx(breathing_state="normal", events=chest(10)))[0], 3)
        # respirația grea confirmă deja de la 3 s
        self.assertEqual(rule_severity(ctx(breathing_state="heavy", events=chest(3.5)))[0], 3)
        # un „sunt bine” coerent coboară la 2, ca la mâna pe piept cu respirație grea
        ok = {"answered": True, "latency_sec": 1.2, "text": "da, sunt bine"}
        self.assertEqual(rule_severity(ctx(events=chest(8), answer=ok))[0], 2)


class TestEvidencePerSecond(unittest.TestCase):
    def test_weight(self):
        w = prob.evidence_weight
        self.assertEqual(w(None, 5, 1.0), 1.0)                 # primul cadru
        self.assertAlmostEqual(w(0.1, 5, 1.0), 0.5)            # 10 Hz: jumătate de observație
        self.assertEqual(w(1.0, 5, 1.0), 1.0)                  # o pauză nu e mai multă evidență
        self.assertEqual(w(0.2, 5, 0.0), 0.0)                  # camera oprită
        self.assertLess(w(0.2, 5, 0.44), 1.0)                  # doar corpul: parțial
        self.assertEqual(w(0.2, 5, None, 0.2), 0.0)            # apelant vechi: pragul pe încredere
        self.assertEqual(w(0.2, 5, None, 0.6), 1.0)

    def _belief(self, hz, per_second, calm=6.0, seconds=1.0, rule=0, a=0.6):
        """calm s de șofer atent, apoi `seconds` s de semn slab (a mediu, reguli 0), ca filtrul să nu fie saturat."""
        f = prob.StateFilter(prob.ProbModel.default(), per_second=per_second)
        for i in range(int(calm * hz)):
            f.update(0, 0.95, 0.6, i / hz, 1.0 if per_second else None)
        for i in range(int(seconds * hz)):
            b = f.update(rule, a, 0.6, calm + i / hz, 1.0 if per_second else None)
        return b

    def test_same_belief_at_5_10_and_30_hz(self):
        b5, b10, b30 = (self._belief(hz, True) for hz in (5, 10, 30))
        for b in (b10, b30):
            self.assertLess(max(abs(x - y) for x, y in zip(b, b5)), 0.05)
        old5, old30 = self._belief(5, False), self._belief(30, False)
        self.assertGreater(max(abs(x - y) for x, y in zip(old5, old30)), 0.05)   # vechiul filtru depinde de rată

    def test_body_only_collapse_raises_p_emergency(self):
        def run(per_second):
            f = prob.StateFilter(prob.ProbModel.default(), per_second=per_second)
            for i in range(50):
                f.update(0, 0.95, 0.6, i / 10, 1.0 if per_second else None)
            for i in range(30):   # 3 s: laptopul feței căzut, corpul vede prăbușirea (încredere 0.27 < 0.3)
                b = f.update(3, 0.3, 0.27, 5 + i / 10, 0.44 if per_second else None)
            return b[3]
        self.assertGreater(run(True), 0.5)
        self.assertLess(run(False), 0.1)                      # filtrul vechi ignora cadrele cu o singură cameră


def tired(t):
    return {"ts": t, "camera_on": True, "face_detected": True, "ear": 0.24, "perclos": 0.35, "blink_rate": 9.0,
            "gaze_on_road": True, "head_drop": False, "phone": 0.0,
            "events": [ev("microsleep", 1.0, 0.8)]}


class TestVirgilPausesSilence(unittest.TestCase):
    def first_emergency(self, busy):
        eng = DecisionEngine(calibrate=False)
        asked = sev3 = None
        for i in range(5 * 45):
            t = i / 5
            if busy and t == 1.0:
                eng.on_assistant({"request": "weather", "status": "working"})
            if busy and t == 21.0:
                eng.on_assistant({"request": "weather", "status": "done"})
            for d in eng.on_vision(tired(t)):
                if asked is None and any(a["tool"] == "ask_driver" for a in d["actions"]):
                    asked = t
                if sev3 is None and d["severity"] == 3:
                    sev3 = t
        return asked, sev3

    def test_question_waiting_behind_virgil_is_not_silence(self):
        asked, sev3 = self.first_emergency(busy=False)
        self.assertEqual(asked, 0.0)
        self.assertLess(sev3, 12.0)                          # fără Virgil: tăcere 8 s -> urgență
        asked, sev3 = self.first_emergency(busy=True)
        # tăcerea se numără abia după ce Virgil termină („done” primește ora ultimului cadru, 20.8 s)
        self.assertGreaterEqual(sev3, 20.8 + 8.0)

    def test_lost_done_message_does_not_block_forever(self):
        eng = DecisionEngine(calibrate=False)
        eng.on_vision(tired(0.0))
        eng.on_assistant({"status": "speaking"})
        self.assertTrue(eng._assistant_busy(10.0))
        self.assertFalse(eng._assistant_busy(200.0))

    def test_bridge_routes_assistant_reply(self):
        sent = []
        br = Bridge(DecisionEngine(calibrate=False), lambda topic, obj: sent.append(topic))
        br.on_message(TOPIC_ASSISTANT_REPLY, b'{"request": "time", "status": "working"}')
        self.assertEqual(br.received["assistant"], 1)
        self.assertIsNotNone(br.engine._assistant_since)


if __name__ == "__main__":
    unittest.main()
