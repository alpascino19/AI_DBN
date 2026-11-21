"""Integrarea cu datele reale ale echipei CV + calibrare + puntea MQTT + evaluarea."""

import json
import unittest

import numpy as np

from ml1.calibration import DriverBaseline
from ml1.cv_adapter import cv_available, cv_demo_messages, normalize_vision
from ml1.engine import DecisionEngine
from ml1.fusion import AttentionFusion
from ml1.mqtt_bridge import TOPIC_DECISION, TOPIC_RESPONSE, TOPIC_STATUS, TOPIC_VISION, Bridge
from ml1.rules import rule_severity

CV_MSG = {  # forma exactă a schemas.VisionMessage.to_dict() de la CV
    "ear": 0.29, "perclos": 0.05, "blink_rate": 15.0, "gaze_on_road": False, "head_drop": False, "phone": 0.85,
    "breathing": {"state": "normal", "rate_bpm": 15.0}, "posture": {"state": "upright", "still_sec": 0.0},
    "events": [{"event": "phone_use", "duration_sec": 4.0, "confidence": 0.85, "in_hand": True}],
    "landmarks": [[0.1, 0.2]] * 468, "body_landmarks": [[0.3, 0.4]] * 12,
    "ts": 1000.0, "camera_on": True, "face_detected": True, "body_detected": True, "fps": 25.0,
}


class TestNormalize(unittest.TestCase):
    def test_cv_event_names_and_fields(self):
        m = normalize_vision(CV_MSG)
        self.assertEqual(m["events"][0]["event"], "phone_in_hand")
        self.assertNotIn("landmarks", m)
        self.assertNotIn("body_landmarks", m)
        self.assertEqual(m["breathing_state"], "normal")
        self.assertEqual(m["posture_state"], "upright")

    def test_heavy_breathing_attached_to_chest(self):
        m = normalize_vision({"breathing": {"state": "heavy"},
                              "events": [{"event": "hand_on_chest", "duration_sec": 4, "confidence": 0.8}]})
        self.assertEqual(m["events"][0]["breathing"], "heavy")

    def test_cleaning_clips_and_drops_nonfinite(self):
        m = normalize_vision({"ear": "NaN", "perclos": 1.7, "blink_rate": -3, "phone": 2.0})
        self.assertIsNone(m["ear"])
        self.assertEqual((m["perclos"], m["blink_rate"], m["phone"]), (1.0, 0.0, 1.0))

    def test_garbage(self):
        self.assertEqual(normalize_vision(None), {})
        self.assertEqual(normalize_vision({"events": "x"})["events"], [])

    def test_phone_use_reaches_rules(self):
        ctx = {"fusion": {"a": 0.7}, "vision": normalize_vision({**CV_MSG, "phone": 0.1}), "driver_response": None}
        self.assertEqual(rule_severity(ctx), (1, "telefon în mână"))


class TestNewRules(unittest.TestCase):
    def ctx(self, **vision):
        return {"fusion": {"a": 0.6}, "vision": {"events": [], **vision}, "driver_response": None}

    def test_head_on_wheel_is_emergency(self):
        c = self.ctx(events=[{"event": "head_on_wheel", "duration_sec": 3, "confidence": 0.85}])
        self.assertEqual(rule_severity(c)[0], 3)

    def test_slumped_and_still_is_emergency(self):
        self.assertEqual(rule_severity(self.ctx(posture_state="slumped", still_sec=4))[0], 3)
        self.assertNotEqual(rule_severity(self.ctx(posture_state="slumped", still_sec=0.5))[0], 3)

    def test_eyes_closed_alone_is_microsleep_not_emergency(self):
        c = self.ctx(events=[{"event": "eyes_closed", "duration_sec": 2.5, "confidence": 0.75}])
        self.assertEqual(rule_severity(c)[0], 2)
        c["driver_response"] = {"answered": False}
        self.assertEqual(rule_severity(c)[0], 3)

    def test_breathing_state_counts_as_heavy(self):
        c = self.ctx(breathing_state="heavy", events=[{"event": "hand_on_chest", "duration_sec": 4, "confidence": 0.8}])
        self.assertEqual(rule_severity(c)[0], 3)

    def test_calibrated_perclos_is_used(self):
        # semantica CV: 0.20 brut ar fi oboseală (>= 0.15), dar corectat cu calibrarea e 0.10
        c = self.ctx(perclos=0.20, perclos_cal=0.10)
        c["fusion"]["a"] = 0.9
        self.assertEqual(rule_severity(c)[0], 0)


class TestFaceMissing(unittest.TestCase):
    def test_short_dropout_is_ignored_long_absence_is_gaze_off(self):
        f = AttentionFusion()
        base = {"camera_on": True, "ear": None, "perclos": None, "phone": 0.0, "events": []}
        r = f.update({**base, "face_detected": False, "ts": 0.0})
        self.assertNotIn("gaze", r.contributions, "un cadru pierdut nu e neatenție")
        r = f.update({**base, "face_detected": False, "ts": 1.2})
        self.assertEqual(r.contributions["gaze"]["score"], 0.0)
        r = f.update({**base, "face_detected": True, "gaze_on_road": True, "ts": 1.4})
        self.assertEqual(r.contributions["gaze"]["score"], 1.0)


class TestCalibration(unittest.TestCase):
    def frames(self, ear, perclos, n=200, start=0.0):
        rng = np.random.default_rng(0)
        return [{"ts": start + i / 5, "camera_on": True, "face_detected": True, "gaze_on_road": True, "phone": 0.0,
                 "events": [], "ear": ear + rng.normal(0, 0.01), "perclos": perclos + rng.normal(0, 0.01),
                 "blink_rate": 14 + rng.normal(0, 1)} for i in range(n)]

    def test_learns_robust_driver_baseline(self):
        b = DriverBaseline()
        frames = self.frames(0.22, 0.08)   # ochi mici; PERCLOS rezidual 0.08 (CV calibrează deja închiderea)
        frames[50]["ear"] = 0.01  # o clipire lungă / outlier nu strică mediana
        done = [b.update(f, f["ts"]) for f in frames]
        self.assertTrue(any(done))
        self.assertEqual(b.source, "driver")
        self.assertAlmostEqual(b.ear, 0.22, delta=0.01)
        self.assertAlmostEqual(b.perclos_offset, 0.05, delta=0.02)

    def test_ignores_unclean_frames(self):
        b = DriverBaseline()
        for f in self.frames(0.30, 0.05):
            f["gaze_on_road"] = False
            b.update(f, f["ts"])
        self.assertFalse(b.ready or b.samples["ear"])

    def test_does_not_learn_drowsy_state_as_normal(self):
        b = DriverBaseline()
        for f in self.frames(0.30, 0.45):
            b.update(f, f["ts"])
        from ml1.profile import P
        self.assertLessEqual(b.perclos_offset, P["perclos_max_baseline"] - P["pop_perclos"] + 1e-9)

    def test_falls_back_to_population(self):
        b = DriverBaseline()
        for i in range(500):
            b.update({"ts": i / 5, "camera_on": False}, i / 5)
        self.assertTrue(b.ready)
        self.assertEqual(b.source, "population")
        self.assertEqual(b.perclos_offset, 0.0)

    def test_engine_applies_calibration(self):
        eng = DecisionEngine()
        for f in self.frames(0.22, 0.07, n=170):
            eng.on_vision(f)
        self.assertTrue(eng.baseline.ready)
        self.assertAlmostEqual(eng.fusion.baseline_ear, 0.22, delta=0.01)
        self.assertGreater(eng.last_result.a, 0.9, "ochi mici, dar normali pentru acest șofer")


@unittest.skipUnless(cv_available(), "pachetul Vision lipsește")
class TestCVDemoEndToEnd(unittest.TestCase):
    def test_cv_demo_flow(self):
        msgs = cv_demo_messages()
        eng = DecisionEngine()
        decs = [(m["cv_phase"], d) for m in msgs for d in eng.on_vision(m)]
        by_phase = {}
        for ph, d in decs:
            by_phase.setdefault(ph, []).append(d)
        self.assertIn(1, [d["severity"] for d in by_phase.get("phone", [])])
        # eyes_closed 2.5 s (sub pragul critic de 3 s) nu e tratat ca prăbușire / adormit; escaladarea
        # pe tăcere (nimeni nu răspunde în demo-ul CV) e voită și are alt motiv.
        reasons = [d["reason"] for d in by_phase.get("drowsy", []) if d["severity"] == 3]
        self.assertFalse([r for r in reasons if "cap căzut" in r or "peste 3 s" in r], reasons)
        self.assertIn(2, [d["severity"] for d in by_phase.get("drowsy", [])])
        tools = [a["tool"] for _, d in decs for a in d["actions"]]
        self.assertEqual(tools.count("call_112"), 1, "urgența CV ajunge la 112 o singură dată (fără răspunsuri vocale)")
        self.assertIn("Sunteți bine?", [a["args"].get("question") for _, d in decs for a in d["actions"]])


class TestBridge(unittest.TestCase):
    def test_routes_messages_and_publishes_contract(self):
        sent = []
        eng = DecisionEngine()
        br = Bridge(eng, lambda t, o: sent.append((t, o)))
        for i in range(60):
            msg = {**CV_MSG, "ts": 1000 + i / 5,
                   "events": [{"event": "head_on_wheel", "duration_sec": 2 + i / 5, "confidence": 0.85}]}
            br.on_message(TOPIC_VISION, json.dumps(msg).encode())
        br.on_message(TOPIC_RESPONSE, json.dumps({"answered": False, "latency_sec": None, "text": None, "ts": 1013}))
        br.on_message(TOPIC_VISION, json.dumps({**CV_MSG, "ts": 1013.2, "events": [
            {"event": "head_on_wheel", "duration_sec": 15, "confidence": 0.85}]}))
        br.on_message(TOPIC_VISION, b"{not json")
        topics = [t for t, _ in sent]
        self.assertIn(TOPIC_DECISION, topics)
        self.assertIn(TOPIC_STATUS, topics)
        decisions = [o for t, o in sent if t == TOPIC_DECISION]
        self.assertTrue(all(set(d) >= {"severity", "actions", "reason"} for d in decisions))
        self.assertIn("call_112", [a["tool"] for d in decisions for a in d["actions"]])
        self.assertEqual(br.received["invalid"], 1)
        json.dumps(sent)  # tot ce pleacă e serializabil


class TestEvaluationQuick(unittest.TestCase):
    def test_quick_evaluation_runs_and_is_sane(self):
        from ml1.evaluate import main
        res = main(["--quick", "--no-write", "--seed", "1"])
        for split in ("test", "shift"):
            r = res[split]["rules+calibrare"]
            self.assertGreaterEqual(r["scores"]["recall"][3], 0.9, "nu ratăm urgențe")
            self.assertEqual(r["calls_112"]["fp"], 0, "niciun 112 fals")
            self.assertEqual(r["calls_112"]["fn"], 0, "niciun 112 ratat")
            self.assertGreater(r["scores"]["macro_f1"], res[split]["clasa majoritară (0)"]["scores"]["macro_f1"])

    def test_timestamps_are_exact(self):
        from ml1.evaluate import HZ, TripSim, make_drivers, make_trip
        rng = np.random.default_rng(5)
        s = TripSim(make_drivers(1, rng, "t")[0], make_trip(rng), rng)
        self.assertTrue(all(abs(f["ts"] - i / HZ) < 1e-9 for i, f in enumerate(s.frames)))


class TestRealCVSemantics(unittest.TestCase):
    """Comportamente aliniate la codul real CV (Vision/vigil_vision/face/*.py) — vezi docs/CV_INTEGRATION.md."""

    def ctx(self, fusion=None, **vision):
        return {"fusion": {"a": 0.9, "low_a_sec": 0, **(fusion or {})},
                "vision": {"events": [], **vision}, "driver_response": None}

    def test_perclos_015_is_fatigue_in_cv_semantics(self):
        self.assertEqual(rule_severity(self.ctx(perclos=0.16))[0], 2)
        self.assertEqual(rule_severity(self.ctx(perclos=0.05))[0], 0)

    def test_yawn_and_looking_down_are_not_microsleep(self):
        closed = {"event": "eyes_closed", "duration_sec": 2.0, "confidence": 0.9}
        self.assertEqual(rule_severity(self.ctx(events=[closed]))[0], 2)
        yawn = {"event": "yawn", "duration_sec": 1.5, "confidence": 0.8}
        self.assertEqual(rule_severity(self.ctx(events=[closed, yawn]))[0], 0)
        down = {"event": "gaze_off_road", "duration_sec": 2.0, "confidence": 0.7, "direction": "down"}
        self.assertNotEqual(rule_severity(self.ctx(events=[closed, down]))[0], 2)

    def test_three_seconds_needs_continuous_closure(self):
        """Tracker-ul CV unește clipirile dese într-un episod lung: 3 s de episod != 3 s de ochi închiși."""
        ev = [{"event": "eyes_closed", "duration_sec": 3.5, "confidence": 0.9}]
        merged = self.ctx(fusion={"ear_available": True, "eyes_closed_sec": 0.6}, events=ev)
        continuous = self.ctx(fusion={"ear_available": True, "eyes_closed_sec": 3.2}, events=ev)
        self.assertEqual(rule_severity(merged)[0], 2)
        self.assertEqual(rule_severity(continuous)[0], 3)

    def test_eyes_clearly_open_discards_merged_episode(self):
        ev = [{"event": "eyes_closed", "duration_sec": 2.0, "confidence": 0.9}]
        self.assertEqual(rule_severity(self.ctx(fusion={"eyes_open_now": True}, events=ev))[0], 0)

    def test_recovery_signals(self):
        self.assertEqual(rule_severity(self.ctx(perclos=0.25, recovered=True))[0], 0)
        self.assertEqual(rule_severity(self.ctx(perclos=0.25, perclos_recent=0.02))[0], 0)
        self.assertEqual(rule_severity(self.ctx(perclos=0.25))[0], 2)

    def test_optional_cv_fields(self):
        ev = [{"event": "eyes_closed", "duration_sec": 3.5, "confidence": 0.9}]
        self.assertEqual(rule_severity(self.ctx(events=ev, closed_sec=0.4))[0], 0)
        self.assertEqual(rule_severity(self.ctx(events=ev, closed_sec=3.2))[0], 3)
        self.assertEqual(rule_severity(self.ctx(off_road_ratio=0.6))[0], 1)

    def test_collapse_without_face_or_perclos(self):
        """CV: head_drop vine și din corpul prăbușit, fără față; PERCLOS pe 60 s nu ajunge la 0.6 repede."""
        self.assertEqual(rule_severity(self.ctx(head_drop=True, perclos=0.1, events=[
            {"event": "eyes_closed", "duration_sec": 1.0, "confidence": 0.8}]))[0], 3)
        self.assertEqual(rule_severity(self.ctx(head_drop=True, face_detected=False, events=[
            {"event": "head_on_wheel", "duration_sec": 2.5, "confidence": 0.85}]))[0], 3)

    def test_provisional_baseline_during_calibration(self):
        """În primele 30 s un șofer cu ochi mici nu mai e comparat cu populația."""
        eng = DecisionEngine()
        rng = np.random.default_rng(1)
        for i in range(50):
            eng.on_vision({"ts": i / 5, "camera_on": True, "face_detected": True, "gaze_on_road": True,
                           "phone": 0.0, "events": [], "ear": 0.22 + rng.normal(0, 0.005), "perclos": None})
        self.assertFalse(eng.baseline.ready)
        self.assertAlmostEqual(eng.fusion.baseline_ear, 0.22, delta=0.01)
        self.assertGreater(eng.last_result.a, 0.9)

    def test_legacy_profile_still_available_for_comparison(self):
        from ml1 import profile
        try:
            profile.use("legacy")
            self.assertEqual(rule_severity(self.ctx(perclos=0.16))[0], 0)
        finally:
            profile.use("cv")


class TestNoAnswerExpiry(unittest.TestCase):
    def test_old_no_answer_stops_counting(self):
        """Testul cap-coadă: un „fără răspuns” de acum minute ținea episodul de urgență deschis și în fazele
        normale de după. După NO_ANSWER_TTL nu mai contează."""
        from ml1.engine import NO_ANSWER_TTL
        eng = DecisionEngine()
        eng._asked_ts, eng._asked_q = 0.0, "Sunteți bine?"
        ctx = eng._context({"a": 0.9}, {"events": []}, eng.ask_timeout + 1)
        self.assertFalse(ctx["driver_response"]["answered"])
        ctx = eng._context({"a": 0.9}, {"events": []}, eng.ask_timeout + NO_ANSWER_TTL + 1)
        self.assertIsNone(ctx["driver_response"])
        self.assertIsNone(eng._asked_ts)

    def test_explicit_silence_from_ml2_also_expires(self):
        from ml1.engine import NO_ANSWER_TTL
        eng = DecisionEngine()
        eng._asked_ts, eng._asked_q = 0.0, "Sunteți bine?"
        eng.on_driver_response({"answered": False, "latency_sec": None, "text": None}, ts=5.0)
        self.assertFalse(eng._context({"a": 0.9}, {"events": []}, 10.0)["driver_response"]["answered"])
        self.assertIsNone(eng._context({"a": 0.9}, {"events": []}, 5.0 + NO_ANSWER_TTL + 1)["driver_response"])


if __name__ == "__main__":
    unittest.main()
