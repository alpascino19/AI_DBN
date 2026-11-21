"""Teste pentru ml1.fusion.  Rulare:  python -m unittest -v"""

import json
import math
import unittest
from pathlib import Path

from ml1.fake_vision import generate
from ml1.fusion import AttentionFusion

DATA = Path(__file__).parent / "data"


def replay(messages, fusion=None):
    fusion = fusion or AttentionFusion()
    out = []
    for m in messages:
        if "driver_response" in m:
            fusion.update_driver_response(m["driver_response"], ts=m["ts"])
        else:
            out.append(fusion.update(m))
    return out


def at(results, t):
    """Rezultatul cel mai apropiat de momentul t."""
    return min(results, key=lambda r: abs(r.ts - t))


class TestScenarios(unittest.TestCase):
    def test_calibration_is_attentive(self):
        r = replay(generate("calibration"))
        self.assertTrue(all(x.a > 0.9 for x in r))
        self.assertFalse(any(x.critical for x in r))

    def test_one_second_glance_is_smoothed(self):
        r = replay(generate("glance"))
        self.assertGreater(min(x.a for x in r), 0.85, "o privire de 1 s nu trebuie să prăbușească scorul")

    def test_phone_is_low_severity(self):
        r = replay(generate("phone"))
        a = at(r, 12).a
        self.assertTrue(0.5 < a < 0.8, a)
        self.assertGreater(r[-1].a, a, "scorul revine după ce telefonul dispare")

    def test_drowsy_decreases_monotonically_ish(self):
        r = replay(generate("drowsy"))
        self.assertGreater(at(r, 5).a, at(r, 15).a)
        self.assertGreater(at(r, 15).a, at(r, 29).a)
        self.assertLess(r[-1].a, 0.5)
        self.assertEqual(r[-1].capped_by, "eyes")

    def test_hand_on_chest_drops_score(self):
        r = replay(generate("chest"))
        self.assertLess(at(r, 10).a, 0.65)
        self.assertEqual(at(r, 10).capped_by, "events")

    def test_collapse_is_critical_and_fast(self):
        r = replay(generate("collapse"))
        first = next(x for x in r if x.critical)
        self.assertLess(first.ts, 5.5, "cap căzut + ochi închiși -> critic imediat")
        self.assertLess(first.a, 0.25, "critic ocolește netezirea")
        self.assertLess(r[-1].a, 0.2)

    def test_privacy_mode_lowers_confidence(self):
        r = replay(generate("privacy"))
        on = at(r, 3)
        off = at(r, 10)
        self.assertTrue(on.camera_on)
        self.assertFalse(off.camera_on)
        self.assertLess(off.confidence, on.confidence)
        self.assertLessEqual(off.confidence, 0.4)

    def test_noisy_data_never_crashes_or_escapes_range(self):
        r = replay(generate("noisy"))
        for x in r:
            self.assertTrue(0.0 <= x.a <= 1.0)
            self.assertTrue(0.0 <= x.confidence <= 1.0)
            self.assertTrue(math.isfinite(x.a))
            json.dumps(x.to_dict())  # serializabil pentru MQTT


class TestUnits(unittest.TestCase):
    def test_drop_faster_than_recovery(self):
        f = AttentionFusion()
        bad = {"perclos": 0.6, "ear": 0.1, "gaze_on_road": False}
        good = {"perclos": 0.05, "ear": 0.3, "gaze_on_road": True}
        f.update({**good, "ts": 0})
        a_after_drop = f.update({**bad, "ts": 1}).a
        drop = 1.0 - a_after_drop
        rise = f.update({**good, "ts": 2}).a - a_after_drop
        self.assertGreater(drop, rise)

    def test_no_response_from_driver(self):
        f = AttentionFusion()
        f.update_driver_response({"answered": False, "latency_sec": None, "text": None}, ts=0)
        for i in range(1, 16):  # 3 s fără cameră, 5 Hz
            r = f.update({"ts": i / 5, "camera_on": False})
        self.assertEqual(r.contributions["voice"]["score"], 0.0)
        self.assertLess(r.a, 0.1)
        self.assertLessEqual(r.confidence, 0.2, "doar vocea -> încredere mică")

    def test_voice_response_expires(self):
        f = AttentionFusion()
        f.update_driver_response({"answered": True, "latency_sec": 1.0}, ts=0)
        self.assertIn("voice", f.update({"ts": 10, "camera_on": False}).contributions)
        self.assertNotIn("voice", f.update({"ts": 60, "camera_on": False}).contributions)

    def test_no_signals_keeps_a_with_zero_confidence(self):
        f = AttentionFusion(initial_a=0.7)
        r = f.update(None)
        self.assertIsNone(r.a_raw)
        self.assertEqual(r.a, 0.7)
        self.assertEqual(r.confidence, 0.0)

    def test_contributions_sum_to_a_raw_when_uncapped(self):
        r = AttentionFusion().update({"ts": 0, "perclos": 0.05, "ear": 0.3, "gaze_on_road": False, "phone": 0.5})
        self.assertIsNone(r.capped_by)
        self.assertAlmostEqual(sum(c["contrib"] for c in r.contributions.values()), r.a_raw, places=2)

    def test_contract_example_message(self):
        """Exemplul exact din TASKS.md (fără ts / camera_on)."""
        msg = {"ear": 0.21, "perclos": 0.3, "blink_rate": 28, "gaze_on_road": False, "head_drop": False,
               "phone": 0.1, "events": [{"event": "hand_on_chest", "duration_sec": 4, "confidence": 0.8,
                                        "breathing": "heavy"}], "landmarks": []}
        r = AttentionFusion().update(msg)
        self.assertTrue(r.camera_on)
        self.assertLess(r.a_raw, 0.6)

    def test_baseline_ear_from_calibration(self):
        """Un șofer cu ochi mai mici (EAR normal 0.22) nu trebuie penalizat după calibrare."""
        msg = {"ts": 0, "ear": 0.22, "perclos": 0.05}
        generic = AttentionFusion().update(msg).contributions["eyes"]["score"]
        calibrated = AttentionFusion(baseline_ear=0.22).update(msg).contributions["eyes"]["score"]
        self.assertGreater(calibrated, generic)


class TestDataFiles(unittest.TestCase):
    def test_jsonl_files_match_generator(self):
        """tests/data/*.jsonl sunt regenerate cu `python -m ml1.fake_vision`."""
        for name in ("calibration", "glance", "phone", "drowsy", "chest", "collapse", "privacy", "noisy"):
            path = DATA / f"{name}.jsonl"
            self.assertTrue(path.exists(), f"lipsește {path}; rulează python -m ml1.fake_vision")
            lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(lines, generate(name), name)


if __name__ == "__main__":
    unittest.main()
