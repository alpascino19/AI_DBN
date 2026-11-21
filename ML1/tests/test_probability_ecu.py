"""Matricea de probabilitate (filtrul HMM, statisticile) și controlerul fuzzy pentru ECU."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ml1 import ecu_fuzzy as fz
from ml1 import probability as prob
from ml1.engine import DecisionEngine
from ml1.mqtt_bridge import TOPIC_ECU, TOPIC_RESPONSE, TOPIC_VISION, Bridge

DATA = Path(__file__).resolve().parent / "data"


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


class TestProbModel(unittest.TestCase):
    def setUp(self):
        self.m = prob.ProbModel.default()

    def test_rows_are_distributions(self):
        for row in self.m.emission:
            self.assertTrue(_close(sum(row), 1.0))
        for dt in (0.0, 0.2, 1.0, 7.3, 500.0):
            for row in self.m.transition(dt):
                self.assertTrue(_close(sum(row), 1.0))
                self.assertTrue(all(x >= 0 for x in row))

    def test_transition_is_identity_at_zero_and_composes(self):
        T0 = self.m.transition(0.0)
        self.assertTrue(all(_close(T0[i][j], 1.0 if i == j else 0.0) for i in range(4) for j in range(4)))
        T1, T2 = self.m.transition(1.0), self.m.transition(2.0)
        T11 = prob._matmul(T1, T1)
        self.assertTrue(all(_close(T2[i][j], T11[i][j], 1e-3) for i in range(4) for j in range(4)))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.m.save(Path(d) / "m.json")
            m2 = prob.ProbModel.load(p)
        self.assertEqual(len(m2.emission), 4)
        self.assertTrue(_close(m2.emission[2][7], self.m.emission[2][7], 1e-5))

    def test_broken_file_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("{nu e json", encoding="utf-8")
            self.assertEqual(prob.ProbModel.load(p).meta["source"], prob.ProbModel.default().meta["source"])

    def test_shipped_matrix_loads(self):
        m = prob.ProbModel.load()
        self.assertEqual(len(m.emission), prob.K)
        self.assertEqual(len(m.emission[0]), prob.N_OBS)


class TestStateFilter(unittest.TestCase):
    def test_sustained_emergency_and_recovery(self):
        f = prob.StateFilter(prob.ProbModel.default())
        for i in range(25):                       # 5 s de severitate 3 din reguli, a mic
            f.update(3, 0.1, 0.6, 100 + i / 5)
        self.assertGreater(f.b[3], 0.9)
        self.assertEqual(f.map_state, 3)
        for i in range(300):                      # 60 s de condus normal
            f.update(0, 0.95, 0.6, 105 + i / 5)
        self.assertGreater(f.b[0], 0.9)

    def test_single_noisy_frame_does_not_flip_state(self):
        f = prob.StateFilter(prob.ProbModel.default())
        for i in range(50):
            f.update(0, 0.95, 0.6, i / 5)
        f.update(3, 0.9, 0.6, 10.2)               # un cadru izolat de severitate 3
        self.assertLess(f.b[3], 0.5)

    def test_camera_off_is_not_evidence(self):
        f = prob.StateFilter(prob.ProbModel.default())
        for i in range(25):
            f.update(3, 0.1, 0.6, i / 5)
        before = f.b[3]
        f.update(0, 0.95, 0.0, 5.2)               # încredere 0: doar predicție, fără observație
        self.assertFalse(f.observed)
        self.assertGreater(f.b[3], before - 0.01)

    def test_bayes_decision_is_not_argmax_under_asymmetric_cost(self):
        b = [0.8, 0.0, 0.0, 0.2]                  # cea mai probabilă = 0, dar 20% urgență
        self.assertEqual(max(range(4), key=lambda k: b[k]), 0)
        self.assertEqual(prob.bayes_decision(b), 3)
        self.assertEqual(prob.bayes_decision([0.99, 0.01, 0.0, 0.0]), 0)


class TestStatsAndFit(unittest.TestCase):
    def test_perfect_confusion(self):
        s = prob.matrix_stats([[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 7, 0], [0, 0, 0, 3]])
        self.assertTrue(_close(s["kappa"], 1.0) and _close(s["mcc"], 1.0) and _close(s["cost_per_frame"], 0.0))
        self.assertTrue(all(_close(s["p_true_given_pred"][k][k], 1.0) for k in range(4)))

    def test_known_values(self):
        # starea 3: 8 din 10 prinse, 2 alarme false din 90 cadre normale
        s = prob.matrix_stats([[88, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 8]])
        c3 = s["per_class"][3]
        self.assertTrue(_close(c3["sensitivity"], 0.8) and _close(c3["ppv"], 0.8))
        self.assertTrue(_close(c3["specificity"], 88 / 90))
        self.assertTrue(_close(c3["lr_pos"], 0.8 / (2 / 90)))
        self.assertEqual(json.loads(json.dumps(s))["n"], 100)   # serializabil (fără Infinity)

    def test_fit_recovers_emission(self):
        truth = [0] * 400 + [2] * 400
        rule = [0] * 400 + [2] * 300 + [0] * 100        # regulile ratează 25% din oboseală
        m = prob.fit_model([{"truth": truth, "rule": rule, "a": [0.95] * 800, "conf": [0.6] * 800}], hz=5,
                           alpha_emission=0.0)
        self.assertTrue(_close(m.emission[2][prob.obs_symbol(2, 0.95)], 0.75, 1e-9))
        self.assertTrue(_close(m.emission[0][prob.obs_symbol(0, 0.95)], 1.0, 1e-9))
        self.assertLess(m.q[0][0], 0)                    # rata de ieșire din „atent” e învățată
        self.assertGreater(m.q[0][2], 0)


class TestFuzzy(unittest.TestCase):
    def test_every_input_fires_a_rule_and_outputs_in_range(self):
        for i in range(11):
            for j in range(11):
                for k in range(6):
                    a, r, c = i / 10, j / 10, k / 5
                    mu = fz.fuzzify(a, r, c)
                    self.assertTrue(any(min(mu[v][t] for v, t in ante.items()) > 0 for ante, _ in fz.RULES), (a, r, c))
                    out = fz.infer(a, r, c)
                    for key, (lo, hi) in fz.CLIP.items():
                        self.assertTrue(lo <= out[key] <= hi)

    def test_monotone_speed(self):
        for c in (0.2, 0.6):
            speeds = [fz.infer(0.9, r / 10, c)["speed"] for r in range(11)]
            self.assertTrue(all(x >= y - 1e-9 for x, y in zip(speeds, speeds[1:])), "riscul mai mare nu accelerează")
            speeds = [fz.infer(a / 10, 0.05, c)["speed"] for a in range(11)]
            self.assertTrue(all(x <= y + 1e-9 for x, y in zip(speeds, speeds[1:])), "atenția mai mare nu încetinește")

    def test_extremes(self):
        ok = fz.infer(1.0, 0.0, 1.0)
        self.assertTrue(_close(ok["speed"], 1.0, 1e-3) and _close(ok["authority"], 0.0, 1e-3))
        bad = fz.infer(0.0, 1.0, 1.0)
        self.assertLess(bad["speed"], 0.4)
        self.assertGreater(bad["authority"], 0.9)

    def test_lut_matches_exact_inference(self):
        err = fz.lut_error(fz.quantize(fz.build_lut()), n=1500)
        for key, (mean, mx) in err.items():
            self.assertLess(mean, 0.01, key)
            self.assertLess(mx, 0.2, key)

    @unittest.skipUnless(shutil.which("gcc"), "gcc lipsește")
    def test_exported_c_header_compiles_and_matches_python(self):
        with tempfile.TemporaryDirectory() as d:
            hdr, _ = fz.export_c(Path(d) / "vigil_fuzzy_lut.h")
            src = Path(d) / "t.c"
            src.write_text((fz.ECU_DIR / "test_lut.c").read_text(encoding="utf-8"), encoding="utf-8")
            exe = Path(d) / "t.exe"
            subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", "-I", d, "-o", str(exe), str(src)],
                           check=True, capture_output=True)
            pts = [(0.9, 0.05, 0.6), (0.4, 0.48, 0.6), (0.15, 0.96, 0.6), (0.9, 0.02, 0.2), (0.0, 0.0, 0.0), (1, 1, 1)]
            out = subprocess.run([str(exe)], input="\n".join(" ".join(map(str, p)) for p in pts), text=True,
                                 capture_output=True, check=True).stdout.split("\n")
        q = fz.quantize(fz.build_lut())
        for p, line in zip(pts, out):
            c_vals = [float(x) for x in line.split()]
            py = fz.lut_eval(q, *p)
            for cv, key in zip(c_vals, ("speed", "authority", "headway")):
                self.assertAlmostEqual(cv, py[key], places=3)


OK = [0.97, 0.01, 0.01, 0.01]
EMERG = [0.01, 0.01, 0.03, 0.95]


class TestEcuController(unittest.TestCase):
    def test_attentive_driver_keeps_speed(self):
        e = fz.EcuController(100)
        for i in range(50):
            out = e.step(i / 5, 0.95, OK, 0.6, 0)
        self.assertEqual(out["mode"], "NORMAL")
        self.assertEqual(out["speed_ref_kmh"], 100.0)
        self.assertFalse(out["hazards"])

    def test_severity3_caps_speed_and_hazards_immediately(self):
        out = fz.EcuController(100).step(0.0, 0.95, OK, 0.6, 3)   # chiar dacă fuzzy-ul crede că e bine
        self.assertLessEqual(out["speed_factor"], 0.5)
        self.assertTrue(out["hazards"])
        self.assertEqual(out["lka_authority"], 1.0)

    def test_112_triggers_mrm_with_limited_decel_and_recovery_releases_it(self):
        e = fz.EcuController(100)
        e.step(0.0, 0.1, EMERG, 0.6, 3, called_112=True)
        prev = e.speed_ref
        for i in range(1, 60):
            out = e.step(i / 5, 0.1, EMERG, 0.6, 3, called_112=True)
            self.assertEqual(out["mode"], "MRM")
            self.assertEqual(out["lane"], "shoulder")
            self.assertLessEqual(prev - e.speed_ref, fz.DECEL_MRM * 3.6 / 5 + 1e-6)   # fără frânare bruscă
            prev = e.speed_ref
        self.assertLess(e.speed_ref, 100 - 30)
        for i in range(60, 400):
            out = e.step(i / 5, 0.95, OK, 0.6, 0)
            self.assertLessEqual(e.speed_ref - prev, fz.ACCEL * 3.6 / 5 + 1e-6)   # revenire lină
            prev = e.speed_ref
        self.assertNotEqual(out["mode"], "MRM")
        self.assertEqual(out["speed_ref_kmh"], 100.0)

    def test_mrm_needs_probability_confirmation_without_112(self):
        e = fz.EcuController(100)
        for i in range(150):                     # 30 s severitate 3 publicată, dar P(urgență) mic (histerezis)
            out = e.step(i / 5, 0.4, [0.02, 0.0, 0.96, 0.02], 0.6, 3)
        self.assertNotEqual(out["mode"], "MRM")
        for i in range(150, 220):                # acum confirmată de matrice: MRM după MRM_HOLD_SEC
            out = e.step(i / 5, 0.1, EMERG, 0.6, 3)
        self.assertEqual(out["mode"], "MRM")

    def test_short_glance_does_not_brake(self):
        e = fz.EcuController(100)
        for i in range(25):
            e.step(i / 5, 0.95, OK, 0.6, 0)
        for i in range(25, 30):                  # 1 s cu a = 0.6
            out = e.step(i / 5, 0.6, OK, 0.6, 0)
        self.assertGreater(out["speed_ref_kmh"], 97.0)


class TestEngineIntegration(unittest.TestCase):
    def test_collapse_scenario_ends_in_mrm_and_bridge_publishes_ecu(self):
        sent = []
        br = Bridge(DecisionEngine(), lambda t, o: sent.append((t, o)))
        for line in (DATA / "collapse.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if "driver_response" in m:
                br.on_message(TOPIC_RESPONSE, json.dumps({**m["driver_response"], "ts": m.get("ts")}))
            else:
                br.on_message(TOPIC_VISION, json.dumps(m))
        ecu = [o for t, o in sent if t == TOPIC_ECU]
        self.assertTrue(ecu, "puntea publică vigil/ecu")
        self.assertEqual(ecu[0]["mode"], "NORMAL")
        self.assertEqual(ecu[-1]["mode"], "MRM")
        self.assertTrue(ecu[-1]["hazards"])
        self.assertGreater(ecu[-1]["state"]["p_state"][3], 0.5)
        ts = [o["ts"] for o in ecu]
        self.assertTrue(all(b - a >= 0.19 for a, b in zip(ts, ts[1:])), "maxim 5 Hz")
        json.dumps(sent)


if __name__ == "__main__":
    unittest.main()
