"""Teste pentru ml1.validate + ml1.rules.  Rulare:  python -m unittest -v"""

import json
import time
import unittest
from pathlib import Path

from ml1.rules import fallback_decision, rule_severity
from ml1.tools import TOOLS, ollama_tools
from ml1.validate import decide

DATA = Path(__file__).parent / "data"
CONTEXTS = json.loads((DATA / "llm_contexts.json").read_text(encoding="utf-8"))
CASES = json.loads((DATA / "llm_outputs.json").read_text(encoding="utf-8"))


def tools_of(dec):
    return [a["tool"] for a in dec["actions"]]


def assert_contract(tc, dec):
    """Orice decizie (LLM sau reguli) respectă contractul `vigil/decision`."""
    tc.assertIn(dec["severity"], (0, 1, 2, 3))
    tc.assertIsInstance(dec["reason"], str)
    tc.assertTrue(dec["reason"])
    for a in dec["actions"]:
        tc.assertIn(a["tool"], TOOLS)
        tc.assertEqual(set(a["args"]) - set(TOOLS[a["tool"]]["args"]), set())
    if "call_112" in tools_of(dec):
        tc.assertEqual(dec["severity"], 3)
    json.dumps(dec, ensure_ascii=False)


class TestLLMOutputs(unittest.TestCase):
    def test_all_cases(self):
        for case in CASES:
            with self.subTest(case["name"]):
                ctx = CONTEXTS[case["ctx"]]
                dec = decide(ctx, lambda _c, raw=case["raw"]: raw)
                exp = case["expect"]
                assert_contract(self, dec)
                self.assertEqual(dec["source"], exp["source"], dec.get("fallback_reason"))
                if "severity" in exp:
                    self.assertEqual(dec["severity"], exp["severity"], dec)
                for t in exp.get("has", []):
                    self.assertIn(t, tools_of(dec))
                for t in exp.get("lacks", []):
                    self.assertNotIn(t, tools_of(dec))
                for t, lvl in exp.get("levels", {}).items():
                    self.assertEqual(next(a for a in dec["actions"] if a["tool"] == t)["args"]["level"], lvl)
                for t, n in exp.get("count", {}).items():
                    self.assertEqual(tools_of(dec).count(t), n)
                if "max_speak_len" in exp:
                    speak = next(a for a in dec["actions"] if a["tool"] == "speak")
                    self.assertLessEqual(len(speak["args"]["text"]), exp["max_speak_len"])
                if exp.get("warnings"):
                    self.assertTrue(dec["warnings"])
                if exp.get("safety"):
                    self.assertTrue(dec["safety_notes"])


class TestFallback(unittest.TestCase):
    def test_timeout_falls_back_fast(self):
        def slow(_ctx):
            time.sleep(1.0)
            return "{}"
        t0 = time.perf_counter()
        dec = decide(CONTEXTS["drowsy"], slow, timeout=0.2)
        self.assertLess(time.perf_counter() - t0, 0.6)
        self.assertEqual(dec["source"], "rules")
        self.assertIn("timeout", dec["fallback_reason"])
        self.assertEqual(dec["severity"], 2)

    def test_llm_exception_falls_back(self):
        def boom(_ctx):
            raise ConnectionError("Ollama nu rulează")
        dec = decide(CONTEXTS["chest"], boom)
        self.assertEqual(dec["source"], "rules")
        self.assertEqual(dec["severity"], 3)
        self.assertIn("ask_driver", tools_of(dec))

    def test_no_llm(self):
        dec = decide(CONTEXTS["calm"], None)
        self.assertEqual((dec["source"], dec["severity"], dec["actions"]), ("rules", 0, []))

    def test_rule_severities_for_all_contexts(self):
        expected = {"calm": 0, "phone": 1, "drowsy": 2, "chest": 3, "collapse_no_answer": 3}
        for name, sev in expected.items():
            with self.subTest(name):
                self.assertEqual(rule_severity(CONTEXTS[name])[0], sev)

    def test_fallback_calls_112_only_without_answer(self):
        self.assertNotIn("call_112", tools_of(fallback_decision(CONTEXTS["chest"])))
        self.assertIn("call_112", tools_of(fallback_decision(CONTEXTS["collapse_no_answer"])))

    def test_fallback_decisions_respect_contract(self):
        for name, ctx in CONTEXTS.items():
            with self.subTest(name):
                assert_contract(self, fallback_decision(ctx))

    def test_garbage_context_does_not_crash(self):
        for ctx in ({}, {"fusion": None, "vision": None}, {"vision": {"events": "x", "perclos": "abc"}}):
            assert_contract(self, decide(ctx, lambda _c: "nu e json"))


class TestToolsSchema(unittest.TestCase):
    def test_ollama_schema(self):
        names = {t["function"]["name"] for t in ollama_tools()}
        self.assertEqual(names, set(TOOLS))
        alarm = next(t for t in ollama_tools() if t["function"]["name"] == "play_alarm")
        self.assertEqual(alarm["function"]["parameters"]["required"], ["level"])


if __name__ == "__main__":
    unittest.main()
