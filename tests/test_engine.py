"""Teste pentru prompt, meniul de acțiuni, rapoarte și engine.  Rulare:  python -m unittest -v"""

import json
import re
import threading
import time
import unittest
from pathlib import Path

from ml1.actions import ACTION_MENU, LLM_SCHEMA, PHRASES, expand_actions
from ml1.engine import DecisionEngine, replay
from ml1.llm_client import OllamaClient
from ml1.prompt import FEW_SHOT, SYSTEM, approx_tokens, build_messages, context_line
from ml1.reports import build_112_report, build_trip_report
from ml1.rules import answer_status, rule_severity, wants_112
from ml1.validate import decide

DATA = Path(__file__).parent / "data"
CONTEXTS = json.loads((DATA / "llm_contexts.json").read_text(encoding="utf-8"))

DEFAULT_ACTIONS = {0: [], 1: ["remind_road"], 2: ["alarm", "esp32", "suggest_break", "ask_tired"],
                   3: ["alarm", "esp32", "ask_ok"]}


def hint_llm(ctx):
    """LLM fals: urmează `hint` din linia de context, în formatul compact (ca Ollama)."""
    line = context_line(ctx)
    sev = int(re.search(r"hint=(\d)", line).group(1))
    acts = list(DEFAULT_ACTIONS[sev])
    if "no answer" in line and sev == 3:
        acts.append("call_112")
    return {"message": {"role": "assistant", "content": json.dumps({"severity": sev, "actions": acts, "reason": "test"})}}


class TestPrompt(unittest.TestCase):
    def test_prompt_fits_small_context(self):
        for ctx in CONTEXTS.values():
            self.assertLess(approx_tokens(build_messages(ctx)), 900, "num_ctx=1024")

    def test_prefix_is_static(self):
        """Prefix identic între apeluri = KV-cache refolosit."""
        a = build_messages(CONTEXTS["calm"])[:-1]
        b = build_messages(CONTEXTS["collapse_no_answer"])[:-1]
        self.assertEqual(a, b)

    def test_context_line_is_compact_and_has_no_landmarks(self):
        ctx = {**CONTEXTS["chest"], "vision": {**CONTEXTS["chest"]["vision"], "landmarks": [[0.1, 0.2]] * 468}}
        line = context_line(ctx)
        self.assertLess(len(line), 400)
        self.assertNotIn("landmarks", line)
        self.assertIn("hand_on_chest 5s heavy", line)
        self.assertIn("hint=3", line)

    def test_privacy_mode_line(self):
        line = context_line({"fusion": {"a": 0.9, "confidence": 0.2, "camera_on": False}, "vision": {"camera_on": False}})
        self.assertIn("cam=off", line)
        self.assertNotIn("perclos", line)

    def test_few_shot_examples_are_valid_and_match_hints(self):
        for user, out in FEW_SHOT:
            self.assertTrue(set(out["actions"]) <= set(ACTION_MENU))
            self.assertEqual(out["severity"], int(re.search(r"hint=(\d)", user).group(1)))
            self.assertLessEqual(len(out["reason"].split()), 12)

    def test_system_lists_all_actions(self):
        for a in ACTION_MENU:
            self.assertIn(a, SYSTEM)

    def test_ollama_payload(self):
        body = OllamaClient(model="qwen2.5:1.5b").payload(CONTEXTS["drowsy"])
        self.assertEqual(body["format"], LLM_SCHEMA)
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertLessEqual(body["options"]["num_predict"], 64)
        self.assertNotIn("think", body)
        self.assertFalse(OllamaClient(model="qwen3:1.7b").payload(CONTEXTS["drowsy"])["think"])


class TestActions(unittest.TestCase):
    def test_compact_expansion_levels_follow_severity(self):
        d = expand_actions({"severity": 2, "actions": ["alarm", "esp32", "ask_tired"]}, CONTEXTS["drowsy"])
        self.assertEqual(d["actions"][0], {"tool": "play_alarm", "args": {"level": 2}})
        self.assertEqual(d["actions"][1], {"tool": "set_esp32", "args": {"level": 2}})
        self.assertEqual(d["actions"][2]["args"]["question"], PHRASES["ask_tired"])

    def test_compact_decision_through_decide(self):
        raw = '{"severity": 1, "actions": ["remind_phone", "dance"], "reason": "telefon"}'
        d = decide(CONTEXTS["phone"], lambda _c: raw)
        self.assertEqual(d["source"], "llm")
        self.assertEqual(d["actions"], [{"tool": "speak", "args": {"text": PHRASES["remind_phone"]}}])
        self.assertTrue(d["warnings"])

    def test_compact_112_gets_template_report(self):
        raw = '{"severity": 3, "actions": ["alarm", "esp32", "call_112"], "reason": "nu răspunde"}'
        d = decide(CONTEXTS["collapse_no_answer"], lambda _c: raw)
        rep = next(a for a in d["actions"] if a["tool"] == "call_112")["args"]["report"]
        self.assertIn("SIMULAT", rep)
        self.assertIn("cap căzut", rep)


class TestRealModelRegressions(unittest.TestCase):
    """Greșeli observate la qwen2.5:1.5b real; garda din validate.apply_safety le prinde."""

    def run_llm(self, ctx, out):
        return decide(ctx, lambda _c: json.dumps(out))

    def test_cannot_raise_above_zero_hint(self):
        ctx = {"fusion": {"a": 0.8, "critical": False}, "vision": {"gaze_on_road": True, "phone": 0.01, "events": []},
               "driver_response": None}
        d = self.run_llm(ctx, {"severity": 1, "actions": ["remind_phone"], "reason": "telefon în mână de 1 s"})
        self.assertEqual(d["severity"], 0)
        self.assertEqual(d["actions"], [])
        self.assertTrue(d["safety_notes"])

    def test_cannot_jump_two_levels_up(self):
        d = self.run_llm(CONTEXTS["phone"], {"severity": 3, "actions": ["alarm", "esp32", "ask_ok"], "reason": "x"})
        self.assertEqual(d["severity"], 2)

    def test_phone_reminder_needs_phone_signal(self):
        ctx = {**CONTEXTS["drowsy"], "fusion": {"a": 0.7, "critical": False}, "vision": {"perclos": 0.1, "gaze_on_road": True,
               "phone": 0.0, "events": [{"event": "gaze_off_road", "duration_sec": 3, "confidence": 0.8}]}}
        d = self.run_llm(ctx, {"severity": 1, "actions": ["remind_phone"], "reason": "x"})
        self.assertNotIn(PHRASES["remind_phone"], [a["args"].get("text") for a in d["actions"]])

    def test_llm_without_question_gets_one(self):
        d = self.run_llm(CONTEXTS["drowsy"], {"severity": 2, "actions": ["alarm", "esp32", "suggest_break"], "reason": "x"})
        asks = [a for a in d["actions"] if a["tool"] == "ask_driver"]
        self.assertEqual(asks[0]["args"]["question"], PHRASES["ask_tired"])

    def test_drowsy_with_llm_escalates_on_silence(self):
        """Fără întrebare adăugată, tăcerea n-ar escalada niciodată la severitatea 3."""
        llm = lambda c: json.dumps({"severity": 2, "actions": ["alarm", "esp32", "suggest_break"], "reason": "x"})
        eng = DecisionEngine(llm_fn=llm)
        decs = replay(DATA / "drowsy.jsonl", eng)
        self.assertIn(3, [d["severity"] for d in decs])
        self.assertNotIn("call_112", [a["tool"] for d in decs for a in d["actions"]])

    def test_no_suggest_break_at_severity_3(self):
        d = self.run_llm(CONTEXTS["chest"], {"severity": 3, "actions": ["alarm", "esp32", "suggest_break"], "reason": "x"})
        self.assertNotIn("suggest_break", [a["tool"] for a in d["actions"]])


class TestRulesAnswers(unittest.TestCase):
    def test_answer_status(self):
        self.assertEqual(answer_status(None), "none")
        self.assertEqual(answer_status({"answered": False}), "no_answer")
        self.assertEqual(answer_status({"answered": True, "latency_sec": 1.0, "text": "da, sunt bine"}), "ok")
        self.assertEqual(answer_status({"answered": True, "latency_sec": 5.5, "text": "da"}), "slow")
        self.assertEqual(answer_status({"answered": True, "latency_sec": 1.0, "text": "nu mă simt bine"}), "distress")

    def test_coherent_answer_lowers_chest_to_2(self):
        ctx = {**CONTEXTS["chest"], "driver_response": {"answered": True, "latency_sec": 1.0, "text": "sunt bine"}}
        self.assertEqual(rule_severity(ctx)[0], 2)

    def test_distress_answer_triggers_112(self):
        ctx = {**CONTEXTS["chest"], "driver_response": {"answered": True, "latency_sec": 2.0, "text": "mă doare pieptul",
                                                       "question": PHRASES["ask_ok"]}}
        self.assertTrue(wants_112(ctx))

    def test_no_answer_to_break_question_is_not_112(self):
        ctx = {**CONTEXTS["collapse_no_answer"],
               "driver_response": {"answered": False, "question": PHRASES["ask_tired"]}}
        self.assertFalse(wants_112(ctx))


class TestReports(unittest.TestCase):
    def test_112_report_content(self):
        rep = build_112_report(CONTEXTS["collapse_no_answer"], now=0)
        for s in ("SIMULAT", "Locație", "cap căzut", "ochi închiși", "niciun răspuns", "nu diagnostic", "0.17"):
            self.assertIn(s, rep)
        self.assertLessEqual(len(rep), 1000)

    def test_112_report_chest_suspicion(self):
        self.assertIn("urgență cardiacă", build_112_report(CONTEXTS["chest"], now=0))

    def test_112_report_on_garbage(self):
        rep = build_112_report({"vision": {"events": "x"}, "fusion": {"a": "?"}}, now=0)
        self.assertIn("șoferul nu reacționează", rep)

    def test_trip_report(self):
        eng = DecisionEngine()
        replay(DATA / "drowsy.jsonl", eng)
        rep = build_trip_report(eng.trip_log())
        self.assertIn("micro-somn", rep)
        self.assertIn("pauză", rep)
        self.assertIn("RAPORT POST-CURSĂ", rep)
        self.assertIn("Durată monitorizată", build_trip_report({}))


class TestEngine(unittest.TestCase):
    def tools(self, decs):
        return [a["tool"] for d in decs for a in d["actions"]]

    def test_calm_scenarios_publish_nothing_and_never_call_llm(self):
        for name in ("calibration", "glance", "privacy"):
            with self.subTest(name):
                eng = DecisionEngine(llm_fn=hint_llm)
                self.assertEqual(replay(DATA / f"{name}.jsonl", eng), [])
                self.assertEqual(eng.stats["llm_calls"], 0)

    def test_llm_called_only_on_change(self):
        eng = DecisionEngine(llm_fn=hint_llm)
        replay(DATA / "drowsy.jsonl", eng)
        self.assertGreater(eng.stats["llm_calls"], 0)
        self.assertLess(eng.stats["llm_calls"], eng.stats["frames"] / 5)

    def test_collapse_flow(self):
        """Mână pe piept -> alarmă + „Sunteți bine?” -> cap căzut, fără răspuns -> 112 o singură dată."""
        for llm in (None, hint_llm):
            with self.subTest(llm=bool(llm)):
                eng = DecisionEngine(llm_fn=llm)
                decs = replay(DATA / "collapse.jsonl", eng)
                tools = self.tools(decs)
                self.assertEqual(tools.count("call_112"), 1)
                self.assertLess(tools.index("ask_driver"), tools.index("call_112"))
                first112 = next(d for d in decs if any(a["tool"] == "call_112" for a in d["actions"]))
                self.assertGreaterEqual(first112["ts"], 10.0, "112 doar după lipsa răspunsului")

    def test_ask_not_repeated_while_waiting(self):
        eng = DecisionEngine(llm_fn=hint_llm)
        decs = replay(DATA / "chest.jsonl", eng)
        asks = [d["ts"] for d in decs if any(a["tool"] == "ask_driver" for a in d["actions"])]
        self.assertEqual(len(asks), 1)

    def test_drowsy_never_calls_112_without_asking_are_you_ok(self):
        eng = DecisionEngine()
        decs = replay(DATA / "drowsy.jsonl", eng)
        self.assertNotIn("call_112", self.tools(decs))
        self.assertIn(3, [d["severity"] for d in decs])

    def test_silence_counts_as_no_answer(self):
        """STT căzut / ML 2 nu trimite nimic: după 8 s tăcerea = fără răspuns."""
        eng = DecisionEngine()
        msgs = [m for m in (json.loads(l) for l in (DATA / "collapse.jsonl").read_text(encoding="utf-8").splitlines())
                if "driver_response" not in m]
        decs = [d for m in msgs for d in eng.on_vision(m)]
        d112 = next(d for d in decs if any(a["tool"] == "call_112" for a in d["actions"]))
        self.assertGreaterEqual(d112["ts"], 3.0 + 8.0)

    def test_fast_path_does_not_wait_for_slow_llm(self):
        gate = threading.Event()

        def slow_llm(ctx):
            gate.wait(2)
            return hint_llm(ctx)

        published = []
        eng = DecisionEngine(llm_fn=slow_llm, publish=published.append, sync=False, llm_timeout=3)
        msgs = [json.loads(l) for l in (DATA / "collapse.jsonl").read_text(encoding="utf-8").splitlines()]
        for m in msgs:
            if "driver_response" in m:
                eng.on_driver_response(m["driver_response"], ts=m["ts"])
            else:
                eng.on_vision(m)
            if m.get("ts", 0) >= 4.0:
                break
        self.assertTrue(any(d["severity"] == 3 and d["source"] == "rules" for d in published),
                        "severitatea 3 publicată din reguli în timp ce LLM-ul încă lucrează")
        gate.set()
        time.sleep(0.3)
        self.assertGreaterEqual(eng.stats["llm_discarded"] + len(published), 1)

    def test_stale_llm_answer_discarded(self):
        eng = DecisionEngine(llm_fn=lambda c: '{"severity": 1, "actions": ["remind_road"], "reason": "vechi"}')
        msgs = [json.loads(l) for l in (DATA / "collapse.jsonl").read_text(encoding="utf-8").splitlines()]
        decs = []
        for m in msgs:
            if "driver_response" in m:
                eng.on_driver_response(m["driver_response"], ts=m["ts"])
            else:
                decs += eng.on_vision(m)
        after3 = [d for d in decs if d["ts"] > next(x["ts"] for x in decs if x["severity"] == 3)]
        self.assertTrue(all(d["severity"] == 3 for d in after3), "un LLM care subestimează nu coboară severitatea")


if __name__ == "__main__":
    unittest.main()
