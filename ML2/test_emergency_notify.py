"""Teste pentru emergency_notify.py, fără rețea și fără telefon. Rulare din ML2/: python -m unittest test_emergency_notify"""
import json
import unittest

from emergency_notify import Notifier, build_message, load_env

CALL = {"severity": 3, "actions": [{"tool": "play_alarm", "args": {"level": 3}},
                                   {"tool": "call_112", "args": {"report": "Sofer fara reactie 8 s."}}]}
NL = chr(10)
ALARM_ONLY = {"severity": 3, "actions": [{"tool": "play_alarm", "args": {"level": 3}}]}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(send_results=(True,), phone="", user="@coleg", **kw):
    sent, calls, sleeps = [], [], []
    results = list(send_results)

    def send(u, text):
        sent.append((u, text))
        return (results.pop(0) if results else True), "raspuns"

    def call(adb, number):
        calls.append(number)
        return True, "Starting: Intent"

    clock = Clock()
    n = Notifier(telegram_user=user, phone=phone, send=send, call=call, clock=clock, sleep=sleeps.append, **kw)
    return n, sent, calls, sleeps, clock


class TestNotifier(unittest.TestCase):
    def test_only_call_112_triggers(self):
        n, sent, calls, _, _ = make(phone="0712345678")
        n.handle(json.dumps(ALARM_ONLY), background=False)
        n.handle("nu e json", background=False)
        n.handle(json.dumps({"severity": 3, "actions": [{"tool": "call_112", "args": {"report": "  "}}]}),
                 background=False)
        self.assertEqual((sent, calls), ([], []))
        n.handle(json.dumps(CALL), background=False)
        self.assertEqual(calls, ["0712345678"])
        self.assertEqual(len(sent), 1)
        self.assertIn("Sofer fara reactie", sent[0][1])
        self.assertIn("Șoferul pare inconștient", sent[0][1])
        self.assertIn("Demo hackathon", sent[0][1])

    def test_cooldown_one_notification_per_emergency(self):
        n, sent, _, _, clock = make()
        for _ in range(5):                                  # ML 1 repetă decizia la fiecare câteva secunde
            n.handle(json.dumps(CALL), background=False)
            clock.t += 20
        self.assertEqual(len(sent), 1)
        clock.t += 300
        n.handle(json.dumps(CALL), background=False)
        self.assertEqual(len(sent), 2)

    def test_retries_when_refused_then_gives_up(self):
        n, sent, _, sleeps, _ = make(send_results=(False, True))
        n.handle(json.dumps(CALL), background=False)
        self.assertEqual((len(sent), sleeps), (2, [70]))
        n2, sent2, _, sleeps2, _ = make(send_results=(False, False, False, False))
        self.assertFalse(n2._send_with_retries("x"))
        self.assertEqual((len(sent2), sleeps2), (3, [70, 70]))

    def test_dry_run_sends_nothing(self):
        n, sent, calls, _, _ = make(phone="0712345678", dry_run=True)
        n.handle(json.dumps(CALL), background=False)
        self.assertEqual((sent, calls), ([], []))

    def test_message_drops_signs_and_location(self):
        msg = build_message("Șofer fără reacție de 8 s. Semne: mână pe piept, respirație grea, cap căzut. "
                            "Locație simulată: A1 km 42.")
        self.assertIn("Detalii: Șofer fără reacție de 8 s.", msg)
        self.assertNotIn("Semne", msg)
        self.assertNotIn("A1 km 42", msg)
        self.assertNotIn("Detalii", build_message("Semne: cap căzut."))

    def test_real_ml1_report_drops_signs_and_location(self):
        real = ("APEL DE URGENȚĂ AUTOMAT – Vigil (SIMULAT)" + NL + "Ora: 00:57:17" + NL +
                "Locație: A1 București–Pitești, km 42, sens Pitești (44.4268, 25.8890)" + NL +
                "Suspiciune: șofer care nu reacționează (semne observate de cameră, nu diagnostic)" + NL +
                "Semne observate:" + NL + "- mână pe piept 8 s (încredere 80%)" + NL +
                "Răspuns la „Sunteți bine?”: niciun răspuns" + NL +
                "Scor atenție: 0.10 / 1" + NL + "Vehicul: avarii pornite, manevră de oprire în siguranță.")
        msg = build_message(real)
        for gone in ("Locație", "A1 București", "Semne observate", "mână pe piept", "APEL DE URGENȚĂ", "00:57:17"):
            self.assertNotIn(gone, msg)
        for kept in ("Suspiciune: șofer care nu reacționează", "niciun răspuns", "Scor atenție"):
            self.assertIn(kept, msg)

    def test_message_is_trimmed(self):
        msg = build_message("x " * 1000)
        self.assertLess(len(msg), 1000)

    def test_bad_phone_is_rejected_without_running_adb(self):
        from emergency_notify import call_phone
        ok, detail = call_phone("adb-care-nu-exista", "0712; rm -rf /")
        self.assertFalse(ok)
        self.assertIn("invalid", detail)


class TestEnv(unittest.TestCase):
    def test_load_env(self):
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("# comentariu\nVIGIL_TEST_A=@coleg   # nota\nVIGIL_TEST_B=\"07 1\"\n", encoding="utf-8")
            os.environ.pop("VIGIL_TEST_A", None)
            os.environ.pop("VIGIL_TEST_B", None)
            load_env(p)
            self.assertEqual(os.environ["VIGIL_TEST_A"], "@coleg")
            self.assertEqual(os.environ["VIGIL_TEST_B"], "07 1")


if __name__ == "__main__":
    unittest.main()
