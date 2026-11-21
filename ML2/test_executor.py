"""Teste fără MQTT și fără audio: python test_executor.py (sau pytest)."""
import json
import threading
import time

from executor import Executor, validate


def make():
    calls, alerts = [], []
    ex = Executor(lambda tool, args, interrupt: calls.append(tool), alerts.append)
    return ex, calls, alerts


def decision(sev, *actions):
    return json.dumps({"severity": sev, "actions": [{"tool": t, "args": a} for t, a in actions]})


def test_validate():
    assert validate({"tool": "play_alarm", "args": {"level": 2}}) == ("play_alarm", {"level": 2})
    assert validate({"tool": "suggest_break", "args": {}})
    assert validate({"tool": "suggest_break"})
    assert not validate({"tool": "rm_rf", "args": {}})
    assert not validate({"tool": "play_alarm", "args": {"level": 9}})
    assert not validate({"tool": "play_alarm", "args": {"level": True}})
    assert not validate({"tool": "play_alarm", "args": {"level": "2"}})
    assert not validate({"tool": "speak", "args": {"text": "  "}})
    assert not validate({"tool": "speak", "args": {"text": "x", "extra": 1}})
    assert not validate("speak")


def test_bad_payloads_ignored():
    ex, _, _ = make()
    for bad in [b"{not json", b"[]", b'{"severity": 5, "actions": []}', b'{"severity": 2}']:
        ex.handle(bad)
    assert ex.q.empty()


def test_severity0_back_to_normal_turns_esp32_off():
    # ML 1 trimite {"severity":0,"actions":[]} la revenirea la normal
    ex, calls, alerts = make()
    ex.handle(json.dumps({"severity": 0, "actions": [], "reason": "revenire la normal"}))
    assert alerts == [0] and ex.q.empty() and calls == []


def test_priority_and_esp32_direct():
    ex, calls, alerts = make()
    ex.handle(decision(1, ("speak", {"text": "privirea la drum"})))
    ex.handle(decision(2, ("set_esp32", {"level": 2}), ("suggest_break", {})))
    assert alerts == [2] and ex.q.qsize() == 2
    ex.step()
    ex.step()
    assert calls == ["suggest_break", "speak"]


def test_emergency_drops_stale_actions():
    ex, calls, _ = make()
    ex.handle(decision(2, ("suggest_break", {}), ("ask_driver", {"question": "Obosit?"})))
    ex.handle(decision(1, ("speak", {"text": "privirea la drum"})))
    ex.handle(decision(3, ("call_112", {"report": "r"})))
    assert ex.q.qsize() == 1
    ex.step()
    assert calls == ["call_112"]


def test_speak_interrupted_during_slow_synthesis():
    import actions
    real = actions._synth
    actions._synth = lambda text: time.sleep(3) or (None, 1)  # edge-tts lent
    try:
        interrupt = threading.Event()
        threading.Timer(0.2, interrupt.set).start()
        start = time.monotonic()
        assert actions.speak("text lung", interrupt) is False
        assert time.monotonic() - start < 1
    finally:
        actions._synth = real


def test_debounce_allows_escalation():
    ex, _, _ = make()
    ex.handle(decision(2, ("play_alarm", {"level": 2})))
    ex.handle(decision(2, ("play_alarm", {"level": 2})))
    assert ex.q.qsize() == 1  # a doua alarmă identică e blocată
    ex.handle(decision(3, ("play_alarm", {"level": 3})))
    assert [item[0] for item in ex.q.queue] == [-3]  # escaladarea trece, alarma veche e anulată


def test_severity3_interrupts():
    interrupted = threading.Event()

    def slow(tool, args, interrupt):
        if tool == "speak" and interrupt.wait(5):
            interrupted.set()

    ex = Executor(slow, lambda level: None)
    ex.handle(decision(1, ("speak", {"text": "lung"})))
    threading.Thread(target=ex.step, daemon=True).start()
    while ex.current != 1:
        time.sleep(0.01)
    ex.handle(decision(3, ("call_112", {"report": "r"})))
    assert interrupted.wait(1)


def converse(answers, replies):
    """Rulează ask_driver cu voce, microfon și LLM false -> (ce a spus Vigil, ce s-a publicat)."""
    import actions
    said, published, answers, replies = [], [], list(answers), list(replies)
    actions.speak = lambda text, interrupt: said.append(text) or True
    actions.listen = lambda interrupt: ({"answered": bool(a := answers.pop(0)), "latency_sec": 1.0 if a else None,
                                         "text": a}, None, {})
    actions.chat = lambda history: replies.pop(0)
    actions.ask_driver("Sunteți bine?", threading.Event(), lambda t, p: published.append(json.loads(p)))
    return said, published


def test_conversation_silence_stops_immediately():
    said, published = converse([None], [])
    assert said == ["Sunteți bine?"] and published == [{"answered": False, "latency_sec": None, "text": None}]


def test_conversation_follows_up_and_ends():
    said, published = converse(["Sunt obosit", "Da, opresc"],
                               [("Vă propun o cafea. Opriți?", False), ("Perfect, drum bun!", True)])
    assert said == ["Sunteți bine?", "Vă propun o cafea. Opriți?", "Perfect, drum bun!"]
    assert [p["text"] for p in published] == ["Sunt obosit", "Da, opresc"]


def test_conversation_capped_and_survives_llm_down():
    import actions
    said, published = converse(["a", "b", "c"], [("1", False), ("2", False), ("închei", False)])
    assert len(published) == actions.MAX_TURNS and said[-1] == "închei"
    said, published = converse(["a"], [None])
    assert said == ["Sunteți bine?"] and len(published) == 1


def test_chat_reply_without_question_ends():
    import io
    import importlib
    import urllib.request
    import actions
    importlib.reload(actions)  # alte teste au înlocuit actions.chat cu variante false
    real = urllib.request.urlopen

    def fake(content):
        body = json.dumps({"message": {"content": json.dumps(content)}}).encode()
        urllib.request.urlopen = lambda request, timeout: io.BytesIO(body)

    try:
        fake({"reply": "Excelent, continuați cu atenție.", "end": False})
        assert actions.chat([]) == ("Excelent, continuați cu atenție.", True)
        fake({"reply": "De când conduceți fără pauză?", "end": False})
        assert actions.chat([]) == ("De când conduceți fără pauză?", False)
        fake({"reply": "", "end": True})
        assert actions.chat([]) is None
    finally:
        urllib.request.urlopen = real


def test_wake_word():
    from actions import heard_wake_word
    for text in ["Hei Virgil", "hei vigil, ce faci?", "Hey, Virgilă!", "Virgil, pune muzică"]:
        assert heard_wake_word(text), text
    for text in ["Bună ziua", "Ce fragil e paharul", "", None]:
        assert not heard_wake_word(text), text


def test_assistant_idle_wake_alert_and_back_to_idle():
    import actions
    said, alerts = [], []
    answers = ["Bună ziua, ce mai faci", None,                  # idle: fără „Hei Virgil” -> ignorat
               "Hei Virgil, ce vreme e?",                         # trezire
               None,                                              # tăcere în conversație -> alarmă
               "Sunt bine, mulțumesc",                            # e bine -> end=true -> idle
               "Cum e traficul?"]                                 # idle din nou: ignorat
    replies = [("E soare. Cum vă simțiți?", False), ("Perfect, vă las în pace.", True)]
    actions.speak = lambda text, interrupt: said.append(text) or True
    # când lista se termină, listen întoarce None (ca la o întrerupere) și asistentul se oprește
    actions.listen = lambda interrupt: (({"answered": bool(a := answers.pop(0)), "latency_sec": None, "text": a},
                                         None, {}) if answers else None)
    actions.chat = lambda history: replies.pop(0)
    actions.assistant(threading.Event(), lambda t, p: None, lambda: alerts.append(1))
    assert said == ["E soare. Cum vă simțiți?", "Perfect, vă las în pace."]
    assert alerts == [1] and replies == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
