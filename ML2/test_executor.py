"""Teste fără MQTT și fără audio: python test_executor.py (sau pytest)."""
import json
import threading
import time

from executor import CarLink, Executor, validate


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


def test_virgil_goes_before_level_1_2_debounce_and_dropped_by_emergency():
    ex, calls, _ = make()
    ex.handle(decision(1, ("speak", {"text": "privirea la drum"})))
    ex.handle(decision(2, ("suggest_break", {})))
    ex.handle_assistant(b'{"request": "weather"}')
    ex.handle_assistant(b'{"request": "weather"}')          # dublu-clic
    ex.handle_assistant(b'{"request": "rm -rf"}')           # cerere necunoscută
    ex.handle_assistant(b"{not json")
    assert ex.q.qsize() == 3
    for _ in range(3):
        ex.step()
    assert calls == ["assistant", "suggest_break", "speak"]  # Virgil, chiar dacă a fost chemat ultimul
    ex.handle_assistant(b'{"request": "time"}')
    ex.handle(decision(3, ("call_112", {"report": "r"})))
    assert [item[2] for item in ex.q.queue] == ["call_112"]  # urgența anulează cererea asistentului


def test_emergency_still_goes_before_virgil():
    ex, calls, _ = make()
    ex.handle_assistant(b'{"request": "time"}')
    ex.handle(decision(3, ("play_alarm", {"level": 3})))
    ex.step()
    assert calls == ["play_alarm"]


def test_only_one_virgil_request_waits():
    ex, calls, _ = make()
    ex.handle_assistant(b'{"request": "time"}')
    ex.handle_assistant(b'{"request": "traffic"}')          # răzgândit: contează ultima apăsare
    assert [(item[2], item[3]) for item in ex.q.queue] == [("assistant", {"request": "traffic"})]


def running(ex, tool):
    while ex.current_tool != tool:
        time.sleep(0.01)


def test_virgil_not_interrupted_by_level_1_2_only_by_emergency():
    interrupted = threading.Event()

    def slow(tool, args, interrupt):
        if tool == "assistant" and interrupt.wait(5):
            interrupted.set()

    ex = Executor(slow, lambda level: None)
    ex.handle_assistant(b'{"request": "listen"}')
    threading.Thread(target=ex.step, daemon=True).start()
    running(ex, "assistant")
    ex.handle(json.dumps({"severity": 0, "actions": []}))   # revenire la normal: nu întrerupe
    ex.handle(decision(1, ("speak", {"text": "privirea la drum"})))
    ex.handle(decision(2, ("play_alarm", {"level": 2}), ("ask_driver", {"question": "Obosit?"})))
    assert not ex.interrupt.is_set()                         # așteaptă în coadă, după Virgil
    assert ex.q.qsize() == 3
    ex.handle(decision(3, ("play_alarm", {"level": 3})))
    assert interrupted.wait(1)                               # urgența taie și vorba lui Virgil


def test_virgil_cuts_a_running_reminder_but_not_an_ml1_question():
    cut = []

    def slow(tool, args, interrupt):
        if interrupt.wait(5):
            cut.append(tool)

    for tool, action, expect_cut in (("speak", ("speak", {"text": "privirea la drum"}), True),
                                     ("play_alarm", ("play_alarm", {"level": 2}), True),
                                     ("ask_driver", ("ask_driver", {"question": "Obosit?"}), False)):
        cut.clear()
        ex = Executor(slow, lambda level: None)
        ex.handle(decision(2, action))
        worker = threading.Thread(target=ex.step, daemon=True)
        worker.start()
        running(ex, tool)
        ex.handle_assistant(b'{"request": "time"}')
        assert ex.interrupt.is_set() is expect_cut, tool
        if expect_cut:
            worker.join(1)
            assert cut == [tool]
            assert [item[2] for item in ex.q.queue] == ["assistant"]   # Virgil urmează imediat
        else:
            ex.interrupt.set()                               # oprim testul: întrebarea ML 1 nu e tăiată de Virgil
            worker.join(1)


def assistant_run(request, answers=(), reply=None, omniroute=None, interrupt=None):
    """assistant_request cu voce, microfon, LLM și internet false -> (ce a spus Virgil, ce s-a publicat).
    omniroute: răspunsul fals al lui info_assistant.answer (cu cheia OMNIROUTE setată); None = fără cheie."""
    import os

    import actions
    import info
    import info_assistant
    said, published, answers = [], [], list(answers)
    actions.speak = lambda text, interrupt: said.append(text) or True
    actions.listen = lambda interrupt: ({"answered": bool(a := answers.pop(0)), "latency_sec": 1.0, "text": a}, None, {})
    actions.chat = lambda history: published.append(("chat", history)) or reply
    info.weather_text = lambda fetch=None: ("Afară sunt 7 grade, cer înnorat.", "Open-Meteo")
    info_assistant.answer = omniroute if callable(omniroute) else (lambda q: published.append(("omni", q)) or omniroute)
    saved = os.environ.pop("OMNIROUTE_API_KEY", None)      # testele nu depind de cheia de pe calculator
    if omniroute is not None:
        os.environ["OMNIROUTE_API_KEY"] = "test"
    try:
        actions.assistant_request(request, interrupt or threading.Event(),
                                  lambda t, p: published.append((t, json.loads(p))))
    finally:
        os.environ.pop("OMNIROUTE_API_KEY", None)
        if saved is not None:
            os.environ["OMNIROUTE_API_KEY"] = saved
    return said, published


def test_assistant_quick_buttons_and_no_driver_response():
    said, pub = assistant_run("weather")
    assert said == ["Afară sunt 7 grade, cer înnorat."]
    topics = {t for t, _ in pub}
    assert topics == {"vigil/assistant/reply"}               # nimic pe vigil/driver_response (ML 1)
    assert pub[-1][1]["status"] == "done" and pub[-1][1]["source"] == "Open-Meteo"
    said, _ = assistant_run("time")
    assert said[0].startswith("Este ora ")


def test_assistant_listens_answers_and_silence_is_not_an_alarm():
    said, pub = assistant_run("listen", ["Hei, cum e vremea afară?"])
    assert said == ["Vă ascult.", "Afară sunt 7 grade, cer înnorat."]
    assert pub[-1][1]["heard"] == "Hei, cum e vremea afară?"
    said, pub = assistant_run("listen", [None])
    assert said == ["Vă ascult.", "Nu v-am auzit. Apăsați din nou când doriți."]
    assert all(t == "vigil/assistant/reply" for t, _ in pub)


def test_assistant_free_question_goes_to_llm_with_real_facts_only():
    said, pub = assistant_run("listen", ["Cine a câștigat meciul aseară?"], reply=("Nu am această informație.", True))
    assert said == ["Vă ascult.", "Nu am această informație."]
    history = next(p for t, p in pub if t == "chat")
    assert "Date reale acum: Este ora" in history[1]["content"]
    assert pub[-1][1]["source"] == "LLM local"


def test_assistant_uses_team_info_assistant_with_omniroute_and_falls_back():
    said, pub = assistant_run("traffic", omniroute="Waze: un accident pe Calea Florești, ocoliți prin centură.")
    assert said == ["Waze: un accident pe Calea Florești, ocoliți prin centură."]
    assert ("omni", "Cum e traficul acum în zonă?") in pub
    said, pub = assistant_run("weather", omniroute="")             # OmniRoute nu răspunde -> Open-Meteo direct
    assert said == ["Afară sunt 7 grade, cer înnorat."] and pub[-1][1]["source"] == "Open-Meteo"
    said, pub = assistant_run("time", omniroute="nu trebuie folosit")  # ora vine mereu de la ceas, instant
    assert said[0].startswith("Este ora ") and not any(t == "omni" for t, _ in pub)


def test_alert_abandons_slow_lookup_immediately():
    interrupt = threading.Event()

    def slow(question):
        time.sleep(3)
        return "prea târziu"

    threading.Timer(0.2, interrupt.set).start()
    start = time.monotonic()
    said, pub = assistant_run("weather", omniroute=slow, interrupt=interrupt)
    assert time.monotonic() - start < 1                              # workerul e liber pentru urgență
    assert said == [] and pub[-1][1]["status"] == "interrupted"


# --- ML1 vigil/ecu -> ECU_Arduino vigil/car/cmd (CarLink, din ML2_Pasca) ---

def ecu(mode, target, hazards=False, decel=0.5):
    """un mesaj vigil/ecu de la ML1 (doar câmpurile pe care le folosește traducerea)"""
    return json.dumps({"mode": mode, "target_kmh": target, "speed_ref_kmh": 100.0, "decel_mps2": decel,
                       "hazards": hazards})


def cmds(out):
    return [{k: v for k, v in c.items() if k != "id"} for c in out]


def test_car_link_translates_only_changes():
    car = CarLink()
    assert car.update(ecu("NORMAL", 100.0)) == []                       # croazieră: nimic de trimis
    assert cmds(car.update(ecu("CAUTION", 73.0))) == [{"cmd": "slow_down", "target_kmh": 73.0, "decel_mps2": 0.5}]
    assert car.update(ecu("CAUTION", 73.0)) == []                       # 5 Hz: fără repetări
    assert car.update(ecu("CAUTION", 70.0)) == []                       # sub 5 km/h diferență: nu spamăm ECU
    assert cmds(car.update(ecu("ASSIST", 50.0, hazards=True, decel=2.0))) == [
        {"cmd": "hazard", "on": True}, {"cmd": "slow_down", "target_kmh": 50.0, "decel_mps2": 2.0}]
    assert cmds(car.update(ecu("MRM", 0.0, hazards=True, decel=2.5))) == [{"cmd": "emergency", "decel_mps2": 2.5}]
    assert car.update(ecu("MRM", 0.0, hazards=True)) == []              # o singură comandă de urgență
    # ML1 iese din MRM doar la severitate 0; ECU nu are „accelerează”, deci revenim cu reset
    assert cmds(car.update(ecu("NORMAL", 100.0))) == [
        {"cmd": "cancel_emergency"}, {"cmd": "hazard", "on": False}, {"cmd": "reset"}]
    assert car.update(ecu("NORMAL", 100.0)) == []


def test_car_link_after_reset_compares_with_ecu_cruise():
    # după reset ECU e la croazieră (100 km/h), chiar dacă referința ML1 încă urcă de la 0 (găsit în scenariul demo)
    car = CarLink()
    car.update(ecu("MRM", 0.0, hazards=True))
    car.update(ecu("NORMAL", 100.0))                                    # reset
    ramping = json.dumps({"mode": "ASSIST", "target_kmh": 50.0, "speed_ref_kmh": 47.0, "decel_mps2": 2.0,
                          "hazards": True})
    assert cmds(car.update(ramping)) == [{"cmd": "hazard", "on": True},
                                         {"cmd": "slow_down", "target_kmh": 50.0, "decel_mps2": 2.0}]


def test_car_link_ids_and_bad_messages():
    car = CarLink()
    out = car.update(ecu("MRM", 0.0, hazards=True)) + car.update(ecu("NORMAL", 100.0))
    ids = [c["id"] for c in out]
    assert all(i.startswith("ml2-") for i in ids) and len(set(ids)) == len(ids)  # ECU le întoarce în evenimente
    for bad in [b"{x", b"[]", b'{"mode": 3, "target_kmh": 50}', b'{"mode": "CAUTION", "target_kmh": "50"}']:
        assert car.update(bad) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
