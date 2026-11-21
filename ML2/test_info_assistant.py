"""Teste pentru asistentul de informare, fără rețea, microfon sau OmniRoute: python test_info_assistant.py"""
import info_assistant as ia


def test_route():
    assert ia.route("Ce vreme e afară?") == {"weather"}
    assert ia.route("Plouă azi în Cluj?") == {"weather"}
    assert ia.route("Care sunt știrile de azi?") == {"news"}
    assert ia.route("Cum e traficul spre centru?") == {"traffic"}
    assert ia.route("E aglomerat pe Calea Florești?") == {"traffic"}
    assert ia.route("Cine a câștigat meciul aseară?") == {"search"}


def test_strip_wake():
    assert ia.strip_wake("Hei Virgil, ce vreme e?") == "ce vreme e"
    assert ia.strip_wake("Hei Vigil") == ""


def test_weather_text():
    ia.http_json = lambda url, *a, **k: {
        "current": {"temperature_2m": 9.7, "weather_code": 3, "wind_speed_10m": 12},
        "daily": {"temperature_2m_max": [12], "temperature_2m_min": [4], "precipitation_probability_max": [20]}}
    text = ia.weather()
    assert "9.7 grade" in text and "înnorat" in text and "20%" in text


def test_traffic_uses_waze_then_news():
    queries = []
    real = ia.search, ia.traffic_waze
    ia.search = lambda q: queries.append(q) or "Știri: ..."
    ia.traffic_waze = lambda: "Waze, 1 alerte ..."  # fără apel real: fiecare cerere Waze costă
    try:
        assert ia.gather("Cum e traficul spre centru?") == "Waze, 1 alerte ...\nȘtiri: ..."
        assert queries == ["trafic Cluj-Napoca"]

        def waze_down():
            raise OSError("OpenWebNinja căzut")
        ia.traffic_waze = waze_down
        assert ia.gather("E aglomerat?") == "Știri: ..."  # Waze căzut: rămân știrile
    finally:
        ia.search, ia.traffic_waze = real


def test_search_omniroute_tavily_format():
    real = ia.http_json
    sent = []
    try:
        ia.http_json = lambda url, body=None, headers=None, timeout=10: sent.append((url, body)) or {"results": [
            {"title": "LPF", "snippet": "Sâmbătă, 21 nov:\n CFR Cluj - Oțelul 17:30"}]}
        assert ia.search("ce meciuri sunt azi") == "Căutare web (Tavily): LPF: Sâmbătă, 21 nov: CFR Cluj - Oțelul 17:30"
        assert sent[0][0].endswith("/api/v1/search") and sent[0][1]["provider"] == "tavily"

        ia.http_json = lambda *a, **k: {"results": []}
        assert ia.search("ceva fără rezultate") is None
    finally:
        ia.http_json = real


def test_dated_adds_today_only_for_time_words():
    import time
    today = time.strftime("%d.%m.%Y")
    assert ia.dated("Ce meciuri sunt azi în România?") == f"Ce meciuri sunt azi în România? {today}"
    assert ia.dated("Ce se joacă diseară?").endswith(today)
    assert ia.dated("Cine a scris Luceafărul?") == "Cine a scris Luceafărul?"


def test_waze_summary():
    # copiat din răspunsul real OpenWebNinja pentru Cluj
    alerts = [
        {"type": "POLICE", "subtype": None, "city": "Cluj-Napoca", "street": "Bd. Muncii", "num_thumbs_up": 1},
        {"type": "HAZARD", "subtype": "HAZARD_ON_SHOULDER_CAR_STOPPED", "city": "Feleacu", "street": "DN1",
         "num_thumbs_up": 9},
        {"type": "JAM", "subtype": "JAM_STAND_STILL_TRAFFIC", "city": "Cluj-Napoca", "street": None},
    ]
    text = ia.summarize_waze(alerts, [])
    assert text.startswith("Waze, 3 alerte")
    assert text.index("ambuteiaj") < text.index("poliție")  # ce încetinește traficul vine primul
    assert "pericol (on shoulder car stopped) pe DN1, Feleacu" in text
    assert "poliție pe Bd. Muncii" in text
    assert "nicio alertă" in ia.summarize_waze([], [])


def test_gather_survives_failed_source():
    def boom():
        raise OSError("fără rețea")
    ia.weather = boom
    assert ia.gather("Ce vreme e?") == ""


def sse(*chunks, model="m"):
    """răspuns în flux, ca de la OmniRoute: câte o linie „data:” pe bucată de text"""
    import json
    lines = [f"data: {json.dumps({'model': model, 'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    return lines + ["data: [DONE]"]


def with_stream(fn, gather="date"):
    real = ia.chat_lines, ia.gather
    ia.gather = lambda q: gather
    try:
        return fn()
    finally:
        ia.chat_lines, ia.gather = real


def test_answer_when_model_down():
    def down(body):
        raise OSError("OmniRoute oprit")
        yield  # generator, ca cel real

    def run():
        ia.chat_lines = down
        assert ia.answer("Ce vreme e?") is None
    with_stream(run)


def test_answer_empty_content_is_no_answer():
    # modelele care „gândesc” pot trimite doar raționament, fără text de răspuns
    def run():
        ia.chat_lines = lambda body: sse(None, None)
        assert ia.answer("Care sunt știrile?") is None
    with_stream(run)


def test_general_question_answered_from_tavily_context():
    prompts = []

    def run():
        ia.chat_lines = lambda body: prompts.append(body["messages"][1]["content"]) or sse("**2-1** pentru ", "România [1].")
        assert ia.answer("Cât s-a terminat România-Polonia?") == "2-1 pentru România."  # fără markdown și trimiteri
        assert "România – Polonia 2-1" in prompts[0]  # modelul primește rezultatele căutării
    with_stream(run, gather="Căutare web (Tavily): România – Polonia 2-1")


def test_answer_falls_back_to_next_model_on_rate_limit():
    import urllib.error
    tried = []

    def fake(body):
        tried.append(body["model"])
        if body["model"] == ia.MODELS[0]:  # prima: limita gratuită depășită
            raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)
        yield from sse("Plouă ușor.")

    def run():
        ia.chat_lines = fake
        assert ia.answer("Ce vreme e afară?") == "Plouă ușor."
        assert tried == ia.MODELS[:2]
    with_stream(run)


def test_first_sentence_comes_before_the_model_finishes():
    progress = []

    def slow(body):
        for line in sse("Azi plouă. ", "Mâine e soare. ", "Poimâine ", "ninge."):
            progress.append(line)
            yield line

    def run():
        ia.chat_lines = slow
        sentences = ia.stream_reply("Ce vreme e?")
        assert next(sentences) == "Azi plouă."
        assert len(progress) < 4  # prima propoziție a plecat înainte ca modelul să termine
        assert list(sentences) == ["Mâine e soare.", "Poimâine ninge."]
    with_stream(run)


def test_conversation_keeps_listening_until_silence_or_thanks():
    import threading
    real = ia.stream_reply, ia.actions.speak, ia.actions.listen, ia.actions.play_alarm
    heard = lambda text: ({"answered": bool(text), "latency_sec": None, "text": text}, None, {})
    try:
        ia.stream_reply = lambda q: iter([f"răspuns la {q}"])
        said = []
        ia.actions.speak = lambda text, interrupt: said.append(text) or True
        ia.actions.play_alarm = lambda level, interrupt: None

        # a doua întrebare vine fără „Hei Virgil”; tăcerea încheie; „Caut imediat.” cât timp caută
        follow = [heard("Și mâine cum va fi vremea?"), heard(None)]
        ia.actions.listen = lambda interrupt: follow.pop(0)
        ia.conversation("Ce vreme e azi?", threading.Event())
        assert said == [ia.SEARCHING, "răspuns la Ce vreme e azi?",
                        ia.SEARCHING, "răspuns la Și mâine cum va fi vremea?"]

        # „e suficient” încheie, fără să mai caute
        said.clear()
        follow = [heard("E suficient, mersi.")]
        ia.conversation("Ce meciuri sunt azi?", threading.Event())
        assert said == [ia.SEARCHING, "răspuns la Ce meciuri sunt azi?"]
    finally:
        ia.stream_reply, ia.actions.speak, ia.actions.listen, ia.actions.play_alarm = real


def test_ends_conversation():
    for text in ["Mersi", "Mulțumesc, atât.", "E suficient.", "Bine, mersi!", "Gata", "Ajunge", "E destul",
                 "Nu", "Nu mai", "Pa", "La revedere"]:
        assert ia.ends_conversation(text), text
    for text in ["Mersi, și mâine cum e vremea?", "Cum e traficul?", "Nu știu ce meciuri sunt azi, îmi spui?"]:
        assert not ia.ends_conversation(text), text


def test_stream_keeps_times_whole_and_stops_at_three_sentences():
    def run():
        ia.chat_lines = lambda body: sse("Azi sunt trei meciuri. Csikszereda joacă la ora 15", ".00 cu Corvinul. ",
                                         "CFR Cluj joacă la 17.30. Dinamo joacă diseară. Mai sunt și în Liga 2!")
        assert list(ia.stream_reply("Ce meciuri sunt azi?")) == [
            "Azi sunt trei meciuri.", "Csikszereda joacă la ora 15.00 cu Corvinul.", "CFR Cluj joacă la 17.30."]
        ia.chat_lines = lambda body: sse("O singură propoziție")
        assert list(ia.stream_reply("?")) == ["O singură propoziție"]  # fără punct la final: tot e citită
    with_stream(run)


def test_answer_is_shortened():
    def run():
        ia.chat_lines = lambda body: sse("Unu. Doi. ", "Trei. Patru. Cinci.")
        assert ia.answer("Ce mai e nou?") == "Unu. Doi. Trei."
    with_stream(run)


def test_safety_silences_on_alert_only():
    s = ia.Safety()
    s.on_decision(b'{"severity": 0, "actions": []}', now=0)
    assert not s.quiet(now=1) and not s.interrupt.is_set()
    s.on_decision(b'{"severity": 2, "actions": []}', now=0)
    assert s.interrupt.is_set() and s.quiet(now=ia.QUIET_SEC - 1) and not s.quiet(now=ia.QUIET_SEC + 1)
    s.on_decision(b"{stricat", now=0)  # mesaj stricat: ignorat, nu cade


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
