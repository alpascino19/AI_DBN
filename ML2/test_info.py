"""Teste pentru info.py (ora, vremea, traficul), fără internet: python test_info.py (sau pytest)."""
import datetime as dt

import info

SAT_EVENING = dt.datetime(2026, 11, 21, 19, 52)   # sâmbătă
MON_PEAK = dt.datetime(2026, 11, 23, 8, 15)       # luni, oră de vârf

OPEN_METEO = {"current": {"temperature_2m": 7.4, "apparent_temperature": 3.6, "precipitation": 0.0,
                          "weather_code": 3, "wind_speed_10m": 14.0}}
TOMTOM = {"flowSegmentData": {"currentSpeed": 41, "freeFlowSpeed": 98, "currentTravelTime": 420,
                              "freeFlowTravelTime": 180, "roadClosure": False}}


def test_intent():
    cases = {"Cât e ceasul?": "time", "Ce oră e?": "time", "Cum e vremea?": "weather", "Plouă la Pitești?": "weather",
             "Câte grade sunt afară?": "weather", "E trafic pe autostradă?": "traffic", "E aglomerat la ora asta?": "traffic",
             "Spune-mi o glumă": None, "": None, None: None}
    for text, want in cases.items():
        assert info.intent(text) == want, text


def test_romanian_numbers():
    assert info.degrees(1) == "un grad" and info.degrees(7.4) == "7 grade" and info.degrees(25) == "25 de grade"
    assert info.degrees(0.2) == "zero grade" and info.degrees(-3) == "minus 3 grade"
    assert info.time_text(SAT_EVENING) == "Este ora 19 și 52 de minute, sâmbătă, 21 noiembrie."
    assert info.time_text(dt.datetime(2026, 11, 23, 8, 0)) == "Este ora 8 fix, luni, 23 noiembrie."
    assert info.time_text(dt.datetime(2026, 11, 23, 8, 1)).startswith("Este ora 8 și un minut")


def test_weather_sentence_and_driving_warnings():
    assert info.weather_from(OPEN_METEO) == "Afară sunt 7 grade, cer înnorat, se simt ca 4 grade."
    rain = {"current": {**OPEN_METEO["current"], "weather_code": 63, "apparent_temperature": 6}}
    assert info.weather_from(rain).endswith("Carosabilul poate fi alunecos: păstrați distanța.")
    ice = {"current": {**OPEN_METEO["current"], "temperature_2m": 0.5, "weather_code": 61, "apparent_temperature": 0}}
    assert "polei" in info.weather_from(ice)


def test_weather_fetch_cache_and_failure():
    info._cache.clear()
    calls = []
    fetch = lambda url: calls.append(url) or OPEN_METEO
    assert info.weather_text(fetch)[1] == "Open-Meteo"
    assert info.weather_text(fetch)[1] == "Open-Meteo" and len(calls) == 1   # al doilea din cache
    assert "latitude=46.7700" in calls[0] and "key" not in calls[0]          # doar coordonatele locației
    info._cache.clear()

    def down(url):
        raise OSError("fără internet")
    assert info.weather_text(down) == ("Nu pot accesa datele meteo acum.", "indisponibil")


def test_traffic_tomtom_and_honest_estimate():
    assert info.traffic_from(TOMTOM, "A1") == (
        "Pe A1, trafic aglomerat: 41 de kilometri pe oră, cu o întârziere de 4 minute pe acest tronson.")
    assert "fluid" in info.traffic_from({"flowSegmentData": {"currentSpeed": 95, "freeFlowSpeed": 100}}, "A1")
    assert "închis" in info.traffic_from({"flowSegmentData": {"roadClosure": True}}, "A1")
    info._cache.clear()
    calls = []
    text, source = info.traffic_text(lambda url: calls.append(url) or TOMTOM, key="", now=MON_PEAK)
    assert calls == [] and source == "estimare după oră"                  # fără cheie: nicio cerere
    assert text.startswith("Nu am date de trafic în timp real") and "oră de vârf" in text
    assert "libere" in info.traffic_estimate(dt.datetime(2026, 11, 23, 23, 30))

    def down(url):
        raise OSError("401")
    assert info.traffic_text(down, key="X", now=SAT_EVENING)[1] == "estimare după oră"
    info._cache.clear()
    assert info.traffic_text(lambda url: TOMTOM, key="X")[1] == "TomTom"
    info._cache.clear()


def test_answer_rejects_unknown():
    try:
        info.answer("bitcoin")
    except ValueError:
        return
    raise AssertionError("cererea necunoscută trebuia respinsă")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
