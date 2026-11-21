"""Informațiile pentru asistentul Virgil: ora, vremea, traficul. Fără audio: testabil separat.

    info.answer("time")      -> {"request": "time", "text": "Este ora 19 și 52 de minute, ...", "source": "ceasul laptopului"}
    info.intent("Cum e vremea afară?") -> "weather"

* Vremea: Open-Meteo (gratuit, fără cheie). Pleacă doar coordonatele locației configurate.
* Traficul: TomTom Traffic Flow dacă există cheia în variabila de mediu TOMTOM_KEY; altfel Virgil spune
  cinstit că nu are date în timp real și dă doar o estimare după oră (nu inventează un ambuteiaj).
* Locația: variabilele VIGIL_LAT / VIGIL_LON / VIGIL_PLACE; implicit Cluj-Napoca, ca în info_assistant.py
  (în mașină ar veni de la GNSS).
Acesta e drumul offline / fără chei. Cu OMNIROUTE_API_KEY, butoanele din dashboard folosesc asistentul de
informare al echipei (info_assistant.py: Waze, Tavily, știri); vezi actions.assistant_lookup.
Textele sunt scurte, pentru voce: numerele se scriu cum se citesc în română („52 de minute”, „un grad”).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request

LAT = float(os.environ.get("VIGIL_LAT", "46.77"))      # aceeași locație ca info_assistant.py (CITY, LAT, LON)
LON = float(os.environ.get("VIGIL_LON", "23.59"))
PLACE = os.environ.get("VIGIL_PLACE", "Cluj-Napoca")
TOMTOM_KEY = os.environ.get("TOMTOM_KEY", "").strip()
HTTP_TIMEOUT_SEC = 4
WEATHER_TTL_SEC = 600      # vremea nu se schimbă de la un minut la altul: o cerere la 10 minute
TRAFFIC_TTL_SEC = 120
REQUESTS = ("time", "weather", "traffic")

log = logging.getLogger("ml2.info")

DAYS = ("luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică")
MONTHS = ("ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie",
          "octombrie", "noiembrie", "decembrie")

# codurile WMO folosite de Open-Meteo -> descriere + dacă merită o atenționare la volan
WEATHER_CODES = {
    0: ("cer senin", False), 1: ("cer predominant senin", False), 2: ("parțial noros", False),
    3: ("cer înnorat", False), 45: ("ceață", True), 48: ("ceață cu depunere de chiciură", True),
    51: ("burniță slabă", True), 53: ("burniță", True), 55: ("burniță densă", True),
    56: ("burniță care îngheață", True), 57: ("burniță care îngheață", True),
    61: ("ploaie slabă", True), 63: ("ploaie", True), 65: ("ploaie torențială", True),
    66: ("ploaie care îngheață", True), 67: ("ploaie care îngheață", True),
    71: ("ninsoare slabă", True), 73: ("ninsoare", True), 75: ("ninsoare abundentă", True), 77: ("grăunțe de zăpadă", True),
    80: ("averse slabe", True), 81: ("averse", True), 82: ("averse puternice", True),
    85: ("averse de ninsoare", True), 86: ("averse puternice de ninsoare", True),
    95: ("furtună", True), 96: ("furtună cu grindină", True), 99: ("furtună cu grindină", True),
}

_INTENTS = (
    ("traffic", re.compile(r"trafic|aglomera|ambuteiaj|coloan|drum(ul)? (e )?liber|blocaj|circula", re.I)),
    ("weather", re.compile(r"vrem|meteo|plou|ploaie|ning|zăpad|zapad|grade|temperatur|frig|cald|soare|ceaț|ceat|vânt|vant", re.I)),
    ("time", re.compile(r"\bor(a|ă)\b|cât e ceasul|cat e ceasul|ce oră|ce ora|\bceas|ce zi|data de azi|ce dată|ce data", re.I)),
)


def intent(text: str | None) -> str | None:
    """Ce cere șoferul: "traffic" | "weather" | "time" | None (altă întrebare, merge la LLM)."""
    for name, pattern in _INTENTS:
        if text and pattern.search(text):
            return name
    return None


# ------------------------------------------------------------ numere în română

def _count(n: int, one: str, many: str, fem: bool = False) -> str:
    """1 -> „un grad” / „o oră”; 2–19 -> „5 grade”; de la 20 -> „25 de grade” (regula lui „de” în română)."""
    if n == 1:
        return f"{'o' if fem else 'un'} {one}"
    return f"{n} {many}" if (abs(n) % 100) < 20 else f"{n} de {many}"


def degrees(t: float) -> str:
    n = round(t)
    if n < 0:
        return "minus " + _count(-n, "grad", "grade")
    return "zero grade" if n == 0 else _count(n, "grad", "grade")


# ------------------------------------------------------------------- ora

def time_text(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    m = now.minute
    minutes = "fix" if m == 0 else "și " + _count(m, "minut", "minute")
    return f"Este ora {now.hour} {minutes}, {DAYS[now.weekday()]}, {now.day} {MONTHS[now.month - 1]}."


# ----------------------------------------------------------------- rețea

def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "vigil-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as r:
        return json.loads(r.read())


_cache: dict[str, tuple[float, dict]] = {}


def _cached(key: str, ttl: float, fetch, url: str) -> dict:
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        return hit[1]
    data = fetch(url)
    _cache[key] = (time.monotonic(), data)
    return data


# ----------------------------------------------------------------- vremea

def weather_url(lat: float = LAT, lon: float = LON) -> str:
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "timezone": "auto",
         "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"}
    return "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q)


def weather_from(data: dict) -> str:
    """Răspunsul Open-Meteo -> o frază pentru voce, cu o atenționare când vremea contează la volan."""
    c = data["current"]
    temp, feels = float(c["temperature_2m"]), c.get("apparent_temperature")
    desc, risky = WEATHER_CODES.get(int(c.get("weather_code", -1)), ("vreme schimbătoare", False))
    text = f"Afară sunt {degrees(temp)}, {desc}"
    if isinstance(feels, (int, float)) and abs(round(feels) - round(temp)) >= 3:
        text += f", se simt ca {degrees(feels)}"
    wind = c.get("wind_speed_10m")
    if isinstance(wind, (int, float)) and wind >= 40:
        text += ", vânt puternic"
    text += "."
    wet = risky or (c.get("precipitation") or 0) > 0
    if wet and temp <= 1:
        text += " Atenție, poate fi polei: reduceți viteza."
    elif wet:
        text += " Carosabilul poate fi alunecos: păstrați distanța."
    return text


def weather_text(fetch=_get_json) -> tuple[str, str]:
    try:
        return weather_from(_cached("weather", WEATHER_TTL_SEC, fetch, weather_url())), "Open-Meteo"
    except Exception as e:                                  # noqa: BLE001 (fără internet, API schimbat...)
        log.warning("vremea indisponibilă: %r", e)
        return "Nu pot accesa datele meteo acum.", "indisponibil"


# ---------------------------------------------------------------- traficul

def traffic_url(key: str, lat: float = LAT, lon: float = LON) -> str:
    return ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?"
            + urllib.parse.urlencode({"point": f"{lat:.5f},{lon:.5f}", "unit": "KMPH", "key": key}))


def traffic_from(data: dict, place: str = PLACE) -> str:
    """TomTom Flow Segment Data -> frază: viteza actuală față de viteza fără trafic."""
    f = data["flowSegmentData"]
    if f.get("roadClosure"):
        return f"Pe {place}, drumul este închis. Căutați o rută ocolitoare."
    cur, free = float(f["currentSpeed"]), float(f["freeFlowSpeed"])
    ratio = cur / free if free > 0 else 1.0
    level = "fluid" if ratio >= 0.85 else "moderat" if ratio >= 0.6 else "aglomerat"
    text = f"Pe {place}, trafic {level}: {_count(round(cur), 'kilometru', 'kilometri')} pe oră"
    if level != "fluid":
        extra = round((float(f.get("currentTravelTime", 0)) - float(f.get("freeFlowTravelTime", 0))) / 60)
        if extra >= 1:
            text += f", cu o întârziere de {_count(extra, 'minut', 'minute')} pe acest tronson"
    return text + "."


def traffic_estimate(now: dt.datetime | None = None) -> str:
    """Fără sursă în timp real: o estimare după oră, spusă ca estimare."""
    now = now or dt.datetime.now()
    workday, h = now.weekday() < 5, now.hour
    if workday and (7 <= h < 10 or 16 <= h < 19):
        guess = "fiind oră de vârf, estimez trafic aglomerat la intrările și ieșirile din oraș"
    elif h >= 22 or h < 6:
        guess = "la ora asta, drumurile sunt de obicei libere"
    else:
        guess = "la ora asta, traficul este de obicei fluid"
    return f"Nu am date de trafic în timp real; {guess}."


def traffic_text(fetch=_get_json, key: str | None = None, now: dt.datetime | None = None) -> tuple[str, str]:
    key = TOMTOM_KEY if key is None else key
    if key:
        try:
            return traffic_from(_cached("traffic", TRAFFIC_TTL_SEC, fetch, traffic_url(key))), "TomTom"
        except Exception as e:                              # noqa: BLE001
            log.warning("traficul indisponibil: %r", e)
    return traffic_estimate(now), "estimare după oră"


# ------------------------------------------------------------------ total

def answer(request: str) -> dict:
    """Răspunsul complet pentru o cerere de la dashboard / o întrebare recunoscută."""
    if request == "time":
        text, source = time_text(), "ceasul laptopului"
    elif request == "weather":
        text, source = weather_text()
    elif request == "traffic":
        text, source = traffic_text()
    else:
        raise ValueError(f"cerere necunoscută: {request!r}")
    return {"request": request, "text": text, "source": source}


def facts() -> str:
    """Contextul pentru LLM la întrebările libere: doar date reale, fără rețea dacă vremea nu e în cache."""
    hit = _cache.get("weather")
    weather = weather_from(hit[1]) if hit and time.monotonic() - hit[0] < WEATHER_TTL_SEC else "necunoscută"
    return f"Date reale acum: {time_text()} Locația: {PLACE}. Vremea: {weather} Traficul în timp real: necunoscut."
