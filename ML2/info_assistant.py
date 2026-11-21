"""Virgil — asistentul vocal de informare, separat de executorul ML 2 (siguranța).

„Hei Virgil, <întrebare>” -> date live (vreme: Open-Meteo; trafic: Waze; știri: RSS; orice altceva: căutare
Tavily prin OmniRoute) -> model prin OmniRoute -> voce. După răspuns ascultă și întrebările următoare.
Tace imediat ce ML 2 are o alertă (vigil/decision cu severitate >= 1): siguranța are mereu prioritate.

    python info_assistant.py
Cheia OmniRoute se citește din variabila de mediu OMNIROUTE_API_KEY.
"""
import json
import logging
import os
import queue
import re
import threading
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

import actions
from executor import BROKER

OMNIROUTE = "http://localhost:20128"
# Direct, nu combo-ul „claude-code”: acela are strategia lkgp (rămâne pe ultimul furnizor care a mers, de obicei
# nemotron, 12-34 s) și e folosit și de Claude Code. gemini-2.5-flash: ~6 s, răspunsuri corecte.
MODEL = os.environ.get("VIRGIL_MODEL", "gemini/gemini-2.5-flash")
# planul gratuit Gemini dă 429 după câteva cereri pe minut: atunci încercăm următorul model
MODELS = [MODEL] + [m for m in ("auto/best-fast", "claude-code") if m != MODEL]
CITY, LAT, LON = "Cluj-Napoca", 46.77, 23.59
NEWS_FEEDS = ("https://hotnews.ro/feed", "https://www.digi24.ro/rss")
QUIET_SEC = 60  # după o alertă de siguranță Virgil nu mai vorbește atâta timp

WMO = {0: "senin", 1: "predominant senin", 2: "parțial înnorat", 3: "înnorat", 45: "ceață", 48: "ceață",
       51: "burniță", 53: "burniță", 55: "burniță", 61: "ploaie slabă", 63: "ploaie", 65: "ploaie puternică",
       71: "ninsoare slabă", 73: "ninsoare", 75: "ninsoare puternică", 80: "averse", 81: "averse",
       82: "averse puternice", 95: "furtună", 96: "furtună cu grindină", 99: "furtună cu grindină"}

PERSONA = """Ești Virgil, asistentul de bord al mașinii, calm și politicos. Șoferul conduce, deci fii scurt.
Răspunzi în română naturală, citită cu voce tare: una până la trei propoziții, fără liste, simboluri, emoji, \
linkuri sau markdown. Scrie unitățile în cuvinte (grade, kilometri pe oră, minute) și rotunjește numerele \
(10 grade, nu 9,7).
Pentru vreme, trafic, știri, sport și orice eveniment recent folosești DOAR datele live primite (rezultatele \
căutării au data publicării: ține cont de ea când spui „ultimul”); dacă lipsesc, spui că nu ai informația acum. \
La întrebări despre „azi”/„diseară”/„mâine” iei DOAR evenimentele cu acea dată; ignori ce s-a jucat sau s-a \
întâmplat în alte zile, chiar dacă apare în rezultate. „În România” înseamnă ce se joacă în țară (ex. Superliga), \
nu doar naționala. \
Nu inventa. Locația mașinii: {city}. Data și ora: {now}."""

log = logging.getLogger("virgil")


def http_json(url, body=None, headers=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return json.loads(r.read())


def omniroute_headers():
    return {"Authorization": f"Bearer {os.environ.get('OMNIROUTE_API_KEY', '')}"}


def plain(text):
    """minuscule fără diacritice, pentru potrivirea cuvintelor-cheie"""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def route(question):
    """Ce surse live cer întrebarea: subset din {weather, news, traffic, search}."""
    q = plain(question)
    topics = set()
    if re.search(r"vreme|ploua|ploaie|ninge|temperatur|grade|frig|cald|soare|umbrel", q):
        topics.add("weather")
    if re.search(r"stir|noutat|ce se (mai )?intampla|evenimente", q):
        topics.add("news")
    if re.search(r"trafic|aglomer|accident|coloan|coada|drum|blocat|ambuteiaj|lucrari", q):
        topics.add("traffic")
    return topics or {"search"}


def weather():
    d = http_json(f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
                  "&current=temperature_2m,weather_code,wind_speed_10m"
                  "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                  "&timezone=Europe%2FBucharest&forecast_days=1")
    c, day = d["current"], d["daily"]
    return (f"Vremea în {CITY} acum: {c['temperature_2m']} grade, {WMO.get(c['weather_code'], 'variabil')}, "
            f"vânt {c['wind_speed_10m']} km/h. Azi: maxima {day['temperature_2m_max'][0]} grade, "
            f"minima {day['temperature_2m_min'][0]} grade, șanse de precipitații "
            f"{day['precipitation_probability_max'][0]}%.")


def rss_items(url):
    """Un flux RSS -> [(titlu, descriere fără HTML, data)]."""
    # unele site-uri (HotNews) refuză cererile fără identificare de browser
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=10) as r:
        root = ET.fromstring(r.read())
    return [(i.findtext("title", "").strip(), re.sub(r"<[^>]+>", "", i.findtext("description", "")).strip(),
             (i.findtext("pubDate") or "")[:16]) for i in root.iter("item") if i.findtext("title")]


def news(limit=6):
    for url in NEWS_FEEDS:
        try:
            items = rss_items(url)[:limit]
        except Exception as e:
            log.warning("RSS %s: %r", url, e)
            continue
        if items:
            return "Titluri de știri de azi: " + " | ".join(t for t, _, _ in items)
    return None


WAZE_TYPES = {"ACCIDENT": "accident", "ROAD_CLOSED": "drum închis", "JAM": "ambuteiaj",
              "CONSTRUCTION": "lucrări", "HAZARD": "pericol", "POLICE": "poliție", "MISC": "altele"}


def traffic_waze(radius_km=10):
    """Alertele și ambuteiajele live Waze din jurul mașinii (OpenWebNinja, OPENWEBNINJA_API_KEY).
    Fiecare apel consumă o cerere din credit."""
    key = os.environ.get("OPENWEBNINJA_API_KEY")
    if not key:
        return None
    d = http_json(f"https://api.openwebninja.com/waze/alerts-and-jams?center={LAT},{LON}&radius={radius_km}"
                  "&max_alerts=20&max_jams=10", headers={"x-api-key": key}, timeout=20)["data"]
    return summarize_waze(d.get("alerts") or [], d.get("jams") or [])


def summarize_waze(alerts, jams, limit=8):
    if not alerts and not jams:
        return f"Waze: nicio alertă de trafic acum în {CITY} și împrejurimi."
    order = list(WAZE_TYPES)  # ce încetinește traficul primul, poliția și restul la final
    alerts = sorted(alerts, key=lambda a: (order.index(a.get("type")) if a.get("type") in order else len(order),
                                           -(a.get("num_thumbs_up") or 0)))
    lines = []
    for a in alerts[:limit]:
        what = WAZE_TYPES.get(a.get("type"), "alertă")
        sub = (a.get("subtype") or "").removeprefix(f"{a.get('type')}_").replace("_", " ").lower()
        where = ", ".join(x for x in (a.get("street"), a.get("city")) if x)
        lines.append(f"{what}{f' ({sub})' if sub else ''}{f' pe {where}' if where else ''}")
    # shortcut: câmpurile ambuteiajelor nu sunt documentate și n-au apărut încă în răspunsuri; luăm doar strada
    lines += [f"ambuteiaj pe {j['street']}" for j in jams[:limit] if j.get("street")]
    return f"Waze, {len(alerts)} alerte și {len(jams)} ambuteiaje în jurul {CITY}: " + "; ".join(lines)


def search_omniroute(query, limit=5):
    """Căutare web prin OmniRoute cu Tavily (cheia Tavily stă în OmniRoute; ~1000 de căutări gratuite/lună)."""
    r = http_json(f"{OMNIROUTE}/api/v1/search", {"query": query, "provider": "tavily", "max_results": limit},
                  omniroute_headers(), timeout=30)
    hits = []
    for x in r.get("results", []):
        text = re.sub(r"\s+", " ", x.get("snippet") or "")[:500]  # textul relevant al paginii, scurtat
        hits.append(f"{x.get('title', '')}: {text}")
    return ("Căutare web (Tavily): " + " | ".join(hits)) if hits else None


def dated(question):
    """„azi”, „acum”... -> adaugă data zilei la căutare, altfel motoarele dau pagini vechi"""
    if re.search(r"\b(azi|astazi|acum|diseara|maine|weekend)\b", plain(question)):
        return f"{question} {time.strftime('%d.%m.%Y')}"
    return question


def search(query):
    """Căutarea web: doar Tavily prin OmniRoute. None dacă nu răspunde (Virgil spune atunci că nu știe)."""
    return search_omniroute(query)


def safe(source):
    """o sursă căzută întoarce None, nu oprește răspunsul"""
    try:
        return source()
    except Exception as e:
        log.warning("sursa %s: %r", getattr(source, "__name__", "?"), e)
        return None


def gather(question):
    """Datele live pentru întrebare; o sursă căzută nu oprește răspunsul."""
    topics = route(question)
    sources = {
        "weather": weather,
        "news": news,
        # Waze = starea live; știrile adaugă accidentele mari de pe drumurile din jur
        "traffic": lambda: "\n".join(filter(None, (safe(traffic_waze), safe(lambda: search(f"trafic {CITY}"))))),
        "search": lambda: search(dated(question)),
    }
    texts = (safe(sources[name]) for name in ("weather", "news", "traffic", "search") if name in topics)
    return "\n".join(t for t in texts if t)


def speakable(text):
    """fără markdown și trimiteri de tip [1] la voce"""
    return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]|[*#_`]+", "", text).strip() or None


def chat_lines(body):
    """Liniile răspunsului în flux (SSE) de la OmniRoute."""
    request = urllib.request.Request(f"{OMNIROUTE}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **omniroute_headers()})
    with urllib.request.urlopen(request, timeout=40) as r:
        for raw in r:
            yield raw.decode("utf-8", "replace").strip()


def stream_reply(question, sentences=3):
    """Propozițiile răspunsului, pe măsură ce le scrie modelul (cel mult `sentences`): Virgil o citește pe
    prima cât timp modelul o scrie pe următoarea. Tăiem doar după .!? + spațiu, ca 15.00 / 21.11 să rămână întregi."""
    context = gather(question) or "(nicio dată live disponibilă)"
    messages = [{"role": "system", "content": PERSONA.format(city=CITY, now=time.strftime("%d.%m.%Y %H:%M"))},
                {"role": "user", "content": f"Date live:\n{context}\n\nÎntrebarea șoferului: {question}"}]
    for model in MODELS:
        # 1024: unele rute (ex. claude-code) ajung la modele care „gândesc” întâi; cu 250 nu mai apucau să răspundă
        body = {"model": model, "max_tokens": 1024, "messages": messages, "stream": True}
        buffer, said, used = "", 0, None
        try:
            for line in chat_lines(body):
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                    buffer += chunk["choices"][0]["delta"].get("content") or ""
                    if chunk.get("model") not in (None, "keepalive"):  # OmniRoute trimite „keepalive” cât așteaptă
                        used = used or chunk["model"]
                except (ValueError, KeyError, IndexError):
                    continue
                *done, buffer = re.split(r"(?<=[.!?])\s+", buffer)  # ultima bucată poate fi neterminată
                for sentence in filter(None, map(speakable, done)):
                    yield sentence
                    said += 1
                    if said >= sentences:
                        return
        except Exception as e:
            log.warning("OmniRoute (%s): %r", model, e)
            if said:
                return  # am început deja să vorbim: nu repetăm cu alt model
            continue
        tail = speakable(buffer)
        if tail:
            yield tail
            said += 1
        if said:
            log.info("model folosit: %s", used)
            return  # altfel (429, răspuns gol): următorul model


def answer(question):
    """Răspunsul întreg, sau None dacă niciun model nu răspunde."""
    return " ".join(stream_reply(question)) or None


def strip_wake(text):
    """„Hei Virgil, ce vreme e?” -> „ce vreme e”"""
    words = re.findall(r"\w+", text)
    return " ".join(w for w in words if plain(w) not in ("hei", "hey", "ei") and not actions.heard_wake_word(w))


class Safety:
    """Ascultă ML 2: la orice alertă Virgil tace imediat și rămâne tăcut QUIET_SEC."""

    def __init__(self):
        self.interrupt = threading.Event()
        self.quiet_until = 0.0

    def on_decision(self, payload, now=None):
        try:
            sev = json.loads(payload).get("severity")
        except (ValueError, AttributeError):
            return
        if isinstance(sev, int) and sev >= 1:
            self.interrupt.set()
            self.quiet_until = (now if now is not None else time.monotonic()) + QUIET_SEC

    def quiet(self, now=None):
        return (now if now is not None else time.monotonic()) < self.quiet_until


def connect_mqtt(safety):
    try:
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-virgil-info")
        c.on_connect = lambda c, u, f, rc, p: c.subscribe("vigil/decision", qos=1)
        c.on_message = lambda c, u, msg: safety.on_decision(msg.payload)
        c.connect(BROKER, 1883)
        c.loop_start()
        return c
    except Exception as e:
        log.warning("fără MQTT (%r): Virgil nu va ști de alertele ML 2", e)
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name, what in (("OMNIROUTE_API_KEY", "modelul și căutarea Tavily"), ("OPENWEBNINJA_API_KEY", "traficul Waze")):
        if not os.environ.get(name):
            log.warning("lipsește %s (%s): deschideți un terminal nou după ce ați setat-o", name, what)
    safety = Safety()
    connect_mqtt(safety)
    actions.transcribe(np.zeros(actions.MIC_SR, np.float32))  # încarcă Whisper acum, nu la prima întrebare
    log.info("Virgil (informare) pornit, model %s", MODEL)
    shown = None  # ultima stare afișată: o arătăm o singură dată, nu la fiecare fereastră de ascultare
    while True:
        if safety.quiet():
            if shown != "quiet":
                status("Virgil tace: alertă de siguranță ML 2 (reia singur după 60 s)")
                shown = "quiet"
            time.sleep(0.5)
            continue
        if shown != "idle":
            status("GATA. Spuneți „Hei Virgil”, apoi întrebarea (sau totul dintr-o suflare)")
            shown = "idle"
        safety.interrupt.clear()
        result = actions.listen(safety.interrupt)
        if result is None or not result[0]["text"] or not actions.heard_wake_word(result[0]["text"]):
            continue
        shown = None
        question = strip_wake(result[0]["text"])
        if len(question.split()) < 2:  # doar „Hei Virgil”: întrebarea vine separat
            if not actions.speak("Da, vă ascult.", safety.interrupt):
                continue
            actions.play_alarm(1, safety.interrupt)  # bip scurt: acum se pune întrebarea
            status(f"VORBIȚI ACUM: puneți întrebarea (aveți {actions.LISTEN_SEC} s să începeți)")
            result = actions.listen(safety.interrupt)
            question = result[0]["text"] if result and result[0]["text"] else None
            if not question:
                status("Nu am auzit nicio întrebare.")
                continue
        conversation(question, safety.interrupt)


SEARCHING = "Caut imediat."
DONE = re.compile(r"\b(multumesc|mersi|merci|gata|atat|nimic|suficient|ajunge|destul|pa|revedere)\b|^nu\b")


def ends_conversation(text):
    """„Mersi”, „E suficient”, „Bine, mersi!”... închid; o replică lungă („mersi, și mâine?”) e o întrebare."""
    return len(text.split()) <= 4 and bool(DONE.search(plain(text)))


def conversation(question, interrupt):
    """Răspunde, apoi ascultă întrebarea următoare fără „Hei Virgil”, până la tăcere sau „mersi/e suficient”.
    Înainte, după primul răspuns Virgil cerea iar „Hei Virgil”, așa că a doua întrebare era ignorată."""
    while question:
        status(f"Am înțeles: „{question}”. Caut răspunsul...")
        # căutarea + modelul rulează în paralel; cât caută spunem „Caut imediat”, apoi citim fiecare propoziție
        # imediat ce e gata, cât timp modelul o scrie pe următoarea
        ready = queue.Queue()

        def produce(q=question):
            try:
                for sentence in stream_reply(q):
                    ready.put(sentence)
            finally:
                ready.put(None)  # gata (sau eroare)

        threading.Thread(target=produce, daemon=True).start()
        if not actions.speak(SEARCHING, interrupt):
            return  # întrerupt de o alertă ML 2
        spoken = 0
        while (sentence := ready.get()) is not None:
            status(f"Virgil: {sentence}")
            if not actions.speak(sentence, interrupt):
                return  # întrerupt de o alertă ML 2
            spoken += 1
        if not spoken:
            fallback = "Nu am putut afla informația acum, încercați puțin mai târziu."
            status(f"Virgil: {fallback}")
            if not actions.speak(fallback, interrupt):
                return
        actions.play_alarm(1, interrupt)  # bip: poate urma o altă întrebare
        status(f"Mai aveți o întrebare? Vorbiți acum ({actions.LISTEN_SEC} s), sau tăceți ca să încheiați")
        result = actions.listen(interrupt)
        text = result[0]["text"] if result and result[0]["text"] else ""
        question = strip_wake(text) if actions.heard_wake_word(text) else text
        if question and ends_conversation(question):
            question = ""
    status("Conversație încheiată.")


def status(message):
    """ce trebuie să facă șoferul acum, vizibil în terminal"""
    print(f"\n>>> {message}", flush=True)


if __name__ == "__main__":
    main()
