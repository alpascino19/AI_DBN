"""Virgil — asistentul vocal de informare, separat de executorul ML 2 (siguranța).

„Hei Virgil, <întrebare>” -> date live (vreme: Open-Meteo; trafic: Waze; durata până la o destinație:
TomTom; știri: RSS; orice altceva: căutare
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
import subprocess
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

import actions
from executor import BROKER

OMNIROUTE = "http://localhost:20128"
# Doar Groq (~2-3 s pe răspuns): Gemini gratuit dă des 429 și atunci fiecare întrebare aștepta eroarea.
# qwen3.8-27b scrie cea mai curată română de citit cu voce tare; gpt-oss rămân de rezervă (pun „°C”, „%”).
MODEL = os.environ.get("VIRGIL_MODEL", "groq/qwen/qwen3.8-27b")
# dacă primul model e indisponibil (429, cădere), încercăm următorul
MODELS = [MODEL] + [m for m in ("groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b") if m != MODEL]
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
La întrebări de cunoștințe generale (mașini și întreținere, presiunea în roți, zgomote, sfaturi, explicații) răspunzi \
din ce știi, concret și util, chiar dacă datele live nu le conțin; la un zgomot sau o defecțiune spui cauzele \
probabile și recomanzi un service, iar dacă pare periculos, să oprească în siguranță. \
Nu inventa fapte de actualitate. Locația mașinii: {city}. Data și ora: {now}."""

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
    """Ce surse live cer întrebarea: subset din {weather, news, traffic, eta, where, search}, sau {chat}."""
    q = plain(question)
    # conversație cu Virgil, nu o întrebare despre lume: fără căutare (pe web nu există „ce face Virgil”)
    if re.search(r"^\W*(ce (mai )?fa[cts]i|cum (te )?(simti|esti)|cine esti|cum te (cheama|numesti)|ce (poti|stii) (sa )?"
                 r"(faci|face)|cu ce (ma )?poti ajuta|salut|buna( ziua| dimineata| seara)?|neata|multumesc|mersi"
                 r"|esti acolo|ma auzi)\b", q):
        return {"chat"}
    topics = set()
    if re.search(r"vreme|ploua|ploaie|ninge|temperatur|grade|frig|cald|soare|umbrel", q):
        topics.add("weather")
    if re.search(r"stir|noutat|ce se (mai )?intampla|evenimente", q):
        topics.add("news")
    if re.search(r"trafic|aglomer|accident|coloan|coada|drum|blocat|ambuteiaj|lucrari", q):
        topics.add("traffic")
    if re.search(r"cat fac|cat (mai )?dureaza|cat mai am|in cat timp|cat de departe|cate minute|cati (km|kilometri)"
                 r"|cat (imi )?ia|distanta|departe|pana la|pina la|pana in|ajun[gt]|merg la|drumul la", q) \
            and destination(question):
        topics.add("eta")
    if re.search(r"unde (sunt|ma aflu|ne aflam|suntem)|locati|pozitia|adresa (mea|curenta)|coordonat", q):
        topics.add("where")
    return topics or {"search"}


PAIR = re.compile(r"\b(?:dintre|între|intre)\s+(.+?)\s+(?:și|si)\s+([^,?!]+?)\s*(?:[,?!]|\.?\s*$)", re.I)


def destination(question):
    """„Cât fac până la Gara Cluj?” -> „Gara Cluj” (textul după până la / spre / către / ajung la / dintre X și)."""
    pair = PAIR.search(question)
    if pair:
        return pair.group(2).strip()
    # Whisper scrie uneori „pînă” (ortografia veche) sau „ajunt”; „ajung la” / „merg la” = aceeași întrebare
    m = re.search(r"\b(?:p[âîai]n[ăa]\s+(?:la|în|in)|spre|c[ăa]tre|ajun[gt](?:e|em|i)?\s+la|merg(?:em)?\s+la|drumul\s+la)\s+"
                  r"([^,?!]+?)\s*(?:[,?!]|\.?\s*$)", question, re.I)
    return m.group(1).strip() if m else None


def origin(question):
    """„... de la strada Observatorului până la gară” -> „strada Observatorului”; None = plecăm din centru.
    Whisper strică uneori „de la” („de ina strada ...”): atunci recunoaștem plecarea după cuvântul strada/calea/..."""
    pair = PAIR.search(question)
    if pair:
        return pair.group(1).strip()
    m = re.search(r"\bde\s+(?:la\s+(.+?)|\S+\s+((?:strada|str\.|bulevardul|calea|pia[țt]a|aleea)\s.+?))"
                  r"\s+(?:p[âîai]n[ăa]|spre|c[ăa]tre)\b", question, re.I)
    return (m.group(1) or m.group(2)).strip() if m else None


# „vremea în România”: câte un oraș mare din fiecare regiune, într-o singură cerere
ROMANIA = [("București", 44.43, 26.10), ("Cluj-Napoca", 46.77, 23.59), ("Iași", 47.16, 27.59),
           ("Timișoara", 45.75, 21.23), ("Constanța", 44.18, 28.63), ("Brașov", 45.65, 25.61),
           ("Craiova", 44.32, 23.80), ("Suceava", 47.65, 26.26)]
# regiunile spuse în vorbire -> orașul care le reprezintă
REGIONS = {"moldova": "Iași", "ardeal": "Cluj-Napoca", "transilvania": "Cluj-Napoca", "banat": "Timișoara",
           "dobrogea": "Constanța", "litoral": "Constanța", "mare": "Constanța", "oltenia": "Craiova",
           "muntenia": "București", "capitala": "București", "maramures": "Baia Mare", "bucovina": "Suceava",
           "crisana": "Oradea", "munte": "Sinaia", "prahova": "Sinaia"}
FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,weather_code,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=Europe%2FBucharest&forecast_days=1")


def weather_place(question):
    """Locul din întrebarea despre vreme: „România” (tot), o regiune, un oraș sau None (Cluj, ca până acum)."""
    q = plain(question)
    if re.search(r"\b(romania|tara|toata tara)\b", q):
        return "România"
    for word, city in REGIONS.items():
        if re.search(rf"\b(in|la|din|pe|spre)\s+(zona\s+|regiunea\s+)?{word}\b", q):
            return city
    # „la Brașov”, „din alba Iulia”: textul după prepoziție, fără majuscule obligatorii (Whisper nu le pune mereu),
    # până la punctuație sau un cuvânt de timp („azi”, „acum”...)
    m = re.search(r"\b(?:în|in|la|din|pentru|spre)\s+(.+?)\s*(?:\b(?:azi|astăzi|astazi|acum|mâine|maine|diseară|"
                  r"diseara|dimineață|dimineata|seara|weekend)\b|[,?!.]|$)", question, re.I)
    place = m.group(1).strip() if m else ""
    # „la noi”, „afară” nu sunt localități (geocodarea ar găsi „Noiștat”)
    if not place or len(place.split()) > 3 or plain(place) in ("noi", "voi", "mine", "afara", "oras", "zona", "exterior"):
        return None
    return place


def geocode(name):
    """Localitate -> (nume, lat, lon), întâi în România, apoi oriunde (Open-Meteo, fără cheie).
    „Scărișoara județul Alba”: județul alege dintre localitățile cu același nume."""
    county = None
    m = re.search(r"[\s,]+(?:jude[tț]ul|jud\.?)\s+(.+)$", name, re.I)
    if m:
        name, county = name[:m.start()], plain(m.group(1))
    for country in ("&countryCode=RO", ""):
        r = http_json(f"https://geocoding-api.open-meteo.com/v1/search?count={10 if county else 1}&language=ro"
                      f"&name={urllib.parse.quote(name)}{country}").get("results")
        if r:
            x = next((x for x in r if county and county in plain(x.get("admin1", ""))), r[0])
            return x["name"], x["latitude"], x["longitude"]
    return None


def describe(name, d):
    c, day = d["current"], d["daily"]
    return (f"Vremea în {name} acum: {c['temperature_2m']} grade, {WMO.get(c['weather_code'], 'variabil')}, "
            f"vânt {c['wind_speed_10m']} km/h. Azi: maxima {day['temperature_2m_max'][0]} grade, "
            f"minima {day['temperature_2m_min'][0]} grade, șanse de precipitații "
            f"{day['precipitation_probability_max'][0]}%.")


def weather(question=""):
    place = weather_place(question)
    if place == "România":
        d = http_json(FORECAST.format(lat=",".join(str(c[1]) for c in ROMANIA),
                                      lon=",".join(str(c[2]) for c in ROMANIA)))
        return "Vremea în România, pe regiuni: " + " ".join(describe(c[0], x) for c, x in zip(ROMANIA, d))
    found = geocode(place) if place else None
    name, lat, lon = found or (CITY, LAT, LON)
    text = describe(name, http_json(FORECAST.format(lat=lat, lon=lon)))
    return text if found or not place else f"Nu am găsit localitatea „{place}”. {text}"


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


# Serviciul de localizare Windows (după rețelele Wi-Fi, ~100 m), fără cheie; InvariantCulture = punct zecimal
HERE_PS = ("Add-Type -AssemblyName System.Device; $w = New-Object System.Device.Location.GeoCoordinateWatcher(1); "
           "$w.Start(); $t = 0; while ($w.Status -ne 'Ready' -and $w.Permission -ne 'Denied' -and $t -lt 100) "
           "{ Start-Sleep -Milliseconds 100; $t++ }; $c = $w.Position.Location; if (-not $c.IsUnknown) "
           "{ [string]::Format([cultureinfo]::InvariantCulture, '{0};{1}', $c.Latitude, $c.Longitude) }")
_here = (0.0, None)


def here_windows():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", HERE_PS], capture_output=True, text=True,
                         timeout=15).stdout.strip()
    return tuple(map(float, out.split(";")))


def here(max_age=60):
    """(lat, lon) curente de la Windows, păstrate `max_age` s; None = localizarea e oprită, plecăm din centru."""
    global _here
    if time.time() - _here[0] < max_age:
        return _here[1]
    try:
        pos = here_windows()
    except Exception as e:
        log.warning("localizarea Windows: %r (plecăm din centrul orașului)", e)
        pos = None
    if pos:   # un eșec nu se memorează: următoarea întrebare reîncearcă
        _here = (time.time(), pos)
    return pos


TOMTOM = "https://api.tomtom.com"


def tomtom_place(query, key, near):
    """Text („gară”, „Iulius Mall”, o stradă scrisă greșit de Whisper) -> (lat, lon, nume), căutat lângă `near`."""
    lat, lon = near
    # „gară” simplu = gara mare, nu cea mai apropiată haltă (TomTom ar alege „Clujana”)
    # shortcut: doar pentru Cluj-Napoca; de generalizat dacă se schimbă CITY
    if re.fullmatch(r"gara( mare| centrala| cluj(-napoca)?| din cluj(-napoca)?)?", plain(query)):
        query = "Stația CFR Cluj-Napoca"
    r = http_json(f"{TOMTOM}/search/2/search/{urllib.parse.quote(query)}.json?key={key}&lat={lat}&lon={lon}"
                  "&countrySet=RO&language=ro-RO&limit=1")["results"]
    if not r:
        return None
    x = r[0]
    return x["position"]["lat"], x["position"]["lon"], x.get("poi", {}).get("name") or query


def eta_tomtom(question):
    """Durata cu traficul de acum până la destinația din întrebare (TomTom Search + Routing, TOMTOM_API_KEY).
    Plecarea: „de la ...” din întrebare, altfel poziția curentă (Windows), altfel centrul orașului."""
    key, dest, start = os.environ.get("TOMTOM_API_KEY"), destination(question), origin(question)
    if not key or not dest:
        return None
    pos = None if start else here()
    near = pos or (LAT, LON)
    a = tomtom_place(start, key, near) if start else (*near, None)
    b = tomtom_place(dest, key, near)
    if not a or not b:
        return f"TomTom: nu am găsit {start if not a else dest}."
    d = http_json(f"{TOMTOM}/routing/1/calculateRoute/{a[0]},{a[1]}:{b[0]},{b[1]}/json?key={key}"
                  "&traffic=true&travelMode=car")
    return summarize_route(b[2], d, start or ("locul în care sunteți acum" if pos else None))


def where_tomtom():
    """„Unde sunt?” -> adresa poziției curente (Windows + TomTom reverse geocoding)."""
    pos = here()
    if not pos:
        return "Localizarea Windows e oprită: nu știu unde se află mașina acum."
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        return f"Poziția aproximativă a mașinii: latitudine {pos[0]:.4f}, longitudine {pos[1]:.4f}."
    a = http_json(f"{TOMTOM}/search/2/reverseGeocode/{pos[0]},{pos[1]}.json?key={key}&language=ro-RO")
    address = a["addresses"][0]["address"].get("freeformAddress") if a.get("addresses") else None
    return (f"TomTom: mașina e aproximativ la adresa {address} (poziția are o precizie de circa 100 de metri)."
            if address else "TomTom: nu am găsit adresa poziției curente.")


def summarize_route(dest, d, start=None):
    routes = d.get("routes") or []
    if not routes:
        return f"TomTom: nu am găsit un traseu până la {dest}."
    r = routes[0]["summary"]
    now, late = round(r.get("travelTimeInSeconds", 0) / 60), round(r.get("trafficDelayInSeconds", 0) / 60)
    delay = f", cu {late} minute în plus din cauza traficului" if late >= 2 else ", fără întârzieri de trafic"
    frm = f"de la {start}" if start else f"din centrul orașului {CITY}"
    km = r.get("lengthInMeters", 0) / 1000
    km = round(km) if km >= 20 else round(km, 1)   # „291.1” îl încurca pe model („douăzeci și nouă”)
    when = f"{now // 60} ore și {now % 60} minute" if now >= 60 else f"{now} minute"
    return f"TomTom: {frm} până la {dest} sunt {km} kilometri, {when} cu traficul de acum{delay}."


def search_omniroute(query, limit=5, recent=False):
    """Căutare web prin OmniRoute cu Tavily (cheia Tavily stă în OmniRoute; ~1000 de căutări gratuite/lună).
    recent=True: doar știrile din ultima zi (topic news). Pentru „meciurile de ieri” aduce articolele cu scorul,
    nu paginile de clasament fără rezultate."""
    body = {"query": query, "provider": "tavily", "max_results": limit}
    if recent:
        body.update(topic="news", time_range="day")
    r = http_json(f"{OMNIROUTE}/api/v1/search", body, omniroute_headers(), timeout=30)
    hits = [f"{x.get('title', '')}: {clean_snippet(x.get('snippet') or '')}" for x in r.get("results", [])]
    hits = [h for h in hits if len(h) > 40]
    # rezultatele cu scoruri („Dinamo - Petrolul 0-1”) primele: altfel se pierdeau după paginile de lot și clasament
    hits.sort(key=lambda h: -len(SCORE.findall(h)))
    label = "Știri din ultimele 24 de ore (Tavily)" if recent else "Căutare web (Tavily)"
    return (f"{label}: " + " || ".join(hits)) if hits else None


SCORE = re.compile(r"\b\d{1,2}-\d{1,2}\b")


def clean_snippet(text):
    """Fără zgomotul de tabel din pagini („| 16 | 5 | 4 | ---”), care îngropa scorurile; scurtat la 1200 caractere
    (la 500 se tăiau scorurile din tabelul LPF)."""
    text = re.sub(r"(?:\|\s*(?:\\?-+|\d{1,3})?\s*)+(?=\||$)", " ", text)   # celule goale sau doar cu numere
    text = re.sub(r"\s*(?:#+|---+|\[\.\.\.\])\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:1200]


def dated(question):
    """„azi”, „acum”... -> adaugă data zilei la căutare, altfel motoarele dau pagini vechi"""
    if re.search(r"\b(azi|astazi|acum|diseara|maine|weekend)\b", plain(question)):
        return f"{question} {time.strftime('%d.%m.%Y')}"
    return question


QUERY_PROMPT = """Transformă întrebarea șoferului într-o căutare web scurtă în română, 3-7 cuvinte, fără politețuri.
Dacă întrebarea se referă la o zi, pune data exactă (zz.ll.aaaa). Azi este {today}, ieri a fost {yesterday}.
„Azi noapte”, „aseară” = ieri. „În România” la sport = Superliga. Răspunde DOAR cu textul căutării."""


def search_query(question):
    """„Spune-mi te rog rezultatele meciurilor din România de azi noapte” -> „rezultate Superliga 21.11.2026”.
    Propoziția vorbită întreagă dădea timeout (504) la Tavily; o căutare scurtă răspunde în ~3 s."""
    day = 86400
    body = {"model": MODEL, "max_tokens": 40, "stream": False, "messages": [
        {"role": "system", "content": QUERY_PROMPT.format(today=time.strftime("%d.%m.%Y"),
                                                          yesterday=time.strftime("%d.%m.%Y", time.localtime(time.time() - day)))},
        {"role": "user", "content": question}]}
    try:
        text = http_json(f"{OMNIROUTE}/v1/chat/completions", body, omniroute_headers(), timeout=8)
        query = (text["choices"][0]["message"].get("content") or "").strip().strip('"„”').splitlines()[0]
        return query if 2 <= len(query.split()) <= 12 else dated(question)
    except Exception as e:
        log.warning("reformularea căutării: %r", e)
        return dated(question)


NO_SEARCH = ("Căutarea web nu a întors nimic acum. Dacă întrebarea e despre actualitate (evenimente, rezultate, "
             "persoane în funcție, prețuri): spune pe scurt că nu poți afla informația acum, NU ghici. Dacă e o "
             "întrebare de cunoștințe generale, răspunde din ce știi.")


def search(query):
    """Căutarea web: doar Tavily prin OmniRoute. None dacă nu răspunde (Virgil spune atunci că nu știe)."""
    # o dată anume, „azi/ieri”, scoruri, meciuri: întâi știrile din ultima zi, apoi căutarea obișnuită
    recent = bool(re.search(r"\d{2}\.\d{2}\.\d{4}|\b(azi|astazi|ieri|aseara|acum|diseara|rezultat|scor|meci)", plain(query)))
    log.info("caut%s: %s", " (știri recente)" if recent else "", query)
    return (recent and search_omniroute(query, recent=True)) or search_omniroute(query)


def safe(source):
    """o sursă căzută întoarce None, nu oprește răspunsul"""
    try:
        return source()
    except Exception as e:
        log.warning("sursa %s: %r", getattr(source, "__name__", "?"), e)
        return None


ABOUT = ("Despre tine, Virgil (folosește doar dacă șoferul te salută sau te întreabă de tine): ești bine și atent "
         "la drum. Ce poți face, la cerere: spui vremea, traficul din jur, cât face șoferul până la o destinație, "
         "unde se află mașina acum, știrile zilei și orice informație căutată pe internet. Siguranța are prioritate: "
         "taci imediat ce apare o alertă. Răspunde cald, într-o propoziție sau două.")


def gather(question):
    """Datele live pentru întrebare; o sursă căzută nu oprește răspunsul."""
    topics = route(question)
    sources = {
        "weather": lambda: weather(question),
        "news": news,
        # Waze = starea live; știrile adaugă accidentele mari de pe drumurile din jur
        "traffic": lambda: "\n".join(filter(None, (safe(traffic_waze), safe(lambda: search(f"trafic {CITY}"))))),
        "eta": lambda: eta_tomtom(question),
        "where": where_tomtom,
        "chat": lambda: ABOUT,
        "search": lambda: safe(lambda: search(search_query(question))) or NO_SEARCH,
    }
    texts = (safe(sources[name]) for name in ("chat", "weather", "news", "traffic", "eta", "where", "search") if name in topics)
    return "\n".join(t for t in texts if t)


def speakable(text):
    """fără markdown, trimiteri de tip [1] și cuvinte în alt alfabet (qwen strecoară rar arabă/chineză) la voce"""
    text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]|[*#_`]+", "", text)
    text = re.sub(r"\S*[^\s\x00-\u024F\u2000-\u206F]\S*", "", text)   # păstrăm latina (cu ș, ț) și „ ” – …
    return re.sub(r"\s{2,}", " ", text).strip() or None


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
    """„Hei Virgil, ce vreme e?” -> „ce vreme e”; tot ce e înainte de nume cade („Ca-i Virgil”, cum aude Whisper uneori)"""
    words = re.findall(r"\w+", text)
    wake = next((i for i, w in enumerate(words) if actions.heard_wake_word(w)), None)
    if wake is not None:
        words = words[wake + 1:]
    return " ".join(w for w in words if plain(w) not in ("hei", "hey", "ei"))


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
    for name, what in (("OMNIROUTE_API_KEY", "modelul și căutarea Tavily"), ("OPENWEBNINJA_API_KEY", "traficul Waze"),
                      ("TOMTOM_API_KEY", "durata până la o destinație")):
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
