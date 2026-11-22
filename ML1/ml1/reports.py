"""Textul raportului 112 (simulat) și al raportului post-cursă.

Generate din șablon, nu de LLM: deterministe, instantanee, fără halucinații —
un dispecer 112 trebuie să primească doar fapte observate.
"""

from __future__ import annotations

import time
from typing import Any

# Locație simulată (în mașină ar veni de la GNSS).
SIM_LOCATION = {
    "road": "A1 București–Pitești, km 42, sens Pitești",
    "lat": 44.4268,
    "lon": 25.8890,
}

EVENT_LABELS = {
    "hand_on_chest": "mână pe piept",
    "heavy_breathing": "respirație grea",
    "head_drop": "cap căzut",
    "head_on_wheel": "prăbușit pe volan, nemișcat",
    "slumped_side": "căzut într-o parte, nemișcat",
    "bad_posture": "postură aplecată prelungit",
    "out_of_position": "ieșit din poziția de condus",
    "phone_use": "telefon în mână",
    "eyes_closed": "ochi închiși",
    "microsleep": "micro-somn",
    "yawn": "căscat",
    "phone_in_hand": "telefon în mână",
    "gaze_off_road": "privire în afara drumului",
}

MAX_112_LEN = 1000  # limita din tools.TOOLS["call_112"]


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _signs(ctx: dict) -> dict[str, dict]:
    """Semnele observate: timeline-ul episodului (de la engine) + evenimentele curente."""
    signs: dict[str, dict] = {}
    for name, e in (ctx.get("timeline") or {}).items():
        signs[name] = {"dur": _f(e.get("duration_sec")), "conf": _f(e.get("confidence")), "breathing": e.get("breathing")}
    evs = (ctx.get("vision") or {}).get("events")
    for e in evs if isinstance(evs, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get("event"), str):
            continue
        cur = signs.setdefault(e["event"], {"dur": 0.0, "conf": 0.0, "breathing": None})
        cur["dur"] = max(cur["dur"], _f(e.get("duration_sec")))
        cur["conf"] = max(cur["conf"], _f(e.get("confidence")))
        cur["breathing"] = cur["breathing"] or e.get("breathing")
    vision = ctx.get("vision") or {}
    if vision.get("head_drop") is True:
        signs.setdefault("head_drop", {"dur": 0.0, "conf": 0.0, "breathing": None})
    return signs


def _suspicion(signs: dict) -> str:
    # respirația grea confirmă; cu analiza respirației oprită în Vision ajunge mâna ținută mult pe piept (rules)
    chest = "hand_on_chest" in signs and ("heavy_breathing" in signs or signs["hand_on_chest"].get("breathing") == "heavy"
                                          or signs["hand_on_chest"]["dur"] >= 6.0)
    unconscious = ("head_on_wheel" in signs or "slumped_side" in signs
                   or ("head_drop" in signs and ("eyes_closed" in signs or "microsleep" in signs)))
    if chest and unconscious:
        return "posibilă urgență cardiacă urmată de pierderea cunoștinței"
    if unconscious:
        return "posibilă pierdere a cunoștinței"
    if chest:
        return "posibilă urgență cardiacă"
    return "șofer care nu reacționează"


def _answer(ctx: dict) -> str:
    r = ctx.get("driver_response")
    if not r:
        return "nu a fost întrebat încă"
    if r.get("answered") is False:
        return "niciun răspuns"
    lat = r.get("latency_sec")
    lat_s = f" după {_f(lat):.1f} s" if lat is not None else ""
    return f"„{r.get('text') or '?'}”{lat_s}"


def build_112_report(ctx: dict, location: dict | None = None, now: float | None = None) -> str:
    loc = location or SIM_LOCATION
    signs = _signs(ctx)
    order = ["hand_on_chest", "heavy_breathing", "head_drop", "head_on_wheel", "slumped_side", "eyes_closed", "microsleep"]
    names = sorted(signs, key=lambda n: order.index(n) if n in order else 99)
    lines = []
    for n in names:
        s = signs[n]
        label = EVENT_LABELS.get(n, n)
        if n == "hand_on_chest" and s.get("breathing") == "heavy" and "heavy_breathing" not in signs:
            label += " + respirație grea"
        dur = f" {s['dur']:.0f} s" if s["dur"] >= 1 else ""
        conf = f" (încredere {s['conf'] * 100:.0f}%)" if s["conf"] else ""
        lines.append(f"- {label}{dur}{conf}")
    a = _f((ctx.get("fusion") or {}).get("a"), 0.0)
    ts = time.strftime("%H:%M:%S", time.localtime(now if now is not None else time.time()))

    text = "\n".join([
        "APEL DE URGENȚĂ AUTOMAT – Vigil (SIMULAT)",
        f"Ora: {ts}",
        f"Locație: {loc['road']} ({loc['lat']:.4f}, {loc['lon']:.4f})",
        f"Suspiciune: {_suspicion(signs)} (semne observate de cameră, nu diagnostic)",
        "Semne observate:",
        *(lines or ["- șoferul nu reacționează"]),
        f"Răspuns la „Sunteți bine?”: {_answer(ctx)}",
        f"Scor atenție: {a:.2f} / 1",
        _vehicle(ctx.get("car")),
    ])
    return text[:MAX_112_LEN]


def _vehicle(car: dict | None) -> str:
    """Starea reală raportată de ECU (vigil/car/state); fără ECU, textul de dinainte."""
    if not car or not isinstance(car.get("speed_kmh"), (int, float)):
        return "Vehicul: avarii pornite, manevră de oprire în siguranță."
    motion = "oprit" if car.get("stopped") else "frânează până la oprire" if car.get("braking") else "în mers"
    return f"Vehicul (ECU): {car['speed_kmh']:.0f} km/h, {motion}, avarii {'pornite' if car.get('hazard') else 'oprite'}."


# ------------------------------------------------------------- post-cursă ----

def build_trip_report(log: dict) -> str:
    """`log` = DecisionEngine.trip_log(): a_samples [(ts, a)], decisions [...], events {name: total_sec}."""
    samples = log.get("a_samples") or []
    decisions = log.get("decisions") or []
    events = log.get("events") or {}
    if samples:
        t0, t1 = samples[0][0], samples[-1][0]
        a_vals = [a for _, a in samples]
        avg, low = sum(a_vals) / len(a_vals), min(a_vals)
        low_ts = next(t for t, a in samples if a == low) - t0
    else:
        t0 = t1 = avg = low = low_ts = 0.0

    by_sev = {s: sum(1 for d in decisions if d.get("severity") == s) for s in (1, 2, 3)}
    n112 = sum(1 for d in decisions for a in d.get("actions", []) if a.get("tool") == "call_112")
    llm = sum(1 for d in decisions if d.get("source") == "llm")

    lines = [
        "RAPORT POST-CURSĂ – Vigil",
        f"Durată monitorizată: {(t1 - t0) / 60:.1f} min",
        f"Atenție medie: {avg:.2f}  |  minimă: {low:.2f} (la {low_ts:.0f} s)",
        f"Alerte: {by_sev[1]} ușoare, {by_sev[2]} medii, {by_sev[3]} ridicate; apeluri 112: {n112}",
        f"Decizii: {llm} de la LLM, {len(decisions) - llm} din reguli fixe",
    ]
    if events:
        lines.append("Evenimente (durată totală):")
        for n, sec in sorted(events.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {EVENT_LABELS.get(n, n)}: {sec:.0f} s")
    recs = []
    if by_sev[2] or any(n in events for n in ("microsleep", "yawn")):
        recs.append("semne de oboseală: faceți o pauză la fiecare 2 ore de condus")
    if "phone_in_hand" in events:
        recs.append("folosiți telefonul doar prin sistemul mașinii")
    if by_sev[3]:
        recs.append("eveniment medical posibil: consultați un medic")
    lines.append("Recomandări: " + ("; ".join(recs) if recs else "nicio problemă, drum bun!"))
    return "\n".join(lines)
