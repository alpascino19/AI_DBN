"""Reguli fixe: (1) plan B când LLM-ul e lent / invalid / lipsește și
(2) calea de siguranță — podeaua pe care LLM-ul nu o poate coborî.

În arhitectura auto, ACESTA e modulul relevant pentru siguranță (determinist,
testabil); LLM-ul e un consilier fără rating de siguranță (QM), care doar nuanțează.

Context (`ctx`), același pe care îl primește și LLM-ul:
    {
      "fusion": FusionResult.to_dict(),     # a, confidence, critical, ...
      "vision": <mesaj vigil/vision>,       # perclos, events, ...
      "driver_response": <vigil/driver_response> | None,   # DOAR la întrebarea curentă
      "timeline": {event: {...}},           # opțional, de la engine (pentru raport)
    }
"""

from __future__ import annotations

import re
from typing import Any

from .actions import PHRASES
from .fusion import eye_artifact
from .profile import P


def _events(vision: dict) -> dict[str, dict]:
    """Cel mai sigur eveniment de fiecare tip."""
    out: dict[str, dict] = {}
    evs = vision.get("events")
    if not isinstance(evs, list):
        return out
    for e in evs:
        if isinstance(e, dict) and isinstance(e.get("event"), str):
            name = e["event"]
            if _f(e.get("confidence"), 0) >= _f(out.get(name, {}).get("confidence"), -1):
                out[name] = e
    return out


def _f(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _has(evs: dict, name: str, conf: float = 0.6, dur: float = 0.0) -> bool:
    e = evs.get(name)
    return bool(e) and _f(e.get("confidence"), 0) >= conf and _f(e.get("duration_sec"), 0) >= dur


def _num(d: dict, key: str, default: float) -> float:
    return _f(d.get(key, default), default)


LOW_A_PERSIST_SEC = 4.0  # ales din evaluare (ml1/evaluate.py): elimină alarmele false din pâlpâiri

_DISTRESS = re.compile(r"nu mă simt|nu ma simt|rău|rau\b|durere|doare|ajutor|nu pot|ambulan", re.I)
_OK = re.compile(r"\b(da|bine|ok|okay|sunt bine|totul e bine)\b", re.I)


def answer_status(resp: dict | None) -> str:
    """none | no_answer | distress | slow | ok"""
    if not resp:
        return "none"
    if resp.get("answered") is False:
        return "no_answer"
    text = resp.get("text") or ""
    if _DISTRESS.search(text):
        return "distress"
    if _f(resp.get("latency_sec"), 99) > 3.0 or not _OK.search(text):
        return "slow"
    return "ok"


def rule_severity(ctx: dict) -> tuple[int, str]:
    fusion = ctx.get("fusion") or {}
    vision = ctx.get("vision") or {}
    evs = _events(vision)
    a = _num(fusion, "a", 1.0)
    ans = answer_status(ctx.get("driver_response"))

    # PERCLOS corectat cu valoarea personală din calibrare, dacă există (engine îl pune în ctx).
    # Semantica CV: fracția de timp cu ochii aproape complet închiși, pe ultimele 60 s.
    perclos = _num(vision, "perclos_cal", _num(vision, "perclos", 0))
    if P["perclos_recent"] and _num(vision, "perclos_recent", -1.0) >= 0:
        perclos = _num(vision, "perclos_recent", 0)   # închiderea pe 15 s: fără întârzierea ferestrei de 60 s
    heavy = (vision.get("breathing_state") == "heavy" or _has(evs, "heavy_breathing", 0.6)
             or (evs.get("hand_on_chest") or {}).get("breathing") == "heavy")
    chest = _has(evs, "hand_on_chest", 0.6, 3) and heavy

    # Pleoape coborâte la căscat / privire în jos / telefon NU înseamnă micro-somn (regulă preluată
    # din alerts.py al echipei CV). Profilul `legacy` nu face excluderea (doar pentru comparație).
    artifact = (eye_artifact(vision.get("events"), _num(vision, "phone", 0), vision.get("direction"))
                or fusion.get("eyes_closed_artifact") is True)
    closed_ev = evs.get("eyes_closed") if not artifact else None
    closed_sec = _f(closed_ev.get("duration_sec"), 0) if closed_ev and _f(closed_ev.get("confidence"), 0) >= 0.7 else 0.0
    # Tracker-ul CV unește activările separate de pauze < 1 s: durata lui `eyes_closed` e a EPISODULUI, nu a
    # închiderii continue (clipiri lungi și dese ar părea „ochi închiși 3 s”). Dacă CV publică `closed_sec`
    # (închiderea continuă, calculată intern de CV1), o folosim. Altfel ignorăm durata doar când EAR-ul
    # arată CLAR ochi deschiși acum; la EAR ambiguu credem CV-ul (nu ascundem un micro-somn real).
    cs = _num(vision, "closed_sec", -1.0)
    if cs >= 0 and not artifact:
        closed_sec = cs
    elif P["artifact_exclusion"] and fusion.get("eyes_open_now") is True:
        closed_sec = 0.0
    eyes_closed_long = closed_sec >= P["eyes_sev2_sec"]
    eyes_closed_crit = P["eyes_sev3_sec"] is not None and closed_sec >= P["eyes_sev3_sec"]
    if eyes_closed_crit and P["eyes_crit_continuous"] and fusion.get("ear_available") is True and cs < 0:
        # criticul (3) cere închidere CONTINUĂ >= 3 s și în EAR; altfel rămâne micro-somn (2), tot cu alarmă
        eyes_closed_crit = _num(fusion, "eyes_closed_sec", 0) >= P["eyes_sev3_sec"]

    # Prăbușire: NU cerem PERCLOS mare (fereastra de 60 s urcă încet: după 10 s cu ochii închiși
    # PERCLOS e abia ~0.2). Cap căzut + ochi închiși, sau corpul prăbușit (CV2), ajung.
    if P["perclos_recent"] or P["artifact_exclusion"]:
        head_and_eyes = vision.get("head_drop") is True and (
            _has(evs, "eyes_closed", 0.6) or _has(evs, "microsleep", 0.6) or perclos >= 0.6)
    else:
        head_and_eyes = vision.get("head_drop") is True and perclos >= 0.6
    collapse = (head_and_eyes
                or _has(evs, "head_on_wheel", 0.6, 2)                       # CV2: prăbușit pe volan
                or (vision.get("posture_state") == "slumped" and _num(vision, "still_sec", 0) >= 2))

    # Revenire rapidă: PERCLOS pe 60 s rămâne mare până la un minut după ce șoferul și-a revenit.
    # Dacă ultimele 15 s arată normal (câmpul CV `recovered` / `perclos_recent`, sau estimarea ML1
    # din engine), nu mai tratăm PERCLOS-ul vechi drept oboseală actuală.
    recovered = False
    if P["perclos_recovered"] is not None:
        if vision.get("recovered") is True:
            recovered = True
        else:
            pr = _num(vision, "perclos_recent", -1.0)
            recovered = 0.0 <= pr <= P["perclos_recovered"]
    perclos_drowsy = perclos >= P["perclos_drowsy"] and not recovered

    if fusion.get("critical") or collapse:
        return 3, "cap căzut / prăbușit, ochi închiși"
    if eyes_closed_crit:
        return 3, "ochi închiși de peste 3 s (adormit la volan)"
    if eyes_closed_long and ans in ("no_answer", "distress"):
        return 3, "ochi închiși prelungit + fără răspuns"
    if chest:
        if ans == "ok":  # a răspuns repede și coerent: rămânem vigilenți, dar nu urgență
            return 2, "mână pe piept, dar șoferul răspunde coerent"
        return 3, "mână pe piept + respirație grea"
    microsleep = _has(evs, "microsleep", 0.6) and not artifact
    fatigue_signs = eyes_closed_long or microsleep or perclos_drowsy
    # Fără răspuns la verificarea vocală, cu semnele încă prezente (nu doar „a mic”): urgență.
    if ans in ("no_answer", "distress") and (a < 0.5 or (P["silence_needs_signs"] and fatigue_signs)):
        return 3, "fără răspuns coerent + semne de oboseală / atenție scăzută"

    # Ochi închiși ≥ 1.5 s fără cap căzut = micro-somn: alarmă puternică (2), nu urgență medicală (3).
    if eyes_closed_long:
        return 2, "ochi închiși prelungit (micro-somn)"
    if microsleep or perclos_drowsy or a < 0.5:
        return 2, "semne de somnolență (PERCLOS / micro-somn)"
    if _has(evs, "frequent_yawning", 0.5) or _has(evs, "low_blink_rate", 0.5, 5) and perclos >= P["perclos_yawn"]:
        return 2, "căscat des / clipit rar: oboseală"
    if _has(evs, "yawn", 0.6) and perclos >= P["perclos_yawn"] and not recovered:
        return 2, "căscat + ochi obosiți"

    if _num(vision, "phone", 0) >= 0.6 or _has(evs, "phone_in_hand", 0.6, 2):
        return 1, "telefon în mână"
    if vision.get("gaze_on_road") is False and _has(evs, "gaze_off_road", 0.6, 2):
        return 1, "privire în afara drumului"
    if _num(vision, "off_road_ratio", 0) >= 0.5:   # dacă CV publică HeadState.off_road_ratio (10 s)
        return 1, "privire des în afara drumului"
    if _has(evs, "hand_on_chest", 0.5) or _has(evs, "heavy_breathing", 0.5):
        # semn medical încă scurt: nu-i spunem „priviți drumul”, așteptăm pragul de 3 s
        return 0, "observăm un posibil semn medical"
    # Regula generică (fără o cauză concretă) e cea mai fragilă la shift de domeniu: cerem
    # persistență (a < 0.75 de cel puțin LOW_A_PERSIST_SEC), nu un singur cadru. Fără câmpul
    # low_a_sec (ex. contexte construite manual) păstrăm comportamentul vechi.
    low_sec = _num(fusion, "low_a_sec", LOW_A_PERSIST_SEC)
    if a < 0.75 and low_sec >= LOW_A_PERSIST_SEC:
        return 1, f"atenție redusă (a={a:.2f})"

    return 0, "șofer atent"


def wants_112(ctx: dict) -> bool:
    """112 doar la severitate 3 ȘI șoferul nu răspunde / spune că nu se simte bine.

    Fluxul din demo: semn grav -> alarmă + „Sunteți bine?” -> fără răspuns -> 112.
    Contează doar răspunsul la „Sunteți bine?”: fără răspuns la „Doriți o pauză?”
    escaladează întâi la alarmă urgentă + „Sunteți bine?”, nu direct la 112.
    """
    sev, _ = rule_severity(ctx)
    resp = ctx.get("driver_response") or {}
    asked = resp.get("question")
    return (sev == 3 and answer_status(resp) in ("no_answer", "distress")
            and (asked is None or asked == PHRASES["ask_ok"]))


def simple_112_report(ctx: dict, reason: str = "") -> str:
    from .reports import build_112_report
    return build_112_report(ctx)


ACTIONS_BY_SEVERITY: dict[int, list[dict[str, Any]]] = {
    0: [],
    1: [{"tool": "speak", "args": {"text": PHRASES["remind_road"]}}],
    2: [
        {"tool": "play_alarm", "args": {"level": 2}},
        {"tool": "set_esp32", "args": {"level": 2}},
        {"tool": "suggest_break", "args": {}},
        {"tool": "ask_driver", "args": {"question": PHRASES["ask_tired"]}},
    ],
    3: [
        {"tool": "play_alarm", "args": {"level": 3}},
        {"tool": "set_esp32", "args": {"level": 3}},
        {"tool": "ask_driver", "args": {"question": PHRASES["ask_ok"]}},
    ],
}


def fallback_decision(ctx: dict, why: str = "fallback") -> dict:
    sev, reason = rule_severity(ctx)
    actions = [dict(a, args=dict(a["args"])) for a in ACTIONS_BY_SEVERITY[sev]]
    if sev == 1 and _num(ctx.get("vision") or {}, "phone", 0) >= 0.6:
        actions = [{"tool": "speak", "args": {"text": PHRASES["remind_phone"]}}]
    if sev == 3 and wants_112(ctx):
        actions = [a for a in actions if a["tool"] != "ask_driver"]
        actions.append({"tool": "call_112", "args": {"report": simple_112_report(ctx)}})
    return {
        "a": (ctx.get("fusion") or {}).get("a"),
        "severity": sev,
        "actions": actions,
        "reason": reason,
        "source": "rules",
        "fallback_reason": why,
    }
