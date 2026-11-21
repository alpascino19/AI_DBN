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

    # PERCLOS corectat cu valoarea personală din calibrare, dacă există (engine îl pune în ctx)
    perclos = _num(vision, "perclos_cal", _num(vision, "perclos", 0))
    heavy = (vision.get("breathing_state") == "heavy" or _has(evs, "heavy_breathing", 0.6)
             or (evs.get("hand_on_chest") or {}).get("breathing") == "heavy")
    chest = _has(evs, "hand_on_chest", 0.6, 3) and heavy
    collapse = ((vision.get("head_drop") is True and perclos >= 0.6)
                or _has(evs, "head_on_wheel", 0.6, 2)                       # CV2: prăbușit pe volan
                or (vision.get("posture_state") == "slumped" and _num(vision, "still_sec", 0) >= 2))
    eyes_closed_long = _has(evs, "eyes_closed", 0.7, 2)

    if fusion.get("critical") or collapse:
        return 3, "cap căzut / prăbușit, ochi închiși"
    if eyes_closed_long and ans in ("no_answer", "distress"):
        return 3, "ochi închiși prelungit + fără răspuns"
    if chest:
        if ans == "ok":  # a răspuns repede și coerent: rămânem vigilenți, dar nu urgență
            return 2, "mână pe piept, dar șoferul răspunde coerent"
        return 3, "mână pe piept + respirație grea"
    if ans in ("no_answer", "distress") and a < 0.5:
        return 3, "fără răspuns coerent + atenție scăzută"

    # Ochi închiși ≥ 2 s fără cap căzut = micro-somn: alarmă puternică (2), nu urgență medicală (3).
    if eyes_closed_long:
        return 2, "ochi închiși prelungit (micro-somn)"
    if _has(evs, "microsleep", 0.6) or perclos >= 0.35 or a < 0.5:
        return 2, "semne de somnolență (PERCLOS / micro-somn)"
    if _has(evs, "yawn", 0.7) and perclos >= 0.2:
        return 2, "căscat + ochi obosiți"

    if _num(vision, "phone", 0) >= 0.6 or _has(evs, "phone_in_hand", 0.6, 2):
        return 1, "telefon în mână"
    if vision.get("gaze_on_road") is False and _has(evs, "gaze_off_road", 0.6, 2):
        return 1, "privire în afara drumului"
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
