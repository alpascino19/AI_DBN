"""Meniul compact de acțiuni pe care îl alege LLM-ul.

Optimizare pentru ECU / domain controller:
  * LLM-ul NU scrie text liber pentru șofer: alege ID-uri din meniu
    -> ~25 tokeni de ieșire în loc de ~80, zero halucinații în mesajele vocale,
       fraze în română validate dinainte (cerință HMI în automotive).
  * Nivelul alarmei / ESP32 rezultă din severitate, nu e ales de model.

`expand_actions` transformă ID-urile în apelurile complete din `tools.TOOLS`
(contractul `vigil/decision` către ML 2 rămâne neschimbat).
"""

from __future__ import annotations

from typing import Any, Callable

PHRASES = {
    "remind_road": "Vă rugăm să păstrați privirea la drum.",
    "remind_phone": "Vă rugăm să lăsați telefonul și să priviți drumul.",
    # Vision (body/slump.py): aplecat / cocoșat de peste 10 s, respectiv corpul nu se mai vede pe scaun
    "remind_posture": "Vă rugăm să vă îndreptați spatele și să stați drept pe scaun.",
    "remind_position": "Vă rugăm să reveniți în poziția de condus.",
    "ask_tired": "Păreți obosit. Doriți să faceți o pauză?",
    "ask_ok": "Sunteți bine?",
}


def _report(ctx: dict) -> str:
    from .reports import build_112_report  # import târziu: evită ciclul rules <-> reports
    return build_112_report(ctx)


# id -> (unealtă, funcție (severitate, ctx) -> argumente)
ACTION_MENU: dict[str, tuple[str, Callable[[int, dict], dict]]] = {
    "remind_road": ("speak", lambda s, c: {"text": PHRASES["remind_road"]}),
    "remind_phone": ("speak", lambda s, c: {"text": PHRASES["remind_phone"]}),
    "remind_posture": ("speak", lambda s, c: {"text": PHRASES["remind_posture"]}),
    "remind_position": ("speak", lambda s, c: {"text": PHRASES["remind_position"]}),
    "alarm": ("play_alarm", lambda s, c: {"level": max(1, min(3, s))}),
    "esp32": ("set_esp32", lambda s, c: {"level": max(0, min(3, s))}),
    "suggest_break": ("suggest_break", lambda s, c: {}),
    "ask_tired": ("ask_driver", lambda s, c: {"question": PHRASES["ask_tired"]}),
    "ask_ok": ("ask_driver", lambda s, c: {"question": PHRASES["ask_ok"]}),
    "call_112": ("call_112", lambda s, c: {"report": _report(c)}),
}

# Schema pentru Ollama `format=` (structured output): modelul nu poate ieși din meniu.
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "integer", "enum": [0, 1, 2, 3]},
        "actions": {"type": "array", "items": {"type": "string", "enum": list(ACTION_MENU)}},
        "reason": {"type": "string"},
    },
    "required": ["severity", "actions", "reason"],
}


def _sev(v: Any) -> int:
    try:
        return max(0, min(3, int(v)))
    except (TypeError, ValueError):
        return 1


def expand_actions(decision: dict, ctx: dict) -> dict:
    """Înlocuiește ID-urile compacte (string) cu {"tool", "args"}. Restul rămâne neatins
    (validatorul acceptă în continuare și formatul complet / tool_calls)."""
    acts = decision.get("actions")
    if not isinstance(acts, list) or not any(isinstance(a, str) for a in acts):
        return decision
    sev = _sev(decision.get("severity"))
    out = []
    for a in acts:
        if isinstance(a, str):
            key = a.strip().lower()
            if key in ACTION_MENU:
                tool, build = ACTION_MENU[key]
                out.append({"tool": tool, "args": build(sev, ctx)})
            else:
                out.append({"tool": key, "args": {}})  # validatorul o raportează ca necunoscută
        else:
            out.append(a)
    return {**decision, "actions": out}
