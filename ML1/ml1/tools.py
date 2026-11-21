"""Uneltele pe care LLM-ul le poate apela (contract comun cu ML 2).

O singură sursă de adevăr: validarea le folosește acum, iar clientul Ollama
va genera din ele schema `tools` (vezi `ollama_tools()`).
"""

from __future__ import annotations

# nume -> {arg: (tip, obligatoriu, limite)}
TOOLS: dict[str, dict] = {
    "speak": {
        "description": "Spune un mesaj vocal șoferului (TTS).",
        "args": {"text": {"type": "string", "required": True, "max_len": 200}},
    },
    "play_alarm": {
        "description": "Pornește alarma sonoră. 1 = blândă, 3 = urgentă.",
        "args": {"level": {"type": "integer", "required": True, "min": 1, "max": 3}},
    },
    "ask_driver": {
        "description": "Pune o întrebare vocală și așteaptă răspunsul (timp + coerență).",
        "args": {"question": {"type": "string", "required": True, "max_len": 150}},
    },
    "set_esp32": {
        "description": "Setează nivelul LED/buzzer pe ESP32. 0 = oprit, 3 = urgent.",
        "args": {"level": {"type": "integer", "required": True, "min": 0, "max": 3}},
    },
    "call_112": {
        "description": "Apel 112 SIMULAT cu raportul semnelor observate. Doar la severitate 3.",
        "args": {"report": {"type": "string", "required": True, "max_len": 1000}},
    },
    "suggest_break": {
        "description": "Sugerează o pauză și afișează cea mai apropiată parcare.",
        "args": {},
    },
}

SEVERITIES = (0, 1, 2, 3)  # 0 = totul e în regulă, nicio acțiune


def ollama_tools() -> list[dict]:
    """Schema `tools` în formatul Ollama / OpenAI function calling."""
    out = []
    for name, spec in TOOLS.items():
        props, required = {}, []
        for arg, a in spec["args"].items():
            p = {"type": a["type"]}
            if "min" in a:
                p["minimum"] = a["min"]
            if "max" in a:
                p["maximum"] = a["max"]
            props[arg] = p
            if a.get("required"):
                required.append(arg)
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


# Schema pentru răspunsul LLM-ului (format compact cu ID-uri): vezi actions.LLM_SCHEMA.
