"""Validarea răspunsului LLM + plan B.

    decision = decide(ctx, llm_fn)      # mereu întoarce o decizie validă

1. parse_llm_output   text / JSON / răspuns Ollama (content sau tool_calls) -> dict
2. validate_decision  format: severitate, unelte cunoscute, argumente corecte
3. apply_safety       garduri: podea de severitate din reguli, 112 doar cu dovezi,
                      acțiuni coerente cu severitatea
Orice eroare fatală / timeout -> rules.fallback_decision (reguli fixe).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

from .actions import PHRASES, expand_actions
from .rules import (ACTIONS_BY_SEVERITY, _events, _num, fallback_decision, rule_severity,
                    simple_112_report, wants_112)
from .tools import SEVERITIES, TOOLS

LLM_TIMEOUT_SEC = 2.5
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")


class InvalidLLMOutput(ValueError):
    pass


# ------------------------------------------------------------------ parse ----

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _extract_json_object(text: str) -> dict:
    """Primul obiect JSON echilibrat din text (tolerant la ```json``` și text în jur)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    raise InvalidLLMOutput("niciun obiect JSON valid în răspuns")


def parse_llm_output(raw: Any) -> dict:
    """Acceptă: str, dict decizie, răspuns Ollama {"message": {...}} sau mesajul direct."""
    if raw is None:
        raise InvalidLLMOutput("răspuns gol")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        if not raw.strip():
            raise InvalidLLMOutput("răspuns gol")
        return _extract_json_object(raw)
    if not isinstance(raw, dict):
        raise InvalidLLMOutput(f"tip neașteptat: {type(raw).__name__}")

    msg = raw.get("message") if isinstance(raw.get("message"), dict) else raw
    tool_calls = msg.get("tool_calls")
    if "severity" not in msg and "content" not in msg and not tool_calls:
        return msg  # deja o decizie (posibil invalidă, o prinde validate_decision)
    if "severity" in msg:
        return msg

    content = msg.get("content") or ""
    decision: dict = {}
    if isinstance(content, str) and "{" in content:
        try:
            decision = _extract_json_object(content)
        except InvalidLLMOutput:
            decision = {}

    if tool_calls:
        actions = []
        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            actions.append({"tool": fn.get("name"), "args": fn.get("arguments", {})})
        decision.setdefault("actions", actions)
        if "reason" not in decision and isinstance(content, str) and content.strip() and "{" not in content:
            decision["reason"] = content.strip()

    if not decision:
        raise InvalidLLMOutput("nici JSON, nici tool_calls în răspuns")
    return decision


# --------------------------------------------------------------- validate ----

def _coerce_arg(name: str, value: Any, spec: dict) -> Any:
    t = spec["type"]
    if t == "integer":
        if isinstance(value, bool):
            raise InvalidLLMOutput(f"{name}: bool în loc de integer")
        if isinstance(value, str):
            value = value.strip()
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise InvalidLLMOutput(f"{name}: nu e număr ({value!r})")
        if f != int(f):
            raise InvalidLLMOutput(f"{name}: nu e întreg ({value!r})")
        v = int(f)
        lo, hi = spec.get("min"), spec.get("max")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise InvalidLLMOutput(f"{name}={v} în afara [{lo}, {hi}]")
        return v
    if t == "string":
        if not isinstance(value, str) or not value.strip():
            raise InvalidLLMOutput(f"{name}: text gol sau lipsă")
        return value.strip()[: spec.get("max_len", 500)]
    raise InvalidLLMOutput(f"tip necunoscut în spec: {t}")


def _infer_severity(actions: list[dict]) -> int | None:
    lv = [a["args"].get("level", 0) for a in actions if a["tool"] in ("play_alarm", "set_esp32")]
    if any(a["tool"] == "call_112" for a in actions):
        return 3
    if lv:
        return max(lv)
    return 1 if actions else None


def validate_decision(d: Any) -> tuple[dict, list[str]]:
    """Normalizează decizia. Ridică InvalidLLMOutput dacă nu se poate salva nimic."""
    if not isinstance(d, dict):
        raise InvalidLLMOutput("decizia nu e obiect")
    warnings: list[str] = []

    raw_actions = d.get("actions", [])
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        raise InvalidLLMOutput("`actions` nu e listă")

    actions: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw_actions):
        if not isinstance(item, dict):
            warnings.append(f"acțiunea #{i} ignorată: nu e obiect")
            continue
        tool = item.get("tool") or item.get("name") or item.get("function")
        tool = tool.strip().lower() if isinstance(tool, str) else None
        if tool not in TOOLS:
            warnings.append(f"unealtă necunoscută ignorată: {tool!r}")
            continue
        if tool in seen:
            warnings.append(f"duplicat ignorat: {tool}")
            continue
        args = item.get("args", item.get("arguments", item.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                warnings.append(f"{tool}: argumente JSON invalide, ignorat")
                continue
        if args is None:
            args = {}
        if not isinstance(args, dict):
            warnings.append(f"{tool}: argumentele nu sunt obiect, ignorat")
            continue
        spec = TOOLS[tool]["args"]
        clean, ok = {}, True
        for arg, aspec in spec.items():
            if arg not in args:
                if aspec.get("required"):
                    warnings.append(f"{tool}: lipsește `{arg}`, ignorat")
                    ok = False
                    break
                continue
            try:
                clean[arg] = _coerce_arg(f"{tool}.{arg}", args[arg], aspec)
            except InvalidLLMOutput as e:
                warnings.append(f"{e}, acțiune ignorată")
                ok = False
                break
        extra = set(args) - set(spec)
        if ok and extra:
            warnings.append(f"{tool}: argumente în plus ignorate {sorted(extra)}")
        if ok:
            actions.append({"tool": tool, "args": clean})
            seen.add(tool)

    sev = d.get("severity")
    if isinstance(sev, str) and sev.strip().isdigit():
        sev = int(sev.strip())
    if isinstance(sev, float) and sev.is_integer():
        sev = int(sev)
    if sev is None:
        sev = _infer_severity(actions)
        if sev is None:
            raise InvalidLLMOutput("lipsește `severity` și nu se poate deduce")
        warnings.append(f"severitate dedusă din acțiuni: {sev}")
    if isinstance(sev, bool) or sev not in SEVERITIES:
        raise InvalidLLMOutput(f"severitate invalidă: {d.get('severity')!r}")

    if raw_actions and not actions:
        raise InvalidLLMOutput("toate acțiunile au fost invalide")

    reason = d.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "(LLM-ul nu a dat un motiv)"
        warnings.append("lipsește `reason`")

    return {"severity": sev, "actions": actions, "reason": reason.strip()[:300]}, warnings


# ----------------------------------------------------------------- safety ----

def _ensure(actions: list[dict], tool: str, args: dict) -> None:
    for a in actions:
        if a["tool"] == tool:
            a["args"] = dict(args)
            return
    actions.append({"tool": tool, "args": dict(args)})


def apply_safety(d: dict, ctx: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []
    d = {**d, "actions": [dict(a, args=dict(a["args"])) for a in d["actions"]]}
    rule_sev, rule_reason = rule_severity(ctx)

    # 1. Podea: LLM-ul poate nuanța (o treaptă mai jos), dar nu poate ignora un caz clar de severitate 3.
    floor = 3 if rule_sev == 3 else max(0, rule_sev - 1)
    if d["severity"] < floor:
        notes.append(f"severitate ridicată {d['severity']}→{floor} (reguli: {rule_reason})")
        d["severity"] = floor
        d["reason"] = f"{d['reason']} [corectat: {rule_reason}]"

    # 1b. Plafon: modelele mici inventează alerte. LLM-ul poate ridica cu cel mult o treaptă
    #     peste reguli, iar dacă regulile spun 0 (nimic de raportat) nu poate ridica deloc.
    ceiling = rule_sev + 1 if rule_sev >= 1 else 0
    if d["severity"] > ceiling:
        notes.append(f"severitate coborâtă {d['severity']}→{ceiling} (reguli: {rule_reason})")
        d["severity"] = ceiling
        d["reason"] = rule_reason
        d["actions"] = [a for a in d["actions"] if a["tool"] not in ("speak", "play_alarm", "set_esp32", "ask_driver", "suggest_break")]
        if ceiling:
            d["actions"] = [dict(a, args=dict(a["args"])) for a in ACTIONS_BY_SEVERITY[ceiling]]

    sev = d["severity"]
    acts = d["actions"]

    # 1c. Reminderul de telefon / drum trebuie să aibă un semn real în date.
    vis = ctx.get("vision") or {}
    names = set(_events(vis))
    phone_ok = _num(vis, "phone", 0) >= 0.5 or "phone_in_hand" in names
    road_ok = vis.get("gaze_on_road") is False or "gaze_off_road" in names
    for a in list(acts):
        if a["tool"] == "speak":
            txt = a["args"]["text"]
            if (txt == PHRASES["remind_phone"] and not phone_ok) or (txt == PHRASES["remind_road"] and not road_ok):
                acts.remove(a)
                notes.append("reminder eliminat: nu există semnul în date")
    if sev == 1 and not acts:
        acts.extend(dict(a, args=dict(a["args"])) for a in ACTIONS_BY_SEVERITY[1])
        if phone_ok:
            acts[:] = [{"tool": "speak", "args": {"text": PHRASES["remind_phone"]}}]
        elif not road_ok:
            acts.clear()
            d["severity"], sev = 0, 0
            notes.append("severitate 1 fără semn concret: coborâtă la 0")

    # 1d. Fără întrebare nu există conversație vocală (și nici escaladarea pe tăcere).
    if sev >= 2 and not any(a["tool"] in ("ask_driver", "call_112") for a in acts):
        q = PHRASES["ask_ok"] if sev == 3 else PHRASES["ask_tired"]
        acts.append({"tool": "ask_driver", "args": {"question": q}})
        notes.append("ask_driver adăugat: severitatea ≥ 2 cere verificare vocală")
    if sev == 3:
        acts[:] = [a for a in acts if a["tool"] != "suggest_break"]

    # 2. Gardul 112: fals-pozitivul e scump, cerem dovezi din reguli.
    if any(a["tool"] == "call_112" for a in acts) and not (sev == 3 and wants_112(ctx)):
        acts[:] = [a for a in acts if a["tool"] != "call_112"]
        notes.append("call_112 blocat: lipsesc dovezile (severitate 3 + fără răspuns vocal)")
        if sev == 3:
            _ensure(acts, "ask_driver", {"question": "Sunteți bine?"})
    if sev == 3 and wants_112(ctx) and not any(a["tool"] == "call_112" for a in acts):
        acts.append({"tool": "call_112", "args": {"report": simple_112_report(ctx, rule_reason)}})
        notes.append("call_112 adăugat: severitate 3 și șoferul nu a răspuns")

    # 3. Nivelurile alarmei / ESP32 coerente cu severitatea.
    for a in list(acts):
        if a["tool"] in ("play_alarm", "set_esp32"):
            lvl = a["args"]["level"]
            if sev == 0:
                acts.remove(a)
                notes.append(f"{a['tool']} eliminat la severitate 0")
            elif lvl > sev:
                a["args"]["level"] = sev
                notes.append(f"{a['tool']} coborât {lvl}→{sev}")
    if sev == 3:
        _ensure(acts, "play_alarm", {"level": 3})
        _ensure(acts, "set_esp32", {"level": 3})

    # 4. Severitate > 0 fără nicio acțiune -> acțiunile standard.
    if sev > 0 and not acts:
        acts.extend(dict(a, args=dict(a["args"])) for a in ACTIONS_BY_SEVERITY[sev])
        notes.append("nicio acțiune validă: folosite acțiunile standard")

    return d, notes


# ----------------------------------------------------------------- decide ----

def decide(ctx: dict, llm_fn: Callable[[dict], Any] | None, timeout: float = LLM_TIMEOUT_SEC) -> dict:
    """Decizia finală pentru `vigil/decision`. Nu aruncă niciodată excepții."""
    t0 = time.perf_counter()

    def _done(dec: dict) -> dict:
        dec["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        return dec

    if llm_fn is None:
        return _done(fallback_decision(ctx, "LLM indisponibil"))
    try:
        raw = _executor.submit(llm_fn, ctx).result(timeout=timeout)
    except FutureTimeout:
        return _done(fallback_decision(ctx, f"timeout LLM > {timeout}s"))
    except Exception as e:  # conexiune Ollama căzută etc.
        return _done(fallback_decision(ctx, f"eroare LLM: {type(e).__name__}: {e}"))

    try:
        parsed = expand_actions(parse_llm_output(raw), ctx)
        dec, warnings = validate_decision(parsed)
    except InvalidLLMOutput as e:
        return _done(fallback_decision(ctx, f"răspuns invalid: {e}"))

    dec, notes = apply_safety(dec, ctx)
    dec.update({
        "a": (ctx.get("fusion") or {}).get("a"),
        "source": "llm",
        "warnings": warnings,
        "safety_notes": notes,
    })
    return _done(dec)
