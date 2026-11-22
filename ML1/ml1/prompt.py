"""Promptul de sistem + exemple + linia de context compactă.

Optimizări pentru modele mici (1–2B) pe hardware auto:
  * SYSTEM + FEW_SHOT sunt IDENTICE la fiecare apel -> prefix comun ->
    llama.cpp/Ollama refolosește KV-cache-ul; per apel se procesează doar
    linia de context (~80 tokeni).
  * Instrucțiuni în engleză (modelele mici le urmează mai bine), răspuns
    `reason` scurt în română (apare pe dashboard).
  * Fără `landmarks` și fără JSON brut în intrare: o singură linie de cifre.
"""

from __future__ import annotations

import json

from .actions import ACTION_MENU
from .rules import _events, answer_status, rule_severity

SYSTEM = f"""You are the decision module of Vigil, an in-car driver monitoring system.
Each user message is one line of camera and voice signals. Reply with ONE JSON object only:
{{"severity": 0-3, "actions": [action ids], "reason": "max 12 words, in Romanian"}}

Severity and typical actions:
0 attentive or harmless (a 1 s glance, short yawn) -> []
1 short distraction (phone, gaze off road a few seconds) -> remind_phone or remind_road; bad_posture -> remind_posture; out_of_position -> remind_position
2 fatigue (high perclos, slow blinks, microsleep, yawning) -> alarm, esp32, suggest_break, ask_tired
3 possible medical emergency (hand on chest + heavy breathing, hand on chest for 6 s when breathing is not measured, head drop, head_on_wheel or slumped_side, eyes closed + no answer) -> alarm, esp32, ask_ok; add call_112 ONLY if answer is "no answer" or distress
Eyes closed for 2-3 s alone is a microsleep: severity 2, not 3.

Rules:
- Judge duration and combinations, never one frame.
- hint is the safety baseline. You may go at most 1 level below it, and never below 3 when hint is 3.
- A slow, missing or distressed answer makes it worse. A quick "sunt bine" makes it better.
- Allowed actions: {", ".join(ACTION_MENU)}."""


FEW_SHOT: list[tuple[str, dict]] = [
    ("a=0.93 conf=0.60 cam=on | perclos=0.06 ear=0.30 blink=16 | gaze_on_road=no head_drop=no phone=0.02 | events: gaze_off_road 1s | answer: none | hint=0",
     {"severity": 0, "actions": [], "reason": "privire scurtă, normală"}),
    ("a=0.65 conf=0.60 cam=on | perclos=0.07 ear=0.29 blink=15 | gaze_on_road=no head_drop=no phone=0.90 | events: phone_in_hand 6s | answer: none | hint=1",
     {"severity": 1, "actions": ["remind_phone"], "reason": "telefon în mână de 6 s"}),
    ("a=0.42 conf=0.60 cam=on | perclos=0.45 ear=0.19 blink=6 | gaze_on_road=yes head_drop=no phone=0.01 | events: microsleep 1s, yawn 3s | answer: none | hint=2",
     {"severity": 2, "actions": ["alarm", "esp32", "suggest_break", "ask_tired"], "reason": "PERCLOS mare, micro-somn și căscat"}),
    ("a=0.57 conf=0.60 cam=on | perclos=0.08 ear=0.29 blink=18 | gaze_on_road=yes head_drop=no phone=0.02 | events: hand_on_chest 5s heavy, heavy_breathing 5s | answer: none | hint=3",
     {"severity": 3, "actions": ["alarm", "esp32", "ask_ok"], "reason": "mână pe piept și respirație grea 5 s"}),
    ("a=0.17 conf=0.80 cam=on | perclos=0.95 ear=0.08 blink=0 | gaze_on_road=no head_drop=yes phone=0.01 | events: head_drop 6s, eyes_closed 6s | answer: no answer | hint=3",
     {"severity": 3, "actions": ["alarm", "esp32", "call_112"], "reason": "cap căzut, ochi închiși, nu răspunde"}),
]


def _fmt(v, nd=2) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    return "?"


def context_line(ctx: dict) -> str:
    fusion = ctx.get("fusion") or {}
    vision = ctx.get("vision") or {}
    cam = fusion.get("camera_on", True) and vision.get("camera_on", True) is not False
    parts = [f"a={_fmt(fusion.get('a'))} conf={_fmt(fusion.get('confidence'))} cam={'on' if cam else 'off'}"]
    if cam:
        perclos = vision.get("perclos_cal", vision.get("perclos"))  # corectat cu calibrarea personală
        face = (" face=missing" if vision.get("face_detected") is False
                else " face_camera=offline" if vision.get("face_camera") is False else "")
        parts.append(f"perclos={_fmt(perclos)} ear={_fmt(vision.get('ear'))} blink={_fmt(vision.get('blink_rate'), 0)}{face}")
        parts.append(f"gaze_on_road={_fmt(vision.get('gaze_on_road'))} head_drop={_fmt(vision.get('head_drop'))} phone={_fmt(vision.get('phone'))}")
        body = []
        if vision.get("breathing_state") in ("heavy", "normal"):
            body.append(f"breathing={vision['breathing_state']}")
        if vision.get("posture_state") in ("upright", "leaning", "slumped"):
            side = "_side" if vision.get("posture_direction") == "side" else ""
            body.append(f"posture={vision['posture_state']}{side}")
        if vision.get("body_camera") is False:
            body.append("body_camera=offline")
        if body:
            parts.append(" ".join(body))
        evs = []
        for name, e in _events(vision).items():
            if (e.get("confidence") or 0) < 0.5:
                continue
            s = f"{name} {float(e.get('duration_sec') or 0):.0f}s"
            if e.get("breathing") == "heavy":
                s += " heavy"
            evs.append(s)
        parts.append("events: " + (", ".join(evs) if evs else "none"))
    trend = ctx.get("a_trend")
    if isinstance(trend, (int, float)):
        parts.append(f"a 10s ago={trend:.2f}")

    resp = ctx.get("driver_response")
    st = answer_status(resp)
    ans = {"none": "none", "no_answer": "no answer"}.get(st)
    if ans is None:
        ans = f"\"{(resp.get('text') or '')[:40]}\" after {float(resp.get('latency_sec') or 0):.1f}s ({st})"
    parts.append(f"answer: {ans}")
    parts.append(f"hint={rule_severity(ctx)[0]}")
    return " | ".join(parts)


def build_messages(ctx: dict) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for user, out in FEW_SHOT:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": context_line(ctx)})
    return msgs


def approx_tokens(messages: list[dict]) -> int:
    """Estimare (~2.8 caractere / token), calibrată pe tokenizer-ul Qwen2.5: un prompt real a avut
    997 tokeni raportați de Ollama (prompt_eval_count) față de 806 estimați cu vechiul 3.5."""
    return int(sum(len(m["content"]) for m in messages) / 2.8) + 4 * len(messages)
