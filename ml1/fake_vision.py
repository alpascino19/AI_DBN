"""Generator de date false pentru `vigil/vision` (scenariile din demo).

    python -m ml1.fake_vision            # scrie tests/data/*.jsonl
    python -m ml1.fake_vision --print drowsy

Fiecare linie = un mesaj `vigil/vision` (+ `ts`, `camera_on`), 5 Hz.
Liniile cu cheia `driver_response` simulează mesaje de pe `vigil/driver_response`.
Seed fix -> date deterministe, bune pentru teste.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HZ = 5
OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


def _frame(rng: random.Random, t: float, **over) -> dict:
    """Cadru normal (șofer atent) + suprascrieri."""
    f = {
        "ts": round(t, 2),
        "camera_on": True,
        "ear": round(rng.gauss(0.30, 0.015), 3),
        "perclos": round(abs(rng.gauss(0.06, 0.02)), 3),
        "blink_rate": round(rng.gauss(16, 2), 1),
        "gaze_on_road": True,
        "head_drop": False,
        "phone": round(abs(rng.gauss(0.02, 0.02)), 3),
        "events": [],
    }
    f.update(over)
    return f


def _ev(name: str, dur: float, conf: float, **extra) -> dict:
    return {"event": name, "duration_sec": round(dur, 1), "confidence": conf, **extra}


def _ticks(start: float, seconds: float):
    for i in range(int(seconds * HZ)):
        yield i, start + i / HZ


# ------------------------------------------------------------- scenarii ----

def calibration(rng):
    """30 s de șofer normal."""
    return [_frame(rng, t) for _, t in _ticks(0, 30)]


def glance(rng):
    """10 s normal, 1 s privire în afara drumului, 9 s normal -> NU trebuie să alarmeze."""
    out = []
    for _, t in _ticks(0, 20):
        off = 10 <= t < 11
        out.append(_frame(rng, t, gaze_on_road=not off))
    return out


def phone(rng):
    """5 s normal, 8 s telefon în mână + privire jos, 5 s normal -> severitate 1."""
    out = []
    for _, t in _ticks(0, 18):
        if 5 <= t < 13:
            d = t - 5
            out.append(_frame(rng, t, gaze_on_road=rng.random() < 0.2,
                              phone=round(rng.uniform(0.8, 0.95), 2),
                              events=[_ev("phone_in_hand", d, 0.85)]))
        else:
            out.append(_frame(rng, t))
    return out


def drowsy(rng):
    """Somnolență progresivă pe 30 s: EAR scade, PERCLOS crește, clipit rar, căscat, micro-somn."""
    out = []
    for _, t in _ticks(0, 30):
        k = t / 30  # 0 -> 1
        events = []
        if 10 <= t < 14:
            events.append(_ev("yawn", t - 10, 0.8))
        if t >= 24:
            events.append(_ev("microsleep", min(t - 24, 1.5), 0.75))
        out.append(_frame(
            rng, t,
            ear=round(0.30 - 0.12 * k + rng.gauss(0, 0.01), 3),
            perclos=round(0.06 + 0.45 * k, 3),
            blink_rate=round(16 - 11 * k, 1),
            events=events,
        ))
    return out


def chest(rng):
    """Mână pe piept + respirație grea, crescând; șoferul răspunde lent la 12 s."""
    out = []
    for _, t in _ticks(0, 15):
        if t >= 3:
            d = t - 3
            out.append(_frame(rng, t, events=[
                _ev("hand_on_chest", d, 0.8, breathing="heavy"),
                _ev("heavy_breathing", d, 0.75),
            ]))
        else:
            out.append(_frame(rng, t))
    out.insert(12 * HZ, {"ts": 12.0, "driver_response": {"answered": True, "latency_sec": 5.5, "text": "nu mă simt bine"}})
    return out


def collapse(rng):
    """Mână pe piept -> cap căzut + ochi închiși, fără răspuns vocal -> severitate 3."""
    out = []
    for _, t in _ticks(0, 15):
        if t < 5:
            out.append(_frame(rng, t, events=[_ev("hand_on_chest", t, 0.8, breathing="heavy")]))
        else:
            d = t - 5
            out.append(_frame(rng, t, ear=0.08, perclos=0.95, blink_rate=0, gaze_on_road=False,
                              head_drop=True, events=[_ev("head_drop", d, 0.9), _ev("eyes_closed", d, 0.9)]))
    out.insert(10 * HZ, {"ts": 10.0, "driver_response": {"answered": False, "latency_sec": None, "text": None}})
    return out


def privacy(rng):
    """5 s cu cameră, apoi camera oprită: doar răspunsul vocal rămâne -> încredere scăzută."""
    out = [_frame(rng, t) for _, t in _ticks(0, 5)]
    out.append({"ts": 5.0, "driver_response": {"answered": True, "latency_sec": 1.2, "text": "da, sunt bine"}})
    out += [{"ts": round(t, 2), "camera_on": False} for _, t in _ticks(5, 10)]
    return out


def noisy(rng):
    """Date murdare: câmpuri lipsă, NaN, valori în afara intervalului, tipuri greșite."""
    out = []
    bad = [
        {"ear": "NaN"}, {"perclos": 1.7}, {"perclos": -0.2}, {"phone": None},
        {"gaze_on_road": "yes"}, {"events": "hand_on_chest"}, {"events": [None, 3, {"event": "unknown"}]},
        {"blink_rate": -5}, {"ear": None, "perclos": None},
    ]
    for i, t in _ticks(0, 10):
        f = _frame(rng, t)
        f.update(bad[i % len(bad)])
        out.append(f)
    out.append({})  # mesaj gol
    return out


def _calm(rng, start: float, seconds: float) -> list[dict]:
    return [_frame(rng, t) for _, t in _ticks(start, seconds)]


def demo(rng):
    """Demo complet (~2.3 min), în ordinea din TASKS.md: calibrare, telefon, somnolență,
    mână pe piept, cap căzut. Pauzele de calm lasă sistemul să revină la normal între episoade."""
    out: list[dict] = []
    t = 0.0

    def add(msgs, seconds):
        nonlocal t
        for m in msgs:
            if m.get("ts") is not None and m["ts"] < seconds:
                out.append({**m, "ts": round(m["ts"] + t, 2)})
        t += seconds

    add([m for m in calibration(rng) if m["ts"] < 10], 10)
    add(phone(rng), 18)
    out += _calm(rng, t, 10); t += 10
    add(drowsy(rng), 30)
    out += _calm(rng, t, 20); t += 20
    add(chest(rng), 15)
    out += _calm(rng, t, 20); t += 20
    add(collapse(rng), 15)
    return out


SCENARIOS = {f.__name__: f for f in (calibration, glance, phone, drowsy, chest, collapse, privacy, noisy, demo)}


def generate(name: str, seed: int = 42) -> list[dict]:
    return SCENARIOS[name](random.Random(seed))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--print", dest="show", choices=SCENARIOS, help="afișează un scenariu în loc să scrie fișiere")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if args.show:
        for m in generate(args.show):
            print(json.dumps(m, ensure_ascii=False))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for m in generate(name):
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"{path}  ({len(generate(name))} mesaje)")


if __name__ == "__main__":
    main()
