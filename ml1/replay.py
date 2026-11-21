"""Rulează fuziunea peste un fișier JSONL și afișează scorul în timp.

    python -m ml1.replay tests/data/drowsy.jsonl
    python -m ml1.replay tests/data/collapse.jsonl --every 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fusion import AttentionFusion


def run(path: Path):
    fusion = AttentionFusion()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        if "driver_response" in msg:
            fusion.update_driver_response(msg["driver_response"], ts=msg.get("ts"))
            continue
        yield fusion.update(msg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--every", type=int, default=5, help="afișează la fiecare N cadre")
    args = p.parse_args()

    print(f"{'t':>6} {'a':>6} {'a_raw':>6} {'conf':>5}  cam  crit  bar")
    for i, r in enumerate(run(args.file)):
        if i % args.every:
            continue
        raw = "  -   " if r.a_raw is None else f"{r.a_raw:6.2f}"
        bar = "#" * round(r.a * 30)
        print(f"{r.ts:6.1f} {r.a:6.2f} {raw} {r.confidence:5.2f}  {'on ' if r.camera_on else 'off'}  {'!!' if r.critical else '  '}    {bar}")


if __name__ == "__main__":
    main()
