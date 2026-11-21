"""Benchmark pentru alegerea modelului (rulează pe hardware-ul țintă).

    ollama pull qwen2.5:1.5b
    python -m ml1.bench --models qwen2.5:1.5b llama3.2:1b qwen3:1.7b --runs 5

Măsoară pe contextele din tests/data/llm_contexts.json:
  * latența p50 / p95 (contează p95: bugetul e 2.5 s, ținta < 1 s)
  * % răspunsuri valide fără corecții (source=llm, fără warnings)
  * % acord cu severitatea așteptată
  * % corecții de siguranță (cât de des a trebuit să intervină garda)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .llm_client import OllamaClient
from .prompt import approx_tokens, build_messages
from .validate import decide

CTX_FILE = Path(__file__).resolve().parent.parent / "tests" / "data" / "llm_contexts.json"
EXPECTED = {"calm": 0, "phone": 1, "drowsy": 2, "chest": 3, "collapse_no_answer": 3}


def bench(model: str, runs: int, timeout: float) -> dict:
    contexts = json.loads(CTX_FILE.read_text(encoding="utf-8"))
    llm = OllamaClient(model=model, timeout=timeout)
    if not llm.available():
        return {"model": model, "error": f"modelul nu e instalat sau Ollama nu rulează (ollama pull {model})"}
    warm = llm.warmup()

    lat, valid, agree, safety, total = [], 0, 0, 0, 0
    for _ in range(runs):
        for name, ctx in contexts.items():
            t0 = time.perf_counter()
            d = decide(ctx, llm, timeout=timeout)
            lat.append(time.perf_counter() - t0)
            total += 1
            valid += d["source"] == "llm" and not d.get("warnings")
            agree += d["severity"] == EXPECTED[name]
            safety += bool(d.get("safety_notes"))
    lat.sort()
    return {
        "model": model,
        "warmup_s": round(warm, 2),
        "p50_ms": round(statistics.median(lat) * 1000),
        "p95_ms": round(lat[max(0, int(len(lat) * 0.95) - 1)] * 1000),
        "valid_%": round(100 * valid / total),
        "severity_ok_%": round(100 * agree / total),
        "safety_fix_%": round(100 * safety / total),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "llama3.2:1b"])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--timeout", type=float, default=2.5)
    args = p.parse_args()

    ctx = json.loads(CTX_FILE.read_text(encoding="utf-8"))["drowsy"]
    print(f"prompt ≈ {approx_tokens(build_messages(ctx))} tokeni (prefix fix + ~80 variabili)\n")
    for m in args.models:
        print(json.dumps(bench(m, args.runs, args.timeout), ensure_ascii=False))


if __name__ == "__main__":
    main()
