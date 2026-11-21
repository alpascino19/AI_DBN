"""ML1 — o singură comandă pentru tot ce ține de motorul de decizie.

    python run_ml1.py                      # lista comenzilor
    python run_ml1.py check                # Ollama + model + pachete + broker
    python run_ml1.py test                 # cele 114 teste (fără Ollama, fără broker)
    python run_ml1.py dashboard --llm      # corp simulat + grafice, http://127.0.0.1:8765
    python run_ml1.py scenario collapse    # un scenariu din tests/data, în consolă (--llm = cu Qwen)
    python run_ml1.py bridge --llm         # ML1 pe MQTT: vigil/vision -> vigil/decision
    python run_ml1.py api --llm            # ML1 pe HTTP pentru serverul Spring Boot (docs/API.md)
    python run_ml1.py evaluate             # evaluarea -> docs/EVALUATION_RESULTS.md + docs/figures/
    python run_ml1.py bench                # latența și validitatea LLM-ului pe calculatorul curent
    python run_ml1.py data                 # regenerează tests/data/*.jsonl
    python run_ml1.py pdf                  # recompilează docs/TECHNICAL_DEEP_DIVE.pdf din LaTeX

Orice opțiune în plus trece mai departe la modulul respectiv (ex. `api --port 9000`, `test -v`).
Se poate rula din orice folder; folosește automat `.venv` din rădăcina repo-ului dacă există.
Sistemul complet (Vision + ML1 + ML2 + dashboard) se pornește din START/ (vezi START/README.md).
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # CodeLap/ML1
ROOT = HERE.parent                              # CodeLap
DATA = HERE / "tests" / "data"
LATEX = HERE / "docs" / "latex"
MODEL = os.environ.get("VIGIL_MODEL", "qwen2.5:1.5b")

# comandă -> (modul python, descriere)
MODULES = {
    "test":      ("unittest", "toate testele ML1 (~1 min)"),
    "dashboard": ("ml1.dashboard", "corp simulat + grafice live în browser (--llm, --autostart demo, --speed 2)"),
    "bridge":    ("ml1.mqtt_bridge", "ML1 pe MQTT, pentru Vision + ML2 (--llm, --host IP)"),
    "api":       ("ml1.http_api", "ML1 pe HTTP/SSE pentru Spring Boot (--llm, --port 8090, --mqtt-out)"),
    "evaluate":  ("ml1.evaluate", "evaluarea pe șoferi nevăzuți (--llm, --quick)"),
    "bench":     ("ml1.bench", "latența și validitatea LLM-ului (--models qwen2.5:1.5b --runs 5)"),
    "data":      ("ml1.fake_vision", "regenerează datele de test din tests/data"),
}
EXTRA = {
    "check":    "verifică Ollama, modelul, pachetele Python și brokerul MQTT",
    "scenario": f"rulează un scenariu în consolă: {', '.join(sorted(p.stem for p in DATA.glob('*.jsonl')))} (--llm)",
    "pdf":      "recompilează PDF-ul deep dive (xelatex, latexmk)",
}


def python() -> str:
    venv = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def run(cmd: list[str], cwd: Path = HERE) -> int:
    env = {**os.environ, "PYTHONUTF8": "1"}       # consola Windows: fără erori la ă/ș/ț
    try:
        return subprocess.call(cmd, cwd=cwd, env=env)
    except KeyboardInterrupt:
        return 130


def usage() -> int:
    print(__doc__.split("\n\n")[0] + "\n")
    print("Comenzi:")
    for name, (_, desc) in MODULES.items():
        print(f"  {name:<10} {desc}")
    for name, desc in EXTRA.items():
        print(f"  {name:<10} {desc}")
    print("\nExemplu: python run_ml1.py dashboard --autostart demo --speed 2 --llm")
    return 0


def check() -> int:
    ok = True

    def line(good: bool, what: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {what}" + ("" if good or not fix else f"  ->  {fix}"))

    print(f"Python: {python()}")
    missing = subprocess.run([python(), "-c", "import numpy, matplotlib, paho.mqtt"],
                             capture_output=True).returncode != 0
    line(not missing, "pachete numpy, matplotlib, paho-mqtt", f"{python()} -m pip install -r ML1/requirements.txt")
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            names = r.read().decode()
        line(True, "Ollama rulează")
        line(MODEL in names, f"modelul {MODEL}", f"ollama pull {MODEL}")
    except OSError:
        line(False, "Ollama rulează", "pornește aplicația Ollama (fără ea, ML1 merge doar cu reguli)")
    try:
        socket.create_connection(("127.0.0.1", 1883), timeout=1).close()
        line(True, "broker MQTT pe 1883 (doar pentru bridge / sistemul complet)")
    except OSError:
        line(False, "broker MQTT pe 1883", "Start-Service mosquitto (necesar doar pentru bridge)")
    print("Totul e în regulă." if ok else "Rezolvă punctele marcate cu !!.")
    return 0 if ok else 1


def scenario(args: list[str]) -> int:
    if not args or args[0].startswith("-"):
        print(EXTRA["scenario"])
        return 2
    name, rest = args[0], args[1:]
    path = Path(name) if name.endswith(".jsonl") else DATA / f"{name}.jsonl"
    if not path.exists():
        print(f"nu există {path}")
        return 2
    rest = ["--ollama" if a == "--llm" else a for a in rest]
    return run([python(), "-m", "ml1.engine", str(path), "--report", *rest])


def pdf() -> int:
    if not shutil.which("latexmk"):
        print("lipsește latexmk (MiKTeX sau TeX Live)")
        return 1
    code = run(["latexmk", "-xelatex", "-interaction=nonstopmode", "deep_dive.tex"], cwd=LATEX)
    if code == 0:
        shutil.copyfile(LATEX / "deep_dive.pdf", HERE / "docs" / "TECHNICAL_DEEP_DIVE.pdf")
        print("-> docs/TECHNICAL_DEEP_DIVE.pdf")
    return code


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # consolă cp1252: ? în loc de eroare
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return check()
    if cmd == "scenario":
        return scenario(rest)
    if cmd == "pdf":
        return pdf()
    if cmd not in MODULES:
        print(f"comandă necunoscută: {cmd}\n")
        usage()
        return 2
    return run([python(), "-m", MODULES[cmd][0], *rest])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
