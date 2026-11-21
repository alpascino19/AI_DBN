"""Pornește tot sistemul Vigil: Vision (CV1 + CV2) -> ML1 (decizie) -> ML2 (voce, alarmă, ESP32, 112 simulat)
+ dashboard-ul React (browser). Variantele cu dublu-clic sunt fișierele .bat din același folder (START/).

    .venv\\Scripts\\python START\\run_vigil.py                 # cameră reală + LLM + voce + dashboard
    .venv\\Scripts\\python START\\run_vigil.py --fake          # fără cameră: scenariul demo al echipei CV
    .venv\\Scripts\\python START\\run_vigil.py --fake --scenario chest --no-llm
    .venv\\Scripts\\python START\\run_vigil.py --input http    # ML1 primește cadre prin HTTP (serverul Spring Boot)
    .venv\\Scripts\\python START\\run_vigil.py --check         # doar verificările, nu pornește nimic

Ordinea de pornire: ML2 (încarcă vocea, 10–50 s) -> ML1 (warmup LLM) -> Vision -> dashboard. Toate comunică
prin brokerul MQTT local (Mosquitto): modulele pe 1883, dashboard-ul din browser pe 9001 (WebSocket).

    Vision ──vigil/vision──► ML1 ──vigil/decision──► ML2 ──vigil/alert──► ESP32
                               ▲                       │
                               └──vigil/driver_response┘          (dashboard-ul le citește pe toate)

Ctrl+C oprește toate procesele. Logurile complete sunt în logs/<modul>.log (la rădăcina repo-ului).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # CodeLap/
LOGS = ROOT / "logs"
DASHBOARD = ROOT / "dashboard"
ML1_MODEL = os.environ.get("VIGIL_MODEL", "qwen2.5:1.5b")
DASH_URL = "http://localhost:5173"
COLORS = {"ML2": "\033[35m", "ML1": "\033[36m", "CV": "\033[33m", "WEB": "\033[34m", "VIGIL": "\033[32m"}
RESET = "\033[0m"


def say(tag: str, msg: str) -> None:
    print(f"{COLORS.get(tag, '')}[{tag:>5}]{RESET} {msg}", flush=True)


# ------------------------------------------------------------- verificări ----

def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except OSError:
        return None


def fresh_path() -> str:
    """PATH din registru (sistem + utilizator) + cel curent: programele instalate după pornirea terminalului
    (ffmpeg pentru vocea edge-tts a ML2, Node.js pentru dashboard) sunt găsite fără repornire."""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if os.name == "nt":
        import winreg
        for root, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                          (winreg.HKEY_CURRENT_USER, "Environment")):
            try:
                with winreg.OpenKey(root, key) as k:
                    value = os.path.expandvars(winreg.QueryValueEx(k, "Path")[0])
                parts += [p for p in value.split(os.pathsep) if p and p not in parts]
            except OSError:
                pass
    return os.pathsep.join(parts)


def python_has(mod: str) -> bool:
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0


def start_ollama(wait: float = 20.0) -> bool:
    """Pornește Ollama dacă e instalat dar oprit (aplicația din tray pornește și serverul)."""
    app = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
    exe = shutil.which("ollama", path=fresh_path())
    if os.name == "nt" and app.exists():
        cmd = [str(app)]
    elif exe:
        cmd = [exe, "serve"]
    else:
        return False
    say("VIGIL", "pornesc Ollama...")
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    t_end = time.time() + wait
    while time.time() < t_end:
        if port_open("127.0.0.1", 11434):
            return True
        time.sleep(0.5)
    return False


def voice_model(args) -> str:
    """Modelul conversației ML2: --voice-llm, altfel cel implicit al ML2 (VIGIL_MODEL / qwen2.5:1.5b)."""
    return args.voice_llm or ML1_MODEL


def preflight(args) -> bool:
    ok = True
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    say("VIGIL", f"Python: {sys.executable} {'(venv)' if in_venv else '(ATENȚIE: nu e .venv)'}")
    if not port_open(args.broker, 1883):
        say("VIGIL", f"EROARE: brokerul MQTT nu răspunde pe {args.broker}:1883 "
                     "(rulează START\\1_instalare.bat; serviciul mosquitto pornește singur)")
        ok = False
    else:
        say("VIGIL", f"broker MQTT: {args.broker}:1883 OK")
    models = ollama_models()
    if models is None and start_ollama():
        models = ollama_models()
    if models is None:
        say("VIGIL", "Ollama nu răspunde: ML1 merge doar cu reguli, ML2 fără replici generate (pornește Ollama)")
    else:
        need = [(ML1_MODEL, "ML1")] + ([] if args.no_ml2 else [(voice_model(args), "ML2 (conversație)")])
        for name, who in need:
            have = any(m == name or m.startswith(name + "-") or m == name + ":latest" for m in models)
            say("VIGIL", f"model {name} pentru {who}: {'OK' if have else 'LIPSEȘTE: ollama pull ' + name}")
    needed = [("paho.mqtt", "toate"), ("numpy", "toate")]
    if not args.fake and not args.no_cv and args.input == "mqtt":
        needed += [("cv2", "Vision"), ("mediapipe", "Vision"), ("ultralytics", "Vision")]
    if not args.no_ml2:
        needed += [("sounddevice", "ML2"), ("faster_whisper", "ML2"), ("edge_tts", "ML2")]
    for mod, who in needed:
        if not python_has(mod):
            say("VIGIL", f"EROARE: lipsește pachetul `{mod}` ({who}): rulează START\\1_instalare.bat")
            ok = False
    if not args.no_dashboard:
        if not shutil.which("npm", path=fresh_path()):
            say("VIGIL", "dashboard: lipsește Node.js (START\\1_instalare.bat) -> pornesc fără dashboard")
            args.no_dashboard = True
        elif not (DASHBOARD / "node_modules").exists():
            say("VIGIL", "dashboard: lipsesc pachetele (cd dashboard; npm install) -> pornesc fără dashboard")
            args.no_dashboard = True
        elif not port_open(args.broker, 9001):
            say("VIGIL", "dashboard: brokerul nu ascultă pe 9001 (WebSocket) -> pagina nu va primi date. "
                         "Rulează o dată START\\0_broker_admin.bat (cere drepturi de administrator)")
        else:
            say("VIGIL", f"dashboard: {DASH_URL} (broker WebSocket 9001 OK)")
    return ok


# --------------------------------------------------------------- procese ----

class Proc:
    def __init__(self, tag: str, cmd: list[str], cwd: Path, env: dict):
        self.tag, self.cmd, self.cwd = tag, cmd, cwd
        LOGS.mkdir(exist_ok=True)
        self.log = open(LOGS / f"{tag.lower()}.log", "a", encoding="utf-8")
        self.log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                  bufsize=1, creationflags=flags)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.p.stdout:
            line = line.rstrip()
            self.log.write(line + "\n")
            self.log.flush()
            if line:
                say(self.tag, line)
        say(self.tag, f"proces încheiat (cod {self.p.wait()})")

    def stop(self) -> None:
        if self.p.poll() is None:
            if os.name == "nt":   # oprește tot arborele (thread-urile audio ale ML2, node sub npm)
                subprocess.run(["taskkill", "/PID", str(self.p.pid), "/T", "/F"], capture_output=True)
            else:
                self.p.terminate()
        self.log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pornește Vision + ML1 + ML2 + dashboard")
    ap.add_argument("--fake", action="store_true", help="Vision trimite date false (fără cameră)")
    ap.add_argument("--scenario", default="demo", help="cu --fake: demo | normal | phone | drowsy | chest | unconscious")
    ap.add_argument("--show", action="store_true", help="fereastra de debug a camerei (Vision)")
    ap.add_argument("--camera", type=int, help="indexul camerei")
    ap.add_argument("--no-llm", action="store_true", help="ML1 doar cu reguli fixe")
    ap.add_argument("--no-ml2", action="store_true", help="fără executorul ML2 (voce / alarmă)")
    ap.add_argument("--no-cv", action="store_true", help="fără Vision (ex. cadrele vin de la serverul Spring)")
    ap.add_argument("--no-dashboard", action="store_true", help="fără dashboard-ul React")
    ap.add_argument("--no-browser", action="store_true", help="nu deschide browserul automat")
    ap.add_argument("--input", choices=["mqtt", "http"], default="mqtt",
                    help="de unde primește ML1 cadrele: broker MQTT (Vision) sau HTTP (serverul Spring Boot)")
    ap.add_argument("--http-port", type=int, default=8090)
    ap.add_argument("--broker", default=os.environ.get("VIGIL_BROKER", "localhost"))
    ap.add_argument("--voice-llm", default=None,
                    help="modelul conversației ML2 (implicit ca ML2: qwen2.5:1.5b, același cu ML1). "
                         "Mai bun în română: gemma3:4b sau qwen2.5:3b")
    ap.add_argument("--check", action="store_true", help="doar verificările")
    ap.add_argument("--duration", type=float, help="oprește automat după atâtea secunde (teste)")
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")   # activează culorile ANSI în consola Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not preflight(args):
        say("VIGIL", "verificările au eșuat: rezolvă erorile de mai sus")
        sys.exit(1)
    if args.check:
        say("VIGIL", "totul e pregătit")
        return

    path = fresh_path()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
           "VIGIL_BROKER": args.broker, "PATH": path}
    env_ml2 = {**env, "VIGIL_MODEL": voice_model(args)}   # doar ML2; ML1 rămâne pe ML1_MODEL
    py = sys.executable
    procs: list[Proc] = []
    try:
        if not args.no_ml2:
            procs.append(Proc("ML2", [py, "executor.py"], ROOT / "ML2", env_ml2))
            time.sleep(1.0)
        if args.input == "http":
            cmd = [py, "-m", "ml1.http_api", "--host", "0.0.0.0", "--port", str(args.http_port),
                   "--mqtt-out", "--broker", args.broker]
        else:
            cmd = [py, "-m", "ml1.mqtt_bridge", "--host", args.broker]
        if not args.no_llm:
            cmd.append("--llm")
        procs.append(Proc("ML1", cmd, ROOT / "ML1", env))
        time.sleep(4.0 if not args.no_llm else 1.0)   # warmup LLM înainte să vină cadre
        if not args.no_cv and args.input == "mqtt":
            cv = [py, "-m", "vigil_vision"]
            if args.fake:
                cv += ["--fake", "--scenario", args.scenario]
            if args.show:
                cv.append("--show")
            if args.camera is not None:
                cv += ["--camera", str(args.camera)]
            procs.append(Proc("CV", cv, ROOT / "Vision", env))
        if not args.no_dashboard:
            npm = shutil.which("npm", path=path)
            procs.append(Proc("WEB", [npm, "run", "dev", "--", "--strictPort"], DASHBOARD, env))
            if not args.no_browser:
                threading.Timer(4.0, webbrowser.open, args=(DASH_URL,)).start()
        say("VIGIL", "sistemul rulează — Ctrl+C pentru oprire; loguri în logs/"
                     + ("" if args.no_dashboard else f"; dashboard: {DASH_URL}"))
        t_end = time.time() + args.duration if args.duration else None
        while any(p.p.poll() is None for p in procs):
            if t_end and time.time() >= t_end:
                say("VIGIL", f"--duration {args.duration:.0f} s atins")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        say("VIGIL", "oprire...")
    finally:
        for p in reversed(procs):
            p.stop()
        say("VIGIL", "oprit")


if __name__ == "__main__":
    main()
