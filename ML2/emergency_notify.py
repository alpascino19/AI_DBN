"""Notificarea de urgență: la `call_112` anunță o persoană de contact (colegul), NU 112.

    vigil/decision ──► emergency_notify.py ── doar când ML 1 cere `call_112`:
                         1. apel GSM de pe telefonul Android legat prin USB (adb), dacă VIGIL_CALL_PHONE e setat
                         2. mesaj Telegram cu raportul (CallMeBot, text.php), dacă VIGIL_NOTIFY_TELEGRAM e setat

    python emergency_notify.py --dry-run     # ascultă și afișează ce ar trimite, fără să trimită nimic
    python emergency_notify.py               # pe bune (într-un terminal separat de executor.py)
    python emergency_notify.py --test        # trimite ACUM o notificare de test, fără MQTT
    python fake_llm.py fara_raspuns          # (alt terminal) publică un call_112 fals

Setările stau în ML2/.env (NU se urcă pe git, repo-ul e public). Model: ML2/.env.example.
Rulează separat de executor.py: dacă internetul / CallMeBot / telefonul nu merg, alarma, vocea
și ESP32 nu sunt afectate. Nu sună niciodată la 112: doar numărul și contul din .env.

Limitele CallMeBot (gratuit): câteva mesaje la rând, apoi refuză o vreme ("FLOOD" / "too many").
De aceea: o singură notificare pe urgență (blocare COOLDOWN_SEC) și cel mult RETRIES reîncercări
la RETRY_DELAY_SEC dacă serviciul refuză.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from executor import BROKER, validate

HERE = Path(__file__).resolve().parent
TOPIC = "vigil/decision"
COOLDOWN_SEC = 300          # o urgență = o notificare; următoarea abia peste 5 min
RETRIES = 3                 # încercări în total pentru mesajul Telegram
RETRY_DELAY_SEC = 70        # CallMeBot: cel mult un apel / minut la același om, iar FLOOD trece în câteva minute
MAX_REPORT_CHARS = 600
TEXT_URL = "https://api.callmebot.com/text.php"
# --test: același raport ca scenariul `fara_raspuns` din fake_llm.py
TEST_REPORT = ("Șofer fără reacție de 8 s. Semne: mână pe piept, respirație grea, cap căzut. "
               "Locație simulată: A1 km 42.")
REFUSED = ("error", "flood", "too many", "not allowed", "not authorized", "blocked")

log = logging.getLogger("notify")
Result = tuple[bool, str]


def load_env(path: Path) -> None:
    """KEY=VALUE din .env în os.environ (fără pachete în plus); variabilele deja setate au prioritate."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


# din raportul ML1 nu trimitem semnele și locația; titlul, ora și vehiculul sunt deja în mesaj
SKIP = ("semne", "locație", "locatie", "- ", "apel de urgență", "apel de urgenta", "ora:", "vehicul:")


def short_report(report: str) -> str:
    """Raportul ML1 fără „Semne ...” și „Locație ...”.

    Raportul real (ML1/ml1/reports.py) are câte un rând pe informație („Locație: ...”,
    „Semne observate:” + rânduri „- ...”); cel din fake_llm.py e o frază („... Semne: ... Locație ...”).
    """
    parts = []
    for line in report.splitlines():
        parts += re.split(r"(?<=[.!?])\s+", " ".join(line.split()))
    kept = "\n".join(p for p in parts if p and not p.lower().startswith(SKIP))
    if len(kept) > MAX_REPORT_CHARS:
        kept = kept[:MAX_REPORT_CHARS - 1] + "…"
    return kept


def build_message(report: str, now: Optional[float] = None) -> str:
    hour = time.strftime("%H:%M", time.localtime(now))
    details = short_report(report)
    lines = [
        "🚨 URGENȚĂ – Șoferul pare inconștient",
        f"Ora: {hour} · trimis automat de Vigil",
        "",
        "Ce s-a întâmplat: șoferul nu mai răspunde la alarme și la întrebările asistentului vocal.",
    ]
    if details:
        lines.append("Detalii: " + details.replace("\n", "\n  "))
    lines += [
        "",
        "Ce face mașina: avarii pornite, alarmă sonoră, încetinește spre oprire (simulat).",
        "Ce să faci: sună-l imediat pe șofer; dacă nu răspunde, anunță 112.",
        "",
        "(Demo hackathon – nu este o urgență reală.)",
    ]
    return "\n".join(lines)


# ---------- canale ----------
def send_telegram(user: str, text: str, timeout: float = 20.0) -> Result:
    """Mesaj Telegram prin CallMeBot. Destinatarul trebuie să fi autorizat botul (@CallMeBot_txtbot)."""
    url = TEXT_URL + "?" + urllib.parse.urlencode({"user": user, "text": text})
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return False, f"fără conexiune: {e}"
    except OSError as e:
        return False, str(e)
    plain = re.sub(r"<[^>]+>", " ", body)
    short = " ".join(plain.split())[:200]
    if any(word in plain.lower() for word in REFUSED):
        return False, short
    return True, short


def call_phone(adb: str, phone: str, timeout: float = 15.0) -> Result:
    """Apel GSM de pe Android-ul legat prin USB (depanare USB pornită, telefon deblocat)."""
    if not re.fullmatch(r"\+?\d{6,15}", phone):
        return False, f"număr invalid: {phone!r}"
    cmd = [adb, "shell", "am", "start", "-a", "android.intent.action.CALL", "-d", f"tel:{phone}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, f"nu găsesc adb: {adb}"
    except subprocess.TimeoutExpired:
        return False, "adb nu răspunde"
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0 or "error" in out.lower():
        return False, out or f"adb a ieșit cu {r.returncode}"
    return True, out


# ---------- logica ----------
class Notifier:
    def __init__(self, telegram_user: str = "", phone: str = "", adb: str = "adb",
                 send: Callable[[str, str], Result] = send_telegram,
                 call: Callable[[str, str], Result] = call_phone,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 cooldown: float = COOLDOWN_SEC, retries: int = RETRIES, retry_delay: float = RETRY_DELAY_SEC,
                 dry_run: bool = False) -> None:
        self.telegram_user, self.phone, self.adb = telegram_user, phone, adb
        self.send, self.call, self.clock, self.sleep = send, call, clock, sleep
        self.cooldown, self.retries, self.retry_delay = cooldown, retries, retry_delay
        self.dry_run = dry_run
        self.lock = threading.Lock()
        self.last_start: Optional[float] = None

    @staticmethod
    def report_from(payload) -> Optional[str]:
        """Raportul din primul `call_112` valid al deciziei, altfel None."""
        try:
            decision = json.loads(payload)
        except (ValueError, TypeError):
            return None
        if not isinstance(decision, dict) or not isinstance(decision.get("actions"), list):
            return None
        for action in decision["actions"]:
            ok = validate(action)
            if ok and ok[0] == "call_112":
                return ok[1]["report"]
        return None

    def handle(self, payload, background: bool = True) -> Optional[threading.Thread]:
        """Mesaj vigil/decision -> notificare (în fundal, ca bucla MQTT să nu aștepte reîncercările)."""
        report = self.report_from(payload)
        if report is None:
            return None
        with self.lock:
            now = self.clock()
            if self.last_start is not None and now - self.last_start < self.cooldown:
                log.info("call_112 primit, dar am notificat acum %.0f s -> sar (blocare %d s)",
                         now - self.last_start, self.cooldown)
                return None
            self.last_start = now
        if not background:
            self.notify(report)
            return None
        th = threading.Thread(target=self.notify, args=(report,), name="notify", daemon=True)
        th.start()
        return th

    def notify(self, report: str) -> dict:
        text = build_message(report)
        result = {"call": None, "telegram": None}
        if self.phone:
            if self.dry_run:
                log.warning("[dry-run] AȘ SUNA de pe Android la %s", self.phone)
                result["call"] = True
            else:
                ok, detail = self.call(self.adb, self.phone)
                result["call"] = ok
                (log.warning if ok else log.error)("apel adb către %s: %s (%s)",
                                                   self.phone, "pornit" if ok else "EȘUAT", detail)
        if self.telegram_user:
            if self.dry_run:
                log.warning("[dry-run] AȘ TRIMITE pe Telegram către %s:\n%s", self.telegram_user, text)
                result["telegram"] = True
            else:
                result["telegram"] = self._send_with_retries(text)
        if not self.phone and not self.telegram_user:
            log.error("nimic de trimis: completează VIGIL_NOTIFY_TELEGRAM și/sau VIGIL_CALL_PHONE în ML2/.env")
        return result

    def _send_with_retries(self, text: str) -> bool:
        for attempt in range(1, self.retries + 1):
            ok, detail = self.send(self.telegram_user, text)
            if ok:
                log.warning("Telegram către %s: TRIMIS (încercarea %d) %s", self.telegram_user, attempt, detail)
                return True
            log.error("Telegram către %s: refuzat (încercarea %d/%d): %s",
                      self.telegram_user, attempt, self.retries, detail)
            if attempt < self.retries:
                self.sleep(self.retry_delay)
        log.error("Telegram: renunț după %d încercări", self.retries)
        return False


def from_env(dry_run: bool) -> Notifier:
    return Notifier(telegram_user=os.environ.get("VIGIL_NOTIFY_TELEGRAM", "").strip(),
                    phone=os.environ.get("VIGIL_CALL_PHONE", "").replace(" ", "").strip(),
                    adb=os.environ.get("VIGIL_ADB", "adb").strip() or "adb",
                    dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description="La call_112 anunță persoana de contact din ML2/.env (nu 112)")
    ap.add_argument("--dry-run", action="store_true", help="afișează ce ar trimite, fără să trimită")
    ap.add_argument("--test", action="store_true", help="trimite acum o notificare de test, fără MQTT")
    ap.add_argument("--env", default=str(HERE / ".env"), help="fișierul cu setări (implicit ML2/.env)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    load_env(Path(args.env))
    notifier = from_env(args.dry_run)
    log.info("contact: Telegram=%s  apel=%s%s", notifier.telegram_user or "-", notifier.phone or "-",
             "  [DRY-RUN]" if args.dry_run else "")

    if args.test:
        notifier.notify(TEST_REPORT)
        return

    import paho.mqtt.client as mqtt

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(TOPIC, qos=1)
        log.info("conectat la %s, ascult %s (reacționez doar la call_112)", BROKER, TOPIC)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-emergency-notify")
    client.on_connect = on_connect
    client.on_message = lambda c, u, msg: notifier.handle(msg.payload)
    client.reconnect_delay_set(1, 10)
    client.connect_async(BROKER, 1883, keepalive=30)
    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
