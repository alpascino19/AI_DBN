"""ML 2 — executorul: vigil/decision -> validare -> coadă pe severitate -> un singur worker."""
import heapq
import itertools
import json
import logging
import os
import queue
import threading
import time

BROKER = os.environ.get("VIGIL_BROKER", "localhost")
DEBOUNCE_SEC = 10

# unealtă -> argumentele obligatorii și tipul lor (contractul cu ML 1)
TOOLS = {
    "speak": {"text": str},
    "play_alarm": {"level": int},
    "ask_driver": {"question": str},
    "set_esp32": {"level": int},
    "call_112": {"report": str},
    "suggest_break": {},
}
LEVELS = {"play_alarm": range(1, 4), "set_esp32": range(0, 4)}

log = logging.getLogger("ml2")


def validate(action):
    """Întoarce (tool, args) dacă acțiunea respectă contractul, altfel None."""
    if not isinstance(action, dict):
        return None
    tool, args = action.get("tool"), action.get("args", {})
    spec = TOOLS.get(tool)
    if spec is None or not isinstance(args, dict) or set(args) != set(spec):
        return None
    for name, typ in spec.items():
        value = args[name]
        if type(value) is not typ:  # type() și nu isinstance: True nu trece drept int
            return None
        if typ is str and not value.strip():
            return None
    if tool in LEVELS and args["level"] not in LEVELS[tool]:
        return None
    return tool, args


class Executor:
    def __init__(self, run_tool, publish_alert):
        self.run_tool = run_tool            # (tool, args, interrupt) -> None
        self.publish_alert = publish_alert  # (level) -> None, pentru ESP32
        self.q = queue.PriorityQueue()
        self.seq = itertools.count()        # FIFO în aceeași severitate
        self.lock = threading.Lock()
        self.interrupt = threading.Event()
        self.current = 0                    # severitatea acțiunii în curs, 0 = liber
        self.last_run = {}                  # (tool, severity) -> momentul ultimei programări

    def handle(self, payload):
        """Primește un mesaj vigil/decision (bytes/str) și programează acțiunile valide."""
        try:
            decision = json.loads(payload)
        except (ValueError, TypeError):
            log.warning("decizie respinsă, JSON invalid: %r", payload)
            return
        sev = decision.get("severity") if isinstance(decision, dict) else None
        # 0 = „revenire la normal” de la ML1 (vine cu set_esp32 0): fără ea, LED-ul rămânea aprins după o urgență
        if type(sev) is not int or sev not in (0, 1, 2, 3) or not isinstance(decision.get("actions"), list):
            log.warning("decizie respinsă, severitate/acțiuni invalide: %r", decision)
            return
        log.info("decizie sev=%d: %s", sev, decision.get("reason", ""))
        now = time.monotonic()
        with self.lock:
            if sev == 3:
                self._drop_below(3)
            for raw in decision["actions"]:
                valid = validate(raw)
                if valid is None:
                    log.warning("acțiune respinsă: %r", raw)
                    continue
                tool, args = valid
                if tool == "set_esp32":  # direct, pe lângă coadă
                    self.publish_alert(args["level"])
                    continue
                # cheia include severitatea, ca escaladarea (alarmă 1 -> 3) să nu fie blocată
                key = (tool, sev)
                if now - self.last_run.get(key, float("-inf")) < DEBOUNCE_SEC:
                    log.info("debounce: %s (sev %d)", tool, sev)
                    continue
                self.last_run[key] = now
                self.q.put((-sev, next(self.seq), tool, args))
            if sev == 3 and 0 < self.current < 3:
                self.interrupt.set()

    def _drop_below(self, sev):
        """Scoate din coadă acțiunile mai slabe: după o urgență, „Vă simțiți obosit?” nu mai are sens."""
        with self.q.mutex:
            dropped = [item[2] for item in self.q.queue if -item[0] < sev]
            self.q.queue[:] = [item for item in self.q.queue if -item[0] >= sev]
            heapq.heapify(self.q.queue)
        if dropped:
            log.info("urgență: anulat din coadă %s", dropped)

    def step(self):
        """Rulează următoarea acțiune din coadă; blochează până apare una."""
        prio, _, tool, args = self.q.get()
        sev = -prio
        with self.lock:
            self.interrupt.clear()
            # o urgență a sosit între get() și lock: nu mai pornim acțiunea mai slabă
            if sev < 3 and self.q.queue and self.q.queue[0][0] == -3:
                log.info("sărit %s: urgență în coadă", tool)
                return
            self.current = sev
        log.info("rulez %s %s (sev %d)", tool, args, sev)
        try:
            self.run_tool(tool, args, self.interrupt)
        except Exception:
            log.exception("unealta %s a eșuat", tool)
        finally:
            with self.lock:
                self.current = 0
        if self.interrupt.is_set():
            log.info("%s întrerupt de urgență", tool)


def main():
    import paho.mqtt.client as mqtt

    import actions

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-ml2")

    def publish(topic, payload):
        client.publish(topic, payload, qos=1)

    def publish_alert(level):
        # retain: ESP32 primește starea curentă și după o reconectare
        client.publish("vigil/alert", str(level), qos=1, retain=True)
        log.info("ESP32 <- %d", level)

    ex = Executor(lambda tool, args, interrupt: actions.run(tool, args, interrupt, publish), publish_alert)

    def on_connect(client, userdata, flags, reason_code, properties):
        log.info("conectat la %s (%s)", BROKER, reason_code)
        client.subscribe("vigil/decision", qos=1)

    client.on_connect = on_connect
    client.on_message = lambda c, u, msg: ex.handle(msg.payload)
    client.connect(BROKER, 1883)
    client.loop_start()  # reconectarea o face paho singur
    publish_alert(0)  # o singură dată, la pornire: altfel ESP32 rămâne pe ultima stare reținută (ex. buzzer)
    threading.Thread(target=actions.warmup, daemon=True).start()
    try:
        while True:
            ex.step()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()


if __name__ == "__main__":
    main()
