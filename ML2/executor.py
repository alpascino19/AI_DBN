"""ML 2 — executorul: vigil/decision -> validare -> coadă pe severitate -> un singur worker.
Plus legătura cu mașina: comanda fuzzy a ML1 (vigil/ecu) -> comenzi ECU (vigil/car/cmd, ECU_Arduino),
și butoanele asistentului din dashboard (vigil/assistant)."""
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
# cererile asistentului vin de pe vigil/assistant (butoanele din dashboard), nu de la ML 1
ASSISTANT_REQUESTS = ("listen", "time", "weather", "traffic")
ASSISTANT_DEBOUNCE_SEC = 2   # dublu-clic pe buton = o singură cerere

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
        self.current = 0                    # severitatea acțiunii în curs (0 = asistentul sau nimic)
        self.running = False                # rulează ceva acum (inclusiv asistentul, care are severitatea 0)
        self.last_run = {}                  # (tool, severity) -> momentul ultimei programări

    def handle(self, payload):
        """Primește un mesaj vigil/decision (bytes/str) și programează acțiunile valide."""
        try:
            decision = json.loads(payload)
        except (ValueError, TypeError):
            log.warning("decizie respinsă, JSON invalid: %r", payload)
            return
        sev = decision.get("severity") if isinstance(decision, dict) else None
        if type(sev) is not int or sev not in (0, 1, 2, 3) or not isinstance(decision.get("actions"), list):
            log.warning("decizie respinsă, severitate/acțiuni invalide: %r", decision)
            return
        log.info("decizie sev=%d: %s", sev, decision.get("reason", ""))
        if sev == 0:  # ML 1: revenire la normal -> stingem ESP32; restul acțiunilor (de obicei niciuna) merg normal
            self.publish_alert(0)
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
            # urgența întrerupe orice acțiune mai slabă; orice alertă ML 1 întrerupe asistentul (siguranța
            # are prioritate față de o întrebare despre vreme)
            if self.running and ((sev == 3 and self.current < 3) or (sev >= 1 and self.current == 0)):
                self.interrupt.set()

    def handle_assistant(self, payload):
        """Primește un mesaj vigil/assistant {"request": "listen" | "time" | "weather" | "traffic"}.
        Merge în aceeași coadă (un singur difuzor și microfon), cu prioritatea cea mai mică."""
        try:
            msg = json.loads(payload)
        except (ValueError, TypeError):
            msg = None
        request = msg.get("request") if isinstance(msg, dict) else None
        if request not in ASSISTANT_REQUESTS:
            log.warning("cerere asistent respinsă: %r", payload)
            return
        now = time.monotonic()
        with self.lock:
            key = ("assistant", request)
            if now - self.last_run.get(key, float("-inf")) < ASSISTANT_DEBOUNCE_SEC:
                return
            self.last_run[key] = now
            log.info("asistent: %s", request)
            self.q.put((0, next(self.seq), "assistant", {"request": request}))

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
            self.running = True
        log.info("rulez %s %s (sev %d)", tool, args, sev)
        try:
            self.run_tool(tool, args, self.interrupt)
        except Exception:
            log.exception("unealta %s a eșuat", tool)
        finally:
            with self.lock:
                self.current = 0
                self.running = False
        if self.interrupt.is_set():
            log.info("%s întrerupt de urgență", tool)


MRM_DECEL = 2.5       # m/s², ca ML1 (ecu_fuzzy.DECEL_MRM): oprire controlată pe banda de urgență
URGENT_DECEL = 2.0    # m/s², ca ML1 (DECEL_URGENT) la severitate 3 (atunci ML1 cere și avariile)
NORMAL_DECEL = 0.5    # m/s², ca ML1 (DECEL_NORMAL): încetinire blândă
MIN_SLOWDOWN_KMH = 5  # ținta fuzzy se mișcă la fiecare cadru; ECU primește doar scăderi reale
ECU_CRUISE_KMH = 100.0  # ECU_Arduino CRUISE_KMH: viteza mașinii simulate la pornire și după reset


class CarLink:
    """ML1 trimite pe vigil/ecu o comandă continuă (5 Hz); ECU-ul (ECU_Arduino) înțelege comenzi discrete pe
    vigil/car/cmd. Traducem doar schimbările: MRM -> emergency, ieșire din MRM -> cancel_emergency,
    țintă mai mică -> slow_down, avariile -> hazard, revenire la NORMAL -> reset."""

    def __init__(self):
        self.mrm = False
        self.hazard = False
        self.target = None    # ultima țintă trimisă (km/h); None = croazieră
        self.slowed = False   # ECU a încetinit și trebuie readus la croazieră
        self.ids = itertools.count(1)

    def _cmd(self, **fields):
        return {**fields, "id": f"ml2-{next(self.ids)}"}  # ECU îl întoarce în vigil/car/event

    def update(self, payload):
        """Un mesaj vigil/ecu -> lista de comenzi pentru vigil/car/cmd (de obicei goală)."""
        try:
            msg = json.loads(payload)
        except (ValueError, TypeError):
            return []
        if not isinstance(msg, dict):
            return []
        mode, target = msg.get("mode"), msg.get("target_kmh")
        if not isinstance(mode, str) or type(target) not in (int, float):
            log.warning("vigil/ecu respins: %r", msg)
            return []
        out = []
        if mode == "MRM":
            if not self.mrm:
                self.mrm, self.hazard, self.slowed, self.target = True, True, True, 0.0
                out.append(self._cmd(cmd="emergency", decel_mps2=MRM_DECEL))
            return out
        if self.mrm:  # ML1 iese din MRM doar la severitate 0
            self.mrm = False
            out.append(self._cmd(cmd="cancel_emergency"))
        hazards = msg.get("hazards") is True
        if hazards != self.hazard:
            self.hazard = hazards
            out.append(self._cmd(cmd="hazard", on=hazards))
        if mode == "NORMAL" and not hazards:
            if self.slowed:
                # shortcut: ECU nu are „accelerează” (resume păstrează viteza), iar reset sare direct la croazieră;
                # de cerut colegului o comandă set_speed dacă revenirea trebuie să fie treptată
                self.slowed, self.target = False, None
                out.append(self._cmd(cmd="reset"))
        # fără țintă trimisă, ECU merge cu viteza de croazieră (nu referința ML1, care după un MRM urcă de la 0)
        elif target <= (self.target if self.target is not None else ECU_CRUISE_KMH) - MIN_SLOWDOWN_KMH:
            # nu folosim decel_mps2 din vigil/ecu: e pasul referinței ML1 (0 cât timp ea urcă), nu frânarea dorită
            self.target, self.slowed = float(target), True
            out.append(self._cmd(cmd="slow_down", target_kmh=round(target, 1),
                                 decel_mps2=URGENT_DECEL if hazards else NORMAL_DECEL))
        return out


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
    car = CarLink()

    def to_car(payload):
        for cmd in car.update(payload):
            client.publish("vigil/car/cmd", json.dumps(cmd), qos=1)  # fără retain: o comandă nu se reia
            log.info("ECU <- %s", cmd)

    def on_connect(client, userdata, flags, reason_code, properties):
        log.info("conectat la %s (%s)", BROKER, reason_code)
        client.subscribe([("vigil/decision", 1), ("vigil/assistant", 1), ("vigil/ecu", 0)])

    def on_message(c, u, msg):
        if msg.topic == "vigil/assistant":
            ex.handle_assistant(msg.payload)
        elif msg.topic == "vigil/ecu":
            to_car(msg.payload)
        else:
            ex.handle(msg.payload)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883)
    client.loop_start()  # reconectarea o face paho singur
    publish_alert(0)  # o singură dată, la pornire: altfel ESP32 rămâne pe ultima stare reținută (ex. buzzer)
    # la pornire mașina simulată pleacă de la croazieră, fără avarii (ca alerta 0 de mai sus)
    client.publish("vigil/car/cmd", json.dumps({"cmd": "reset", "id": "ml2-start"}), qos=1)
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
