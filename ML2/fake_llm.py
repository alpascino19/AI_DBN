"""Publică decizii false pe vigil/decision, ca ML 2 să fie testat fără ML 1.

python fake_llm.py                      # toate scenariile, la 15 s
python fake_llm.py oboseala medical -d 2  # întreruperea: urgența vine peste oboseală
"""
import argparse
import json
import time

import paho.mqtt.client as mqtt

from executor import BROKER

SCENARIOS = {
    "distragere": {"a": 0.6, "severity": 1, "reason": "telefon 3 s", "actions": [
        {"tool": "speak", "args": {"text": "Vă rugăm să păstrați privirea la drum."}},
        {"tool": "set_esp32", "args": {"level": 1}}]},
    "oboseala": {"a": 0.4, "severity": 2, "reason": "PERCLOS 0.4, clipit rar", "actions": [
        {"tool": "play_alarm", "args": {"level": 2}},
        {"tool": "set_esp32", "args": {"level": 2}},
        {"tool": "suggest_break", "args": {}},
        {"tool": "ask_driver", "args": {"question": "Vă simțiți obosit?"}}]},
    "medical": {"a": 0.2, "severity": 3, "reason": "mână pe piept 4 s + respirație grea", "actions": [
        {"tool": "play_alarm", "args": {"level": 3}},
        {"tool": "set_esp32", "args": {"level": 3}},
        {"tool": "ask_driver", "args": {"question": "Sunteți bine?"}}]},
    "fara_raspuns": {"a": 0.0, "severity": 3, "reason": "cap căzut, fără răspuns vocal", "actions": [
        {"tool": "play_alarm", "args": {"level": 3}},
        {"tool": "set_esp32", "args": {"level": 3}},
        {"tool": "call_112", "args": {"report": "Șofer fără reacție de 8 s. Semne: mână pe piept, "
                                                "respirație grea, cap căzut. Locație simulată: A1 km 42."}}]},
    "invalid": {"a": 0.5, "severity": 2, "reason": "test validare", "actions": [
        {"tool": "rm_rf", "args": {}},
        {"tool": "play_alarm", "args": {"level": 9}}]},
    "reset": {"a": 1.0, "severity": 1, "reason": "șofer atent", "actions": [
        {"tool": "set_esp32", "args": {"level": 0}}]},
}

parser = argparse.ArgumentParser()
parser.add_argument("scenarios", nargs="*", default=list(SCENARIOS), help=" | ".join(SCENARIOS))
parser.add_argument("-d", "--delay", type=float, default=15)
opts = parser.parse_args()
if unknown := set(opts.scenarios) - set(SCENARIOS):
    parser.error(f"scenarii necunoscute: {unknown}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c, u, f, rc, p: c.subscribe("vigil/driver_response")
client.on_message = lambda c, u, msg: print("<- driver_response", msg.payload.decode())
client.connect(BROKER, 1883)
client.loop_start()
for i, name in enumerate(opts.scenarios):
    if i:
        time.sleep(opts.delay)
    print("->", name)
    client.publish("vigil/decision", json.dumps(SCENARIOS[name], ensure_ascii=False), qos=1).wait_for_publish()
time.sleep(10)  # mai ascultăm puțin răspunsul șoferului
client.loop_stop()
