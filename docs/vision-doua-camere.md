# Vision pe două laptopuri — conflicte MQTT și soluția propusă

**Pentru:** echipa Vision (CV 1 + CV 2) · **De la:** ML 2 · **Status:** propunere, de discutat împreună

## Contextul
La demo avem 4 calculatoare pe același hotspot:

| Calculator | Rol |
|---|---|
| Laptop 1 | camera feței (Vision) |
| Laptop 2 | camera corpului (Vision) |
| PC-ul lui Alex (`172.22.118.134`) | broker Mosquitto (porturile 1883 și 9001) + dashboard React + ESP32 |
| PC ML | ML 1 + ML 2 |

Laptopurile cu camere se conectează la broker pe **portul 1883**, la IP-ul de mai sus. Portul 9001 e doar pentru dashboard-ul din browser.

Așa cum e codul acum, două instanțe Vision care rulează în același timp intră în două conflicte.

---

## Conflictul 1: același `client_id` → laptopurile se deconectează reciproc
`publisher.py` se conectează mereu cu `client_id="vigil-vision"`. Mosquitto permite un singur client cu un nume dat. Când se conectează al doilea laptop, primul e deconectat, se reconectează și îl deconectează pe al doilea, la nesfârșit. În logul brokerului apare `session taken over`.

**Soluția:** în `Vision/vigil_vision/publisher.py`, în cele două locuri cu `client_id="vigil-vision"`:
```python
import socket
...
client_id=f"vigil-vision-{socket.gethostname()}"   # unic per laptop
```

---

## Conflictul 2: două fluxuri `vigil/vision` amestecate → deciziile ML 1 oscilează
Ambele laptopuri ar publica pe `vigil/vision`. Laptopul feței trimite EAR și PERCLOS, dar corpul îl vede prost. Laptopul corpului vede mâna pe piept, dar nu și fața. ML 1 primește mesajele pe rând, deci „vede” alternativ fața lipsă și corpul lipsă, iar scorul și deciziile sar.

**Soluția: fiecare laptop pe topicul lui, plus un mic program care le combină.** ML 1, ML 2 și dashboard-ul nu se schimbă deloc: primesc tot un singur `vigil/vision`, în formatul din `schemas.py`.

```
laptop față  ── vigil/vision/face ─┐
                                   ├─► merge (pe PC-ul cu brokerul) ── vigil/vision ──► ML 1, dashboard
laptop corp  ── vigil/vision/body ─┘
```

### a) Pe laptopuri: doar config, fără cod
Topicul e deja configurabil. În `Vision/config.yaml`:
```yaml
mqtt:
  host: 172.22.118.134
  port: 1883
  topic: vigil/vision/face    # pe laptopul cu camera corpului: vigil/vision/body
```

### b) Programul de combinare (propus: `Vision/vigil_vision/merge.py`)
Rulează pe PC-ul cu brokerul. Ascultă cele două topicuri și publică un mesaj complet de 10 ori pe secundă.
```python
"""Combină vigil/vision/face + vigil/vision/body într-un singur vigil/vision (pentru ML 1 și dashboard)."""
import json
import time

import paho.mqtt.client as mqtt

FACE = ("ear", "perclos", "blink_rate", "gaze_on_road", "landmarks", "face_detected")
BODY = ("phone", "breathing", "posture", "body_landmarks", "body_detected")
STALE_SEC = 1.0          # o cameră tăcută mai mult de atât e considerată căzută
latest = {}              # "face"/"body" -> (momentul primirii, mesajul)


def merge(now: float) -> dict:
    out = {"ts": time.time(), "events": [], "head_drop": False, "camera_on": False}
    for role, keys in (("face", FACE), ("body", BODY)):
        t, m = latest.get(role, (0.0, None))
        if m is None or now - t > STALE_SEC:
            continue                                   # câmpurile lipsesc -> ML 1 scade încrederea
        out.update({k: m.get(k) for k in keys})        # de la fiecare cameră doar ce vede ea bine
        out["events"] += m.get("events", [])
        out["head_drop"] |= bool(m.get("head_drop"))   # capul căzut îl pot vedea ambele camere
        out["camera_on"] = True
    return out


def on_message(client, userdata, msg):
    try:
        latest[msg.topic.rsplit("/", 1)[1]] = (time.monotonic(), json.loads(msg.payload))
    except ValueError:
        pass                                           # mesaj stricat: îl ignorăm


if __name__ == "__main__":
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vigil-merge")
    c.on_connect = lambda c, u, f, rc, p: c.subscribe("vigil/vision/+")   # nu prinde vigil/vision -> fără buclă
    c.on_message = on_message
    c.connect("localhost", 1883)
    c.loop_start()
    while True:
        c.publish("vigil/vision", json.dumps(merge(time.monotonic()), separators=(",", ":")))
        time.sleep(0.1)
```
Pornire pe PC-ul cu brokerul: `python -m vigil_vision.merge`.

### Test minim propus (`Vision/tests/test_merge.py`)
- ambele camere proaspete → `ear` vine de la face, `phone` de la body, evenimentele ambelor apar în listă;
- camera corpului tăcută peste 1 s → câmpurile corpului lipsesc, dar `camera_on` rămâne `true` (fața merge);
- ambele tăcute → `camera_on: false`, `events: []`;
- `head_drop` e `true` dacă oricare cameră îl raportează.

---

## (Opțional, mai târziu) Imaginile camerelor în dashboard
Dashboard-ul React are deja locul pregătit pentru imaginea live a fiecărei camere. Formatul propus:
- topic `vigil/frame/face` și `vigil/frame/body`, separat de `vigil/vision`, ca ML 1 să nu primească imagini;
- mesaj `{"camera": "face", "ts": 1716300000.5, "w": 640, "h": 480, "jpeg": "<JPEG base64>"}`;
- ~8 cadre pe secundă, micșorate la 640 px lățime, JPEG calitate 70, adică ~0,5–1 MB/s pentru ambele camere;
- în Privacy Mode nu se trimit imagini.

Avem o implementare de ~30 de linii pentru `publisher.py` (cu limită de cadre pe secundă și teste). O putem aplica noi sau o puteți face voi.

---

## Ce avem nevoie să decideți
1. Sunteți de acord cu soluția pentru Conflictul 2, adică topicuri separate + `merge`? Sau preferați altă variantă, de exemplu ca fiecare laptop să trimită doar câmpurile lui?
2. Listele `FACE` și `BODY` sunt corecte? De exemplu, `head_drop` vine de la ambele.
3. Cine scrie modificările: voi, sau le aplicăm noi și vă trimitem diff-ul?
4. Imaginile camerelor: acum sau după demo?
