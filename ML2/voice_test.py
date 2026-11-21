"""Asistentul vocal Virgil la microfon:
- idle: ascultă, dar răspunde doar la „Hei Virgil”;
- conversație: răspunde la ce spuneți; dacă tăceți în fereastra de ascultare -> alarmă (+ ESP32 nivel 2);
- când spuneți că sunteți bine / încheiați -> înapoi în idle.
Fiecare replică se scrie live în voice_live.json.

python voice_test.py   # Ctrl+C = stop
"""
import json
import os
import threading
import time
import wave

import numpy as np

import actions

OUT = actions.HERE / "voice_live.json"
REC = actions.HERE / "recordings"
state = {"status": "pornire", "near_dbfs_threshold": actions.NEAR_DBFS, "turns": []}
never = threading.Event()  # aici nu întrerupe nimeni
_speak, _listen = actions.speak, actions.listen
mode = "idle"
alerted = False

# ESP32 prin MQTT, dacă brokerul e pornit; fără el, alarma rămâne doar sonoră
try:
    import paho.mqtt.client as mqtt

    from executor import BROKER
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, 1883)
    client.loop_start()
except Exception as e:
    print(f"(fără ESP32: brokerul nu răspunde, {e!r})")
    client = None


def esp32(level):
    if client:
        client.publish("vigil/alert", str(level), qos=1, retain=True)


def update(**changes):
    state.update(changes, updated=time.strftime("%H:%M:%S"))
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)  # atomic: cine citește fișierul nu vede JSON pe jumătate


def speak_logged(text, interrupt):
    global mode
    mode = "conversație"
    state["turns"].append({"time": time.strftime("%H:%M:%S"), "virgil": text})
    update(status="vorbește Virgil")
    print("Virgil:", text)
    return _speak(text, interrupt)


def listen_logged(interrupt):
    update(status="idle — spuneți „Hei Virgil”" if mode == "idle" else f"ascultă {actions.LISTEN_SEC} s — răspundeți")
    start = time.monotonic()
    result = _listen(interrupt)
    response, audio, details = result
    if response["answered"] or details["rejected"]:  # tăcerea curată nu umple logul
        # fiecare frază se păstrează ca WAV: pe ea comparăm modelele Whisper cu vocea reală
        REC.mkdir(exist_ok=True)
        wav = REC / f"{time.strftime('%H%M%S')}.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(actions.MIC_SR)
            f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        state["turns"].append({
            "wav": wav.name,
            "time": time.strftime("%H:%M:%S"),
            "mode": mode,
            "sofer": response["text"],
            **response,
            **details,  # speech_dbfs = volumul vorbirii acceptate; rejected = ce s-a auzit dar s-a ignorat
            "processing_sec": round(time.monotonic() - start - len(audio) / actions.MIC_SR, 2),
            "mic_peak": round(float(np.abs(audio).max()), 4),
        })
    if response["answered"]:
        print(f"Tu ({mode}):", response["text"])
    return result


def on_silence():
    global alerted
    alerted = True
    state["turns"].append({"time": time.strftime("%H:%M:%S"), "alert": "fără răspuns"})
    update(status="ALARMĂ: fără răspuns")
    print("!! Fără răspuns — alarmă")
    esp32(2)
    actions.play_alarm(2, never)


def back_to_idle(*args, **kwargs):
    """Înfășoară converse: la final revenim în idle și stingem ESP32 dacă am alertat."""
    global mode, alerted
    try:
        return _converse(*args, **kwargs)
    finally:
        mode = "idle"
        if alerted:
            esp32(0)
            alerted = False
        update(status="idle — spuneți „Hei Virgil”")
        print("(idle — spuneți „Hei Virgil”)")


# asistentul apelează speak/listen/converse din modul, așa că le înlocuim cu variantele care loghează
_converse = actions.converse
actions.speak, actions.listen, actions.converse = speak_logged, listen_logged, back_to_idle
update(status="încarc modelele")
actions.warmup()
print("Gata (Ctrl+C = stop). Spuneți „Hei Virgil” ca să începeți.")
update(status="idle — spuneți „Hei Virgil”")
try:
    actions.assistant(never, lambda topic, payload: None, on_silence)
except KeyboardInterrupt:
    pass
finally:
    esp32(0)
    update(status="oprit")
