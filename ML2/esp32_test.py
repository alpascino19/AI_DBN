"""Test cap-coadă ESP32: publică 0..3 pe vigil/alert și verifică pe serial că placa a primit fiecare nivel.

python esp32_test.py            # COM13 implicit
python esp32_test.py COM5
"""
import sys
import time

import paho.mqtt.client as mqtt
import serial

from executor import BROKER

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM13"

# resetăm placa (RTS), ca să prindem de la capăt mesajele de conectare
ser = serial.Serial(PORT, 115200, timeout=0.2)
ser.dtr, ser.rts = False, True
time.sleep(0.1)
ser.rts = False
ser.close()
time.sleep(1.5)  # USB-ul ESP32-C3 reapare după reset
ser = serial.Serial(PORT, 115200, timeout=0.2)
# așteptăm ca placa să raporteze conexiunea MQTT (vezi mesajele din vigil_alert.ino)
print(f"aștept ESP32 pe {PORT}...")
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    line = ser.readline().decode(errors="replace").strip()
    if line:
        print("  esp32:", line)
    if line == "mqtt: ok":
        break
else:
    sys.exit("ESP32 nu s-a conectat la broker în 30 s (vezi mesajele de mai sus)")

# la abonare brokerul trimite ultima stare reținută; o aruncăm, altfel ar trece drept răspuns la test
time.sleep(0.5)
ser.reset_input_buffer()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883)
client.loop_start()
failed = []
for level in (1, 2, 3, 0):
    sent = time.monotonic()
    client.publish("vigil/alert", str(level), qos=1, retain=True).wait_for_publish()
    while time.monotonic() - sent < 3:
        line = ser.readline().decode(errors="replace").strip()
        if line == f"alert {level}":
            print(f"nivel {level}: OK în {(time.monotonic() - sent) * 1000:.0f} ms — verifică LED/buzzer")
            break
    else:
        print(f"nivel {level}: NEPRIMIT")
        failed.append(level)
    time.sleep(2)  # timp să vezi LED-ul / să auzi buzzerul
client.loop_stop()
sys.exit(f"eșuat: {failed}" if failed else 0)
