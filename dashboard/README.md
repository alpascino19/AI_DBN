# Dashboard Vigil (React)

Interfața live a proiectului: ce vede camera, ce decide LLM-ul, ce răspunde șoferul și ce face ESP32-ul. Pagina se conectează **direct la brokerul Mosquitto prin WebSocket**, fără backend.

```
Vision ─┐                                  ┌─► browser: dashboard React (mqtt.js)
ML 1   ─┼─► Mosquitto  1883 (module)       │
ML 2   ─┘              9001 (WebSocket) ───┘
```

## Pornire

```powershell
cd dashboard
npm install        # o singură dată
npm run dev        # http://localhost:5173
```
Pagina se poate deschide și de pe alt dispozitiv din rețea, de exemplu o tabletă la demo: `http://IP-LAPTOP:5173`.

### Brokerul trebuie să accepte WebSocket
În `C:\Program Files\mosquitto\mosquitto.conf` trebuie să existe, pe lângă portul 1883:
```
listener 9001 0.0.0.0
protocol websockets
```
Configul complet e în `ML2/mosquitto.conf`. Se copiază ca administrator, apoi `Restart-Service mosquitto` (vezi `ML2/README.md`, secțiunea 2.3).
Verificare: `netstat -ano | findstr :9001` trebuie să arate `LISTENING`.

### Alt broker
Implicit, pagina caută brokerul pe calculatorul care o servește (`ws://<host>:9001`). Pentru alt broker: `http://localhost:5173/?broker=ws://192.168.1.20:9001`.

## Ce afișează

| Panou | Sursa |
|---|---|
| Bara de sus: conexiune, calibrare, nivel ESP32, buton **Reset alerte** | `vigil/ml1/status`, `vigil/alert` |
| Atenție: scorul `a` și încrederea | `vigil/ml1/status` (1/s); dacă lipsește, `a` din ultima decizie |
| Camera față / Camera corp: scheletul | `vigil/vision` → `landmarks` (puncte de față), `body_landmarks` (25 de puncte MediaPipe Pose, unite cu linii) |
| Semnale cameră: EAR, PERCLOS, clipit, privire, cap căzut, telefon, respirație, postură, FPS | `vigil/vision`. Valorile lipsă apar „—”, valorile îngrijorătoare cu portocaliu |
| Evenimente active: tip, durată, încredere | `vigil/vision` → `events` |
| Decizie LLM: severitate, motiv, unelte, sursă, latență + istoric (ultimele 50) | `vigil/decision` |
| ECU (control fuzzy): mod, viteza de referință și ținta, asistența de bandă, distanța, frânarea, avariile + P(atent / distras / obosit / urgență) | `vigil/ecu` (5/s, de la ML1) |
| Răspunsul șoferului: textul și latența, sau „fără răspuns” | `vigil/driver_response` |
| **Fereastră roșie de urgență** cu raportul 112 | o decizie care conține `call_112` |

**Reset alerte** publică `0` pe `vigil/alert` (cu retain): stinge LED-ul ESP32.

## Testare fără cameră și fără ML 1

```powershell
cd Vision;  ..\.venv\Scripts\python -m vigil_vision --fake --scenario chest   # date false de cameră
cd ML2;     ..\.venv\Scripts\python executor.py                              # opțional: ESP32 + voce
cd ML2;     ..\.venv\Scripts\python fake_llm.py oboseala fara_raspuns -d 5   # decizii false
```
Datele false de la Vision nu conțin puncte de schelet, deci panourile de cameră rămân goale. Scheletul apare cu camera reală.

Teste automate (prelucrarea mesajelor și geometria scheletului):
```powershell
npm test
```

## Structura

| Fișier | Rol |
|---|---|
| `src/state.ts` | transformă mesajele MQTT în starea paginii. Mesajele stricate sunt ignorate, ca pagina să nu cadă |
| `src/skeleton.ts` | desenul scheletului din coordonatele 0..1 |
| `src/useVigil.ts` | conexiunea MQTT (reconectare automată la 2 s) și butonul de reset |
| `src/components/` | panourile |

## Ce urmează
- **Imaginile celor 2 camere.** Panourile au deja locul pregătit sub schelet. Formatul propus: `vigil/frame/face` și `vigil/frame/body`, cu `{"camera","ts","w","h","jpeg":"<base64>"}`, de stabilit cu echipa CV.
- **Butoanele Calibrare și Privacy Mode:** au nevoie de un topic de comandă către CV, de stabilit cu echipa.
