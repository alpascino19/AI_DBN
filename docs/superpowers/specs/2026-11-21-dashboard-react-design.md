# Dashboard React Vigil — design

Aprobat: 2026-11-21.

## De ce
Dashboard-ul principal al proiectului nu există încă: `Vision/dashboard/` e un stub gol. Documentul de arhitectură cere o interfață React pentru demo. Ea arată live ce vede camera, ce decide LLM-ul, ce răspunde șoferul și ce face ESP32-ul.

## Obiective
- Afișare live a: scorului de atenție `a` și încrederii, semnalelor din cameră, evenimentelor active, scheletului feței și corpului, deciziei LLM (severitate, motiv, unelte, istoric), răspunsurilor șoferului, nivelului ESP32.
- O fereastră roșie de urgență la `call_112`, cu raportul.
- Butonul „Reset alerte”, care publică `0` pe `vigil/alert`.
- Două panouri de cameră pregătite să afișeze imaginile, când CV le va trimite.

## Ce nu intră (deocamdată)
- backend (Spring Boot sau alt server) și istoric persistent;
- imaginile camerelor: formatul nu e stabilit; propunerea e `vigil/frame/face|body` cu `{"camera","ts","w","h","jpeg":base64}`;
- butoanele de calibrare și Privacy Mode: cer un topic de comandă nou, care trebuie stabilit cu echipa.

## Anti-obiective
- încă un serviciu de pornit la demo;
- schimbarea contractelor MQTT existente.

## Abordări luate în calcul
| | Abordare | Cost |
|---|---|---|
| **A (aleasă)** | React se conectează direct la Mosquitto prin WebSocket (`mqtt.js`) | o linie în `mosquitto.conf` (drepturi de administrator), istoric doar în memorie |
| B | Spring Boot (STOMP + REST) + React, ca în documentul de arhitectură | de ~3–4 ori mai mult cod, încă un serviciu Java |
| C | Server Python (stub-ul FastAPI din Vision) + React | încă un proces, nu e „full React” |

A a fost aleasă de utilizator: „full React”, cât mai puține piese care pot pica la demo.

## Design
- **Loc:** `dashboard/` în rădăcina repo-ului. Vite + React + TypeScript + Tailwind; `npm run dev`.
- **Broker:** se adaugă `listener 9001` + `protocol websockets` în `mosquitto.conf`. Adresa brokerului e configurabilă (implicit hostname-ul paginii, port 9001).
- **Topicuri ascultate:** `vigil/vision` (10 Hz), `vigil/decision`, `vigil/driver_response`, `vigil/alert`, `vigil/ml1/status` (1 Hz, cu `a`, `confidence`, `calibration`).
- **Stare:** un reducer pur `(state, topic, payload) -> state`, testabil. Mesajele invalide sunt ignorate. Istoricul e limitat la ultimele 50 de decizii și răspunsuri.
- **Schelet:** `landmarks` (puncte de față, 0..1) se desenează ca puncte. `body_landmarks` (25 de puncte MediaPipe Pose, 0..1) se desenează cu liniile standard dintre articulații. Desenul e pe un canvas suprapus peste locul viitoarei imagini.
- **Urgență:** o decizie care conține `call_112` deschide un overlay roșu cu raportul, închis manual.
- **Șters:** stub-ul `Vision/dashboard/` (server.py gol, index.html gol).

## Lăsat la latitudinea constructorului
Așezarea exactă, culorile, componentele, alegerea testelor (vitest pentru reducer și pentru geometria scheletului).
