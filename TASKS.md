# Vigil — Împărțirea taskurilor (4 persoane: 2 CV, 2 ML)

## CV 1 — Fața și ochii
- [ ] MediaPipe Face Mesh: EAR, PERCLOS, rata de clipire (prea rară = somnolență, prea deasă = clipit anormal)
- [ ] Poziția capului (pitch, yaw, roll) și direcția privirii → `gaze_on_road`
- [ ] Căderea capului + ochi închiși prelungit (pierderea cunoștinței)
- [ ] Calibrarea din primele 30 s (praguri adaptate șoferului)
- [ ] **Dashboard** (FastAPI + web / Streamlit): scor `a`, contribuția semnalelor, scheletul feței, ponderea șofer/MPC, viteză vs referință

## CV 2 — Corp, mâini și obiecte
- [ ] **Bucla camerei**: un singur proces deschide camera și dă cadrul ambelor module
- [ ] MediaPipe Pose + Hands: mâna pe piept (încheietura în zona toracelui, câteva secunde)
- [ ] Respirație grea: oscilația umerilor/pieptului + gura deschisă
- [ ] YOLO clasa „cell phone” + corelare cu mâna și privirea
- [ ] Publică rezultatele pe `vigil/vision`

## ML 1 — Fuziune, LLM și voce
- [ ] Fuziune: semnale → scor `a` ∈ [0, 1] (medie ponderată, netezită); fără cameră → încredere mai mică
- [ ] LLM local (Ollama, Llama 3 / Mistral): JSON cu evenimente → nivel de alertă 0–3
- [ ] Verificare vocală: întrebare TTS → STT, timp și coerență răspuns
- [ ] Raport 112 și raport post-cursă (+ șabloane fixe ca plan B la latență)
- [ ] **Owner pitch**

## ML 2 — Control și simulare
- [ ] `highway-env` + semnalul de comportament la volan (corecții, ieșiri de pe bandă)
- [ ] Controler P de rezervă (primul, ca să existe mereu ceva care conduce)
- [ ] MPC `cvxpy` + OSQP: cost cu `a`, constrângeri pe stare, orizont ~2 s (20 pași la 10 Hz)
- [ ] Manevra de risc minim: bandă de urgență, viteză → 0, avarii
- [ ] **ESP32 + MQTT**: LED/buzzer după nivelul de alertă

> ⚠️ MPC-ul e control, nu ML — cel mai mare risc tehnic. Începe imediat. Dacă nu converge la jumătatea timpului, demo-ul merge pe controlerul P.

## Contracte (stabilite în prima oră)
Comunicare prin brokerul MQTT local (oricum necesar pentru ESP32), ~10 Hz.

| Topic | De la → la | Conținut |
|---|---|---|
| `vigil/vision` | CV 1+2 → ML 1, dashboard | `{"ear":0.21,"perclos":0.3,"blink_rate":28,"gaze_on_road":false,"head_drop":false,"hand_on_chest":0.8,"breathing":"heavy","phone":0.1,"landmarks":[...]}` |
| `vigil/driving` | ML 2 → ML 1 | `{"lane_offset":0.4,"steer_activity":0.05}` |
| `vigil/attention` | ML 1 → ML 2, dashboard | `{"a":0.35,"state":"fatigue","confidence":0.7,"alert_level":2}` |
| `vigil/alert` | ML 1 → ESP32 | `2` |

## Etape
1. **Prima oră:** contractele de mai sus + fiecare scrie un generator de date false pentru topicul lui.
2. **Prima jumătate:** totul merge cap-coadă pe date false (cameră → fuziune → MPC → dashboard).
3. **A doua jumătate:** date reale în locul celor false, reglarea pragurilor la lumina din sală.
4. **Ultimele ore:** fără funcții noi, repetăm demo-ul de 3 minute.

## Owneri pe scenariul de demo
| Pas | Owner |
|---|---|
| Calibrare (30 s) | CV 1 |
| Conducere autonomă + predare control | ML 2 (+ ML 1 pentru răspunsul vocal) |
| Distragere / telefon | CV 2 |
| Somnolență | CV 1 |
| Mână pe piept | CV 2 |
| Cap căzut, fără răspuns → tragere pe dreapta + raport 112 | CV 1, ML 1, ML 2 |
| Privacy mode (camera oprită) | ML 1 |
