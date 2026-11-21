# Vigil pe 3 laptopuri — cine ce deschide

Toate trei pe **același hotspot**. Laptopul 1 e creierul (broker + ML1 + ML2 + dashboard), laptopurile 2 și 3 sunt
doar camere care trimit date la laptopul 1.

```
Laptop 2 (cameră FAȚĂ) ──┐                                  ┌─► ML1 ──► ML2 (voce, alarmă, ESP32, 112)
                         ├─► Laptop 1 (broker + combinare) ─┤
Laptop 3 (cameră CORP) ──┘                                  └─► dashboard (browser)
```

| | Laptop 1 — CENTRAL | Laptop 2 — FAȚA | Laptop 3 — CORPUL |
|---|---|---|---|
| **Ce are** | broker Mosquitto, ML1, ML2, dashboard, ESP32, boxe, microfon | camera spre fața șoferului | camera spre piept, mâini, volan |
| **Ce deschizi** | `START\4c_pornire_PC_central.bat` | `START\7_laptop_fata.bat` | `START\7_laptop_corp.bat` |
| **Ce îți cere** | nimic | IP-ul laptopului 1 | IP-ul laptopului 1 |
| **Verificare** | `START\8_cine_e_conectat.bat` | fereastra `vigil - FATA` | fereastra `vigil - CORP` |

IP-ul laptopului 1 pe hotspot (acum): **`172.22.118.20`**. Se poate schimba la o nouă conectare: îl arată
`8_cine_e_conectat.bat` și `4c`.

---

## Pregătire (o singură dată)

### Laptop 1
1. `START\1_instalare.bat` (durează 10–20 min: Python, Ollama + modele, Mosquitto, Node.js, dashboard).
2. `START\0_broker_admin.bat`, **clic dreapta → Run as administrator**: brokerul acceptă laptopurile din rețea
   și browserul (porturile 1883 și 9001).
3. Firewall: `0_broker_admin.bat` (pasul 2) adaugă singur regula „Vigil MQTT” pentru porturile 1883 și 9001, pe toate
   profilurile. Fără ea, Windows blochează laptopurile 2/3 chiar dacă brokerul merge (hotspot-urile sunt „Public”).
4. ESP32 și microfon / boxe conectate (vezi [ML2/README.md](../ML2/README.md)).

### Laptopurile 2 și 3
1. Copiați folderul `CodeLap` (USB sau `git pull`). Sunt necesare `START\` și `Vision\`.
2. Python 3.11+ instalat. Prima pornire a scriptului instalează singură pachetele Vision (câteva minute,
   cu internet) și descarcă modelele camerei.
3. Conectați-le la hotspot.

---

## La demo, în ordine

| # | Pe ce laptop | Ce faci | Ce trebuie să vezi |
|---|---|---|---|
| 1 | **1** | dublu-clic `4c_pornire_PC_central.bat` | `sistemul rulează`; dashboard-ul se deschide în browser (așteaptă ~30 s: ML2 încarcă vocea) |
| 2 | **2** | dublu-clic `7_laptop_fata.bat`, scrii IP-ul de mai sus | fereastra `vigil - FATA`; în consolă `MQTT conectat` |
| 3 | **3** | dublu-clic `7_laptop_corp.bat`, același IP | fereastra `vigil - CORP`; în consolă `MQTT conectat` |
| 4 | **1** | privești fereastra lui `4c` | `[NET] + conectat: ...` de două ori și `[MERGE] prima cameră face / body a răspuns` |
| 5 | **2** | șoferul se uită drept 5 s | calibrarea ochilor s-a încheiat |
| 6 | **1** | dashboard-ul | camerele și semnalele apar live |

Apoi scenariile din [README.md](README.md): telefon în mână, ochi închiși, mână pe piept, prăbușire. Oprire:
Ctrl+C în fereastra lui `4c` pe laptopul 1 și `q` în ferestrele camerelor.

---

## Dacă ceva nu merge

| Simptom | Unde | Ce faci |
|---|---|---|
| nu apare `[NET] + conectat` pentru un laptop | laptop 1 | același hotspot? IP corect? `0_broker_admin.bat` rulat? firewall 1883? Verifică cu `8_cine_e_conectat.bat` |
| laptopul 2/3 nu afișează `MQTT conectat`, curg doar linii `{...}` | laptop 2/3 | nu ajunge la broker. Test, în PowerShell: `Test-NetConnection 172.22.118.20 -Port 1883`. `TcpTestSucceeded : True` = ajunge (atunci IP-ul din comandă e altul); `False` = firewall pe laptopul 1 (rulează `0_broker_admin.bat` ca administrator) sau alt hotspot |
| e conectat, dar `prima cameră ... a răspuns` lipsește | laptop 1 | ați pornit același script pe ambele laptopuri; unul trebuie `fata`, celălalt `corp` |
| `camera body: nu mai trimite` | laptop 1 | laptopul 3 s-a oprit sau a pierdut Wi-Fi; ML1 continuă doar cu fața și scade încrederea |
| `Nu pot deschide camera 0` | laptop 2/3 | închideți Teams / Zoom / Camera; altă cameră: `START\7_laptop_fata.bat IP --index 1` |
| `Vigil rulează deja pe acest calculator` | laptop 1 | există o altă fereastră `4c` deschisă: închide-o și repornește |
| dashboard gol | laptop 1 | `0_broker_admin.bat` ca administrator (portul 9001) |
| fără voce | laptop 1 | `ollama` pornit? boxe / microfon alese corect? vezi `logs\ml2.log` |
| nicio decizie, deși camerele merg | laptop 1 | `logs\ml1.log` și `logs\merge.log` |

Logurile complete sunt pe laptopul 1, în `logs\`. Cele ale camerelor se văd în consola laptopurilor 2 și 3.

## Reguli de aur
* **Laptopul 1 pornește primul**, cu 30 s înainte de camere.
* **Un singur** `4c` pe laptopul 1 (nu două ferestre).
* **Un `fata` și un `corp`**, nu de două ori același.
* Combinarea (`vigil_vision.merge`) pornește doar din `4c`; nu o porniți de mână pe alt laptop.
* Dacă IP-ul laptopului 1 s-a schimbat, relansați scripturile de pe laptopurile 2 și 3 cu IP-ul nou.

## Video live în dashboard
Ambele camere trimit imaginea live, implicit (orice pornire Vision cu `--publish`): JPEG-uri de 640 px, 20 cadre/s
(~30 KB/cadru, ~0.6 MB/s pe cameră; prin hotspot ~10 Mbit/s pentru ambele), pe `vigil/video/face` și
`vigil/video/body`. Imaginea are firul ei, la viteza camerei (30 cadre/s), deci și camera corpului dă 20 cadre/s,
deși detecția corpului merge doar la 7–11. Ferestrele FAȚA și CORP din dashboard arată imaginea cu punctele peste
ea și, pe eticheta `LIVE`, câte cadre/s ajung de fapt.
* Cu o singură cameră (`4_pornire_camera.bat`), aceeași imagine apare în ambele ferestre.
* Dacă rețeaua nu ține pasul, camera **sare** cadrele vechi (nu se adună întârziere): imaginea rămâne actuală,
  doar cu mai puține cadre/s. Hotspot slab: adăugați `--video-fps 10` sau `--video-width 480` în `.bat`.

## Teste & reset (dashboard)
Cardul **Teste & reset** arată fiecare componentă separat, fără camere:
* **Reset:** alerte (ESP32 stins) · mașina (croazieră, fără avarii) · ML1 (călătorie nouă, calibrare de la zero) ·
  **Tot** (cere a doua apăsare).
* **Componente:** ESP32 1→2→3→0 (câte 2.5 s) · voce ML2 (o frază de test, cel mult o dată la 10 s) · **112 simulat**
  (cere a doua apăsare; raport marcat TEST, fereastra roșie, revenire la normal după 15 s; nimic nu pleacă de pe laptop).
* **ECU:** frânare de urgență, încetinire la 60 / 30, avarii pornite / oprite, anulează urgența, reia. Răspunsul
  plăcii apare în graficul mașinii, cu `id`-ul comenzii. Cât timp ML1 rulează, ML2 poate suprascrie o comandă
  manuală la următoarea schimbare a comenzii fuzzy.
* Doar cifre și puncte (privacy mode, ca în conceptul inițial): `--no-video` în `.bat`, sau `run_vigil.py --no-video`.
* Nimic nu se salvează (QoS 0, fără retain).

## Placa ECU (mașina simulată)
`ECU_Arduino/ECU_Arduino.ino` pe ESP32-C3: parola Wi-Fi stă în `ECU_Arduino/wifi_secrets.h` (copiat din
`wifi_secrets.h.example`, nu se urcă pe git). ML 2 îi trimite comenzile pe `vigil/car/cmd`; graficul vitezei din
dashboard arată ce cere ML 1 și ce face placa.
