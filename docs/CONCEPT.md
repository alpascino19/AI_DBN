# Vigil — Sistem Real-Time de Monitorizare a Siguranței și Sănătății Șoferului

## 📌 1. CONTEXTUL PRINCIPAL ȘI VIZIUNEA PROIECTULUI

Proiectul este axat pe construirea unui sistem inteligent de monitorizare în timp real a siguranței și sănătății utilizatorului/șoferului, combinând Computer Vision (CV) și un Model de Limbaj de Mari Dimensiuni (LLM). Folosind o cameră web standard sau o cameră instalată în cabină, sistemul monitorizează continuu o persoană pentru a detecta semne timpurii ale unei urgențe medicale (cum ar fi stopul cardiac), precum și semne de oboseală și distragere a atenției.

Când sunt detectați declanșatori vizuali specifici, pipeline-ul de viziune transmite date structurate despre evenimente către un LLM open-source. Acesta acționează ca un motor de decizie pentru a evalua severitatea situației și a executa intervențiile sau răspunsurile corespunzătoare.

---

### Arhitectura pe Module

#### Modulul A: Detecție prin Computer Vision (Python)
Un pipeline de procesare video în timp real care urmărește reperele faciale, postura corporală și mișcările mâinilor pentru a identifica două categorii principale de declanșatori:

1. **Categorie 1: Urgențe Medicale și Simptome de Stop Cardiac**
   * **Prinderea pieptului (Chest Clutching):** Detectarea situațiilor în care utilizatorul își duce mâna/mâinile în zona inimii/pieptului (indicativ pentru dureri toracice sau angină).
   * **Clipit anormal (Hyper-blinking):** Detectarea unui ritm rapid, erratic sau excesiv de clipire.
   * **Respirație grea / Gasping agonal:** Urmărirea mișcărilor rapide de ridicare și coborâre a pieptului/umerilor sau a tiparelor de deschidere distressed a gurii.
   * **Pierderea cunoștinței:** Detectarea căderii bruște a capului, închiderea prelungită a ochilor și lipsa tonusului postural sau a capacității de răspuns.

2. **Categorie 2: Monitorizarea Distragerii și Oboselii Șoferului**
   * **Somnolență / Micro-somn:** Urmărirea EAR (Eye Aspect Ratio) pentru a detecta clipiri prea rare (privit în gol), închiderea prelungită a ochilor sau rate lent severe ale clipitului.
   * **Privire distrasă (Nu se uită la drum):** Estimarea poziției capului (pitch, yaw, roll) și urmărirea privirii (eye-gaze) pentru a determina când atenția utilizatorului este mutată de la drum.
   * **Utilizarea telefonului:** Detecția obiectelor și corelarea mâinii/privirii pentru a identifica dacă utilizatorul ține sau se uită la un telefon mobil.

---

#### Modulul B: Motorul de Decizie și Acțiune LLM
În loc să se bazeze exclusiv pe reguli rigide de tip `if/else`, sistemul transmite declanșatorii detectați (împreună cu durata și scorul de încredere) către un **LLM open-source** (de exemplu Llama 3, Mistral sau Phi-3 rulat local via Ollama, vLLM sau Hugging Face).

* **Intrare LLM:** Loguri structurate JSON sau text care reprezintă declanșatorii activi și contextul lor (de exemplu: `{"event": "hand_on_chest", "duration_sec": 4, "breathing": "heavy", "gaze": "down"}`).
* **Rolul LLM-ului:** 
  1. **Raționament contextual:** Diferențierea între o mică distragere (de exemplu o privire de 1 secundă spre telefon) și o urgență critică (de exemplu prindere a pieptului + respirație grea + cap căzut).
  2. **Selecția acțiunii:** Decizia asupra nivelului de intervenție în funcție de severitate.
* **Acțiuni Executate de Sistem:**
  * *Severitate Redusă (Distragere):* Emite un semnal sonor blând sau un memento vocal („Vă rugăm să păstrați privirea la drum”).
  * *Severitate Medie (Oboseală):* Declanșează o alertă mai puternică, sugerează o pauză sau angajează șoferul într-o scurtă conversație vocală pentru a-l menține alert.
  * *Severitate Ridicată (Urgență Medicală / Lipsă Cunoștință):* Declanșează o alarmă urgentă, simulează sau inițiază un apel automat de urgență (112) cu coordonate GPS, deblochează ușile vehiculului sau activează semnalele de avarie și protocoalele de oprire pe dreapta.

---

### Stiva Tehnologică Nucleu
* **Limbaj:** Python 3.10+
* **Computer Vision & Pose Estimation:** OpenCV, MediaPipe (Face Mesh, Pose, Hands) sau YOLOv8 (pentru detecția telefonului/obiectelor).
* **Integrări LLM:** Ollama / Hugging Face `transformers` / LangChain (pentru apelarea structurată a funcțiilor și selecția acțiunilor).
* **Audio / Alerte:** `pyttsx3` sau echivalent OpenAI Whisper / TTS pentru interacțiunea vocală.

---

## ⚙️ 2. CONTROLUL VEHICULULUI ȘI IMPLEMENTAREA SISTEMULUI VIGIL

### Conceptul de Conducere Adaptivă și Cadru Legal
Vigil extinde motorul de decizie prin integrarea cu un controler MPC (Model Predictive Control) care împarte autoritatea de conducere (pe autostradă, semi-autonom Level 3) între șofer și mașină în funcție de scorul de atenție $a \in [0, 1]$:

* **Șofer atent ($a \approx 1$):** Controlul trece treptat la el, iar MPC-ul păstrează doar limitele de siguranță.
* **Șofer obosit sau distras ($a \in (0, 1)$):** MPC-ul conduce mai prudent (viteză mai mică, distanță mai mare) și escaladează avertizările.
* **Șofer care nu reacționează ($a = 0$):** Manevră de risc minim până la oprirea pe banda de urgență, plus un raport generat automat pentru 112.

*Context legal și de piață:*
Neatenția la volan este o cauză majoră de accidente. Din 7 iulie 2026, sistemele ADDW (Advanced Driver Distraction Warning) sunt obligatorii în toate vehiculele noi din UE conform General Safety Regulation. În timp ce sistemele DMS existente doar observă, iar Emergency Assist intervine doar în ultimul moment, Vigil adaptează condusul continuu și treptat între aceste două extreme.

---

### Detecția Fuzionată și Confidențialitatea (Privacy Mode)

Patru semnale se combină într-un scor de atenție de la 0 la 1, iar sistemul funcționează și cu camera oprită (folosind doar semnale indirecte):

| Semnal | Sursă | Ce detectează | Cât de sigur |
| --- | --- | --- | --- |
| Ochi, privire, poziția capului | Camera laptopului + MediaPipe Face Mesh (EAR, PERCLOS, head pose) | Somnolență, privit în afara drumului, capul căzut | Ridicat |
| Telefon în mână | Cameră + YOLO (clasa „cell phone”) | Distragere | Ridicat |
| Timp de reacție vocal | Microfonul laptopului + speech-to-text | Capacitatea de a răspunde | Mediu |
| Comportament la volan | Inputul din simulare (tastatură sau gamepad) | Lipsa corecțiilor, ieșiri de pe bandă | Mediu |

* **Privacy mode:** Niciun cadru video nu este salvat sau trimis în rețea. Din cameră ies doar cifre, iar pe dashboard se afișează scheletul de puncte al feței, nu imaginea reală.
* **Fuziune:** Scorul este o medie ponderată a semnalelor disponibile, netezită în timp. Calibrarea din primele 30 de secunde învață parametrii normali ai șoferului. Dezactivarea camerei scade încrederea generală, iar MPC-ul devine automat mai prudent.

---

### Controlul Predictiv (MPC)

Un singur MPC conduce în toate stările. Scorul de atenție $a$ schimbă doar ponderile și constrângerile acestuia, garantând tranziții line fără comutări bruște între controlere.

#### Funcția de cost cu shared control:
$$J = \sum_{k=0}^{N-1} \left[ (1-a)\,\|x_k - x_{ref}\|_Q^2 + a\,\|u_k - u_{driver,k}\|_W^2 + \|u_k\|_R^2 \right]$$

La $a = 1$, MPC-ul urmează comanda șoferului și intervine doar la limitele de siguranță. La $a = 0$, conduce singur spre referința sigură.

#### Constrângeri adaptate la stare:
| Stare | Viteză maximă | Distanță minimă față de mașina din față | Accelerație permisă |
| --- | --- | --- | --- |
| Atent | limita drumului | 2 s | normală |
| Obosit sau distras | limita − 20 km/h | 3 s | redusă |
| Urgență | spre 0 | 3 s | doar frânare lină |

* **Manevra de risc minim:** Referința laterală se mută pe banda de urgență, referința de viteză coboară la 0 cu decelerație limitată, se aprind avariile și se trimite raportul generat pentru 112.
* **Implementare:** `cvxpy` + `OSQP` pentru MPC-ul liniar (sau `CasADi` pentru modelul neliniar). Orizont de aproximativ 2 secunde (20 de pași la 10 Hz). Simulare în `highway-env`.

---

## 📊 3. ARHITECTURA HARDWARE, DASHBOARD ȘI SCENARIU DEMO

### Arhitectura Hardware & Comunicare
* Tot calculul (viziune, voce, fuziune, MPC, simulare, dashboard) rulează local pe **laptop**.
* Placa **ESP32** primește treapta de alertă prin protocolul **MQTT** (pe rețeaua locală) și acționează un LED sau buzzer hardware pentru a demonstra interacțiunea cu hardware-ul real.

---

### Dashboard Live și Funcții GenAI
* **Dashboard Live (FastAPI + Web Frontend / Streamlit):**
  * Starea curentă a sistemului și treapta de escaladare.
  * Scorul de atenție în timp real și contribuția fiecărui semnal.
  * Scheletul feței (Privacy mode).
  * Ponderea de control între șofer și MPC.
  * Viteza reală comparată cu referința și traiectoria prezisă pe orizontul MPC.
* **Funcții GenAI:**
  * **Verificarea vocală:** Asistentul pune o întrebare scurtă, iar timpul și coerența răspunsului intră în scorul de atenție.
  * **Raportul pentru 112:** Generat automat în caz de urgență medicală sau lipsă de reacție, conținând locația, semnele observate și durata de inacțiune.
  * **Raportul post-cursă:** Analiza momentelor de oboseală și recomandări de pauze.

---

### Scenariul de Demo (3 Minute)

1. **Calibrare (30s):** Șoferul stă în fața camerei, iar sistemul își stabilizează scorul de atenție.
2. **Conducere Autonomă:** Mașina conduce în `highway-env`, iar șoferul este atent.
3. **Predare Reușită:** Apare cererea de preluare, șoferul se uită la drum și răspunde vocal. Controlul trece treptat la șofer.
4. **Distragere:** Șoferul ridică telefonul sau își mută privirea. Sistemul avertizează, iar MPC-ul reduce viteza și mărește distanța.
5. **Somnolență / Prindere Piept:** Detectare clipit excesiv, ochii închiși sau mână pe piept. Escaladare avertizări vizuale și sonore, apoi MPC preia controlul.
6. **Urgență Medicală / Pierdere Cunoștință:** Capul cade, răspunsul vocal lipsește. Mașina efectuează manevra de risc minim (tragere pe dreapta și oprire), iar raportul 112 este afișat.
7. **Privacy Mode:** Camera se oprește manual, iar sistemul continuă monitorizarea pe baza semnalelor indirecte.

---

## 🚀 4. PLAN DE LUCRU ȘI RISCURI

### Împărțire în Echipă (3–4 persoane)
* **Viziune:** MediaPipe, YOLO, calibrarea pragurilor, detecție prindere piept / ochi / telefon.
* **Control:** MPC, logică de stări, integrare `highway-env`.
* **Embedded + GenAI:** ESP32 (MQTT), dialog vocal, generare raport 112.
* **Dashboard + Pitch:** Frontend în timp real, prezentare, scenariu de demo.

### Risc & Plan B
* *MPC lent/neconvergent:* Trecere la MPC liniar simplu cu orizont mai scurt sau controler proporțional (P) de rezervă.
* *Probleme de iluminare:* Calibrare rapidă la fața locului și ajustare din dashboard.
* *Cameră acoperită/oprită:* Trecere pe modul de încredere scăzută (voce și stil de condus).
* *Latență GenAI:* Folosire șabloane locale predefinite pentru raportul 112.