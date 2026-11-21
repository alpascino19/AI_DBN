# Dashboard React Vigil — plan de implementare

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** dashboard React live pentru Vigil, conectat direct la Mosquitto prin WebSocket.

**Architecture:** o aplicație Vite fără backend. `mqtt.js` primește mesajele. Un reducer pur le transformă în stare, iar componentele desenează starea. Scheletul se desenează pe canvas, din coordonate normalizate 0..1.

**Tech Stack:** Vite, React 18, TypeScript, Tailwind CSS v4 (`@tailwindcss/vite`), `mqtt` (mqtt.js v5), vitest.

**Spec:** `docs/superpowers/specs/2026-11-21-dashboard-react-design.md`

## Global Constraints
- Fără backend. Contractele MQTT existente nu se schimbă.
- Topicuri ascultate: `vigil/vision`, `vigil/decision`, `vigil/driver_response`, `vigil/alert`, `vigil/ml1/status`.
- Singura publicare: `0` pe `vigil/alert`, cu retain, de la butonul „Reset alerte”.
- Broker WebSocket: `ws://<hostname-ul paginii>:9001`, suprascris de `?broker=ws://IP:9001`.
- Istoricul în memorie are maximum 50 de decizii și 50 de răspunsuri.
- Textele interfeței sunt în română.

## Review Focus
1. JSON invalid sau ne-obiect pe orice topic: ignorat, starea nu se schimbă (Task 2).
2. `vigil/vision` cu câmpuri `null` sau lipsă (camera oprită, fața nedetectată): afișează „—”, nu cade (Task 2, Task 4).
3. Decizie cu `severity: 0` (ML 1, „revenire la normal”): acceptată și afișată ca normal (Task 2).
4. Broker inaccesibil sau deconectat: indicatorul arată „deconectat”, iar aplicația se reconectează singură (Task 4).
5. `vigil/alert` cu conținut ne-numeric sau în afara 0–3: ignorat (Task 2).

---

### Task 1: Scaffold + broker WebSocket

**Files:**
- Create: `dashboard/` (Vite react-ts), `dashboard/vite.config.ts` (plugin Tailwind + configurare vitest), `dashboard/src/index.css` (`@import "tailwindcss";`)
- Modify: `ML2/mosquitto.conf`: se adaugă `listener 9001` și `protocol websockets`
- Delete: `Vision/dashboard/server.py`, `Vision/dashboard/static/index.html`

- [ ] `npm create vite@latest dashboard -- --template react-ts`, apoi `npm i mqtt` și `npm i -D tailwindcss @tailwindcss/vite vitest`. Se adaugă scriptul `"test": "vitest run"`.
- [ ] Verificare: `npm run build` trece, `npm test` rulează (0 teste).
- [ ] Commit: `dashboard: scaffold Vite + Tailwind + mqtt; mosquitto websockets 9001`

### Task 2: Reducerul de stare

**Files:**
- Create: `dashboard/src/state.ts`, `dashboard/src/state.test.ts`

**Interfaces — Produces:**
```ts
type Action = { tool: string; args: Record<string, unknown> }
type Decision = { severity: number; actions: Action[]; a?: number; reason?: string; source?: string; latency_ms?: number; receivedAt: number }
type DriverResponse = { answered: boolean; latency_sec: number | null; text: string | null; receivedAt: number }
type Vision = Record<string, unknown> & { landmarks?: number[][]; body_landmarks?: number[][]; events?: { event: string; duration_sec: number; confidence: number }[] }
type Status = { a: number; confidence: number; camera_on?: boolean; calibration?: Record<string, unknown> }
type State = { vision: Vision | null; status: Status | null; decision: Decision | null; decisions: Decision[]; responses: DriverResponse[]; alertLevel: number | null; emergency: { report: string; at: number } | null; lastMessageAt: number | null }
const initialState: State
function reduce(state: State, topic: string, payload: string, now: number): State
function dismissEmergency(state: State): State
const HISTORY = 50
```

- [ ] Teste (`state.test.ts`):
  - `vigil/vision` valid → `state.vision` e obiectul; `lastMessageAt === now`.
  - JSON invalid (`"{x"`) și `"[1,2]"` pe orice topic → starea întoarsă `toBe(state)`.
  - Decizie `{"severity":3,"actions":[{"tool":"call_112","args":{"report":"R"}}]}` → `decision.severity === 3`, `emergency.report === "R"`, `decisions.length === 1`.
  - Decizie `{"severity":0,"actions":[]}` → acceptată, `decision.severity === 0`.
  - Decizie fără `actions` sau cu `severity: 5` → ignorată.
  - 60 de decizii → `decisions.length === 50`, iar cea mai nouă e prima.
  - `vigil/driver_response` `{"answered":false,"latency_sec":null,"text":null}` → `responses[0].answered === false`.
  - `vigil/alert` `"2"` → `alertLevel === 2`; `"x"` și `"7"` → ignorate.
  - `vigil/ml1/status` `{"a":0.4,"confidence":0.9}` → `status.a === 0.4`.
  - `dismissEmergency` → `emergency === null`.
- [ ] Rulat `npm test`: pică (modulul lipsește).
- [ ] Implementat `state.ts`: imutabil, `JSON.parse` în try/catch, validează doar câmpurile folosite mai sus.
- [ ] `npm test` trece.
- [ ] Commit: `dashboard: reducer de stare + teste`

### Task 3: Geometria scheletului

**Files:**
- Create: `dashboard/src/skeleton.ts`, `dashboard/src/skeleton.test.ts`

**Interfaces — Produces:**
```ts
const POSE_CONNECTIONS: [number, number][]   // MediaPipe Pose 0..24: față 0-10, umeri 11-12, brațe 11-13-15, 12-14-16, mâini 15-17/19/21, 16-18/20/22, trunchi 11-23, 12-24, 23-24
function toCanvas(points: unknown, w: number, h: number): [number, number][]   // puncte valide 0..1 → pixeli; ignoră intrările invalide
function drawSkeleton(ctx: CanvasRenderingContext2D, face: unknown, body: unknown, w: number, h: number): void
```

- [ ] Teste:
  - `toCanvas([[0.5,0.25]], 200, 100)` → `[[100,25]]`;
  - `toCanvas(null,...)` → `[]`;
  - `toCanvas([[1],"x",[0.1,0.2]],10,10)` → `[[1,2]]`;
  - orice pereche din `POSE_CONNECTIONS` are indici < 25.
- [ ] `npm test` pică → implementare → `npm test` trece. `drawSkeleton` desenează puncte pentru față, linii și puncte pentru corp, și sare peste conexiunile cu puncte lipsă.
- [ ] Commit: `dashboard: geometria scheletului + teste`

### Task 4: Conexiunea MQTT și interfața

**Files:**
- Create: `dashboard/src/useVigil.ts`, `dashboard/src/components/{Header,AttentionGauge,CameraPanel,Signals,Events,DecisionCard,Responses,EmergencyOverlay}.tsx`
- Modify: `dashboard/src/App.tsx`, `dashboard/src/main.tsx`

**Interfaces:**
- Consumes: `reduce`, `initialState`, `dismissEmergency` (Task 2); `drawSkeleton` (Task 3).
- Produces: `useVigil(): { state: State; connected: boolean; broker: string; resetAlerts(): void; dismiss(): void }`.

- [ ] `useVigil`:
  - `mqtt.connect(broker, { reconnectPeriod: 2000 })`;
  - abonare la cele 5 topicuri;
  - `useReducer` peste `reduce`;
  - `connected` din evenimentele `connect`, `close` și `offline`;
  - `resetAlerts` publică `"0"` pe `vigil/alert` cu `{ retain: true, qos: 1 }`.
- [ ] Componentele, după schița din spec:
  - `CameraPanel({ title, face, body })` desenează pe canvas la fiecare actualizare de `vision`, cu textul „imagine: în curând”;
  - `Signals` afișează „—” pentru valori `null` sau lipsă;
  - `DecisionCard` are culori pe severitate (0 verde, 1 galben, 2 portocaliu, 3 roșu) și istoricul pliabil;
  - `EmergencyOverlay` apare când `emergency != null` și are butonul „Închide”.
- [ ] Verificare manuală: `npm run dev` + broker cu WebSocket + `python -m vigil_vision.fake.fake_vision` (Vision) + `python ML2/fake_llm.py`. Trebuie să se vadă scheletul, semnalele, deciziile, overlay-ul la `fara_raspuns`, iar „Reset alerte” stinge LED-ul ESP32.
- [ ] `npm run build` trece.
- [ ] Commit: `dashboard: interfața live`

### Task 5: Documentație
**Files:** Create `dashboard/README.md`; Modify `CLAUDE.md` (dashboard = `dashboard/`, React, port 9001).
- [ ] README: instalare (`npm i`), pornire, linia de WebSocket din `mosquitto.conf`, parametrul `?broker=`, testare cu datele false.
- [ ] Commit: `dashboard: README + CLAUDE.md`
