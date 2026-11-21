// Starea dashboard-ului, din mesajele MQTT. Funcții pure: (stare, topic, mesaj) -> stare nouă.
// Mesajele stricate sau incomplete se ignoră (întorc exact aceeași stare), ca dashboard-ul să nu cadă la demo.

export type Action = { tool: string; args: Record<string, unknown> }
export type Decision = {
  severity: number
  actions: Action[]
  a?: number
  reason?: string
  source?: string
  latency_ms?: number
  receivedAt: number
}
export type DriverResponse = { answered: boolean; latency_sec: number | null; text: string | null; receivedAt: number }
export type VisionEvent = { event: string; duration_sec: number; confidence: number }
export type Vision = Record<string, unknown> & {
  landmarks?: number[][]
  body_landmarks?: number[][]
  events?: VisionEvent[]
}
export type Status = { a: number; confidence: number; camera_on?: boolean; calibration?: Record<string, unknown> }
/** Comanda fuzzy a ML1 pentru ECU (vigil/ecu, 5 Hz). */
export type Ecu = {
  mode: 'NORMAL' | 'CAUTION' | 'ASSIST' | 'MRM' | string
  speed_factor: number
  speed_ref_kmh: number
  target_kmh?: number
  decel_mps2?: number
  lka_authority: number
  headway_s: number
  hazards?: boolean
  lane?: string
  state?: { p_state?: number[]; p_risk?: number; expected_severity?: number }
  inputs?: { a?: number; risk?: number; confidence?: number; severity?: number }
}
export type AssistantRequest = 'listen' | 'time' | 'weather' | 'traffic'
/** Starea asistentului Virgil (vigil/assistant/reply, de la ML2). */
export type Assistant = {
  request: string
  status: 'listening' | 'working' | 'speaking' | 'done' | 'interrupted' | string
  heard?: string
  text?: string
  source?: string
  receivedAt: number
}
/** Mașina simulată pe ESP32 (ECU_Arduino): vigil/car/state, la 0.5 s cât frânează și la 5 s altfel. */
export type Car = {
  speed_kmh: number
  target_kmh?: number
  braking?: boolean
  hazard?: boolean
  stopped?: boolean
  emergency?: boolean
  alert?: number
  receivedAt: number
}
/** vigil/car/event: emergency_braking_started, stopped, target_reached, hazard_on, reset, error... */
export type CarEvent = { event: string; speed_kmh?: number; id?: string; detail?: string; at: number }
/** Un punct din graficul vitezei; ref/target vin de la ML1 (vigil/ecu), speed de la ECU (vigil/car/state). */
export type EcuSample = { t: number; ref: number; target: number | null; mode: string; hazards: boolean; a: number | null }
export type CarSample = { t: number; speed: number; braking: boolean }
export type State = {
  vision: Vision | null
  status: Status | null
  ecu: Ecu | null
  car: Car | null
  carEvents: CarEvent[]
  /** placa ECU: "online" / "offline" (vigil/esp32/status, reținut + last will); null = nu știm încă */
  ecuOnline: boolean | null
  series: { ecu: EcuSample[]; car: CarSample[] }
  assistant: Assistant | null
  decision: Decision | null
  decisions: Decision[]
  responses: DriverResponse[]
  alertLevel: number | null
  emergency: { report: string; at: number } | null
  lastMessageAt: number | null
}

export const HISTORY = 50
export const SERIES_MS = 120_000 // graficul arată ultimele 2 minute

export const initialState: State = {
  vision: null,
  status: null,
  ecu: null,
  car: null,
  carEvents: [],
  ecuOnline: null,
  series: { ecu: [], car: [] },
  assistant: null,
  decision: null,
  decisions: [],
  responses: [],
  alertLevel: null,
  emergency: null,
  lastMessageAt: null,
}

/** Adaugă un punct și taie tot ce e mai vechi de SERIES_MS. */
function push<T extends { t: number }>(list: T[], item: T): T[] {
  const from = item.t - SERIES_MS
  let i = 0
  while (i < list.length && list[i].t < from) i++
  return [...list.slice(i), item]
}

function parseObject(payload: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(payload)
    return v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null
  } catch {
    return null
  }
}

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

export function reduce(state: State, topic: string, payload: string, now: number): State {
  if (topic === 'vigil/alert') {
    // exact o cifră 0-3: un mesaj gol (ștergerea stării reținute) nu înseamnă nivelul 0
    const text = payload.trim()
    return /^[0-3]$/.test(text) ? { ...state, alertLevel: Number(text), lastMessageAt: now } : state
  }
  if (topic === 'vigil/esp32/status') {
    const text = payload.trim()
    return text === 'online' || text === 'offline' ? { ...state, ecuOnline: text === 'online' } : state
  }
  const msg = parseObject(payload)
  if (!msg) return state
  const seen = { ...state, lastMessageAt: now }

  switch (topic) {
    case 'vigil/vision':
      return { ...seen, vision: msg as Vision }
    case 'vigil/ml1/status':
      return isNum(msg.a) && isNum(msg.confidence) ? { ...seen, status: msg as Status } : state
    case 'vigil/assistant/reply': {
      if (typeof msg.request !== 'string' || typeof msg.status !== 'string') return state
      const s = (k: string) => (typeof msg[k] === 'string' ? (msg[k] as string) : undefined)
      // „working” / „speaking” fără text păstrează întrebarea auzită din pasul anterior
      const prev = state.assistant?.request === msg.request ? state.assistant : null
      return {
        ...seen,
        assistant: {
          request: msg.request,
          status: msg.status,
          heard: s('heard') ?? (msg.status === 'listening' ? undefined : prev?.heard),
          text: s('text'),
          source: s('source'),
          receivedAt: now,
        },
      }
    }
    case 'vigil/ecu': {
      if (!(typeof msg.mode === 'string' && isNum(msg.speed_factor) && isNum(msg.speed_ref_kmh) &&
        isNum(msg.lka_authority) && isNum(msg.headway_s))) return state
      const ecu = msg as Ecu
      const a = ecu.inputs?.a
      const sample: EcuSample = {
        t: now, ref: ecu.speed_ref_kmh, target: isNum(ecu.target_kmh) ? ecu.target_kmh : null,
        mode: ecu.mode, hazards: ecu.hazards === true, a: isNum(a) ? a : null,
      }
      return { ...seen, ecu, series: { ...state.series, ecu: push(state.series.ecu, sample) } }
    }
    case 'vigil/car/state':
    case 'vigil/car/event': {
      // evenimentele au și ele viteza curentă: intră în grafic la fel ca starea periodică
      if (!isNum(msg.speed_kmh)) return state
      const b = (k: string) => (typeof msg[k] === 'boolean' ? (msg[k] as boolean) : undefined)
      const car: Car = {
        ...(state.car ?? {}),
        speed_kmh: msg.speed_kmh,
        ...(isNum(msg.target_kmh) ? { target_kmh: msg.target_kmh } : {}),
        ...Object.fromEntries(['braking', 'hazard', 'stopped', 'emergency']
          .filter((k) => b(k) !== undefined).map((k) => [k, b(k)])),
        ...(isNum(msg.alert) ? { alert: msg.alert } : {}),
        receivedAt: now,
      }
      const carEvents = topic === 'vigil/car/event' && typeof msg.event === 'string'
        ? [{ event: msg.event, speed_kmh: msg.speed_kmh, at: now,
          ...(typeof msg.id === 'string' ? { id: msg.id } : {}),
          ...(typeof msg.detail === 'string' ? { detail: msg.detail } : {}) }, ...state.carEvents].slice(0, 8)
        : state.carEvents
      return {
        ...seen, car, carEvents, ecuOnline: true,
        series: { ...state.series, car: push(state.series.car, { t: now, speed: msg.speed_kmh, braking: car.braking === true }) },
      }
    }
    case 'vigil/driver_response':
      if (typeof msg.answered !== 'boolean') return state
      return {
        ...seen,
        responses: [
          {
            answered: msg.answered,
            latency_sec: isNum(msg.latency_sec) ? msg.latency_sec : null,
            text: typeof msg.text === 'string' ? msg.text : null,
            receivedAt: now,
          },
          ...state.responses,
        ].slice(0, HISTORY),
      }
    case 'vigil/decision': {
      const sev = msg.severity
      if (!Number.isInteger(sev) || (sev as number) < 0 || (sev as number) > 3 || !Array.isArray(msg.actions)) return state
      const actions = (msg.actions as unknown[]).filter(
        (a): a is Action => !!a && typeof a === 'object' && typeof (a as Action).tool === 'string',
      )
      const decision: Decision = {
        severity: sev as number,
        actions,
        a: isNum(msg.a) ? msg.a : undefined,
        reason: typeof msg.reason === 'string' ? msg.reason : undefined,
        source: typeof msg.source === 'string' ? msg.source : undefined,
        latency_ms: isNum(msg.latency_ms) ? msg.latency_ms : undefined,
        receivedAt: now,
      }
      const call = actions.find((a) => a.tool === 'call_112')
      const report = call && typeof call.args?.report === 'string' ? call.args.report : null
      return {
        ...seen,
        decision,
        decisions: [decision, ...state.decisions].slice(0, HISTORY),
        emergency: report !== null ? { report, at: now } : state.emergency,
      }
    }
    default:
      return state
  }
}

export function dismissEmergency(state: State): State {
  return { ...state, emergency: null }
}
