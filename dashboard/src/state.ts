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
export type State = {
  vision: Vision | null
  status: Status | null
  ecu: Ecu | null
  assistant: Assistant | null
  decision: Decision | null
  decisions: Decision[]
  responses: DriverResponse[]
  alertLevel: number | null
  emergency: { report: string; at: number } | null
  lastMessageAt: number | null
}

export const HISTORY = 50

export const initialState: State = {
  vision: null,
  status: null,
  ecu: null,
  assistant: null,
  decision: null,
  decisions: [],
  responses: [],
  alertLevel: null,
  emergency: null,
  lastMessageAt: null,
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
    case 'vigil/ecu':
      return typeof msg.mode === 'string' && isNum(msg.speed_factor) && isNum(msg.speed_ref_kmh) &&
        isNum(msg.lka_authority) && isNum(msg.headway_s)
        ? { ...seen, ecu: msg as Ecu }
        : state
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
