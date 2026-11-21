import mqtt from 'mqtt'
import { useEffect, useReducer, useRef, useState } from 'react'
import { dismissEmergency, initialState, reduce, type AssistantRequest, type State } from './state'
import { VideoStore, type Cam } from './videoStore'

const TOPICS = ['vigil/vision', 'vigil/decision', 'vigil/driver_response', 'vigil/alert', 'vigil/ml1/status', 'vigil/ecu',
  'vigil/assistant/reply', 'vigil/car/state', 'vigil/car/event', 'vigil/esp32/status']
// imaginea camerelor: JPEG binar, nu JSON; merge în VideoStore, nu prin reducer
const VIDEO_TOPICS: Record<string, Cam> = { 'vigil/video/face': 'face', 'vigil/video/body': 'body' }

// brokerul: ?broker=ws://IP:9001, altfel același calculator care servește pagina
export const BROKER =
  new URLSearchParams(location.search).get('broker') ?? `ws://${location.hostname || 'localhost'}:9001`

type Msg = { kind: 'mqtt'; topic: string; payload: string; now: number } | { kind: 'dismiss' }

function reducer(state: State, msg: Msg): State {
  return msg.kind === 'dismiss' ? dismissEmergency(state) : reduce(state, msg.topic, msg.payload, msg.now)
}

export function useVigil() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [connected, setConnected] = useState(false)
  const [video] = useState(() => new VideoStore())
  const client = useRef<mqtt.MqttClient | null>(null)

  useEffect(() => {
    const c = mqtt.connect(BROKER, { reconnectPeriod: 2000, connectTimeout: 4000 })
    client.current = c
    c.on('connect', () => {
      setConnected(true)
      c.subscribe([...TOPICS, ...Object.keys(VIDEO_TOPICS)])
    })
    c.on('close', () => setConnected(false))
    c.on('offline', () => setConnected(false))
    c.on('message', (topic, payload) => {
      const cam = VIDEO_TOPICS[topic]
      if (cam) video.push(cam, payload)
      else dispatch({ kind: 'mqtt', topic, payload: payload.toString(), now: Date.now() })
    })
    return () => {
      c.end(true)
      video.dispose()
    }
  }, [video])

  /** Publicare generică (panoul de teste): obiectele pleacă ca JSON, textul ca atare. */
  const publish = (topic: string, payload: string | object, opts: { qos?: 0 | 1; retain?: boolean } = {}) =>
    client.current?.publish(topic, typeof payload === 'string' ? payload : JSON.stringify(payload),
      { qos: opts.qos ?? 1, retain: opts.retain ?? false })

  return {
    state,
    video,
    connected,
    broker: BROKER,
    publish,
    resetAlerts: () => publish('vigil/alert', '0', { retain: true }),
    dismiss: () => dispatch({ kind: 'dismiss' }),
    // butoanele asistentului: ML2 (executor.py) răspunde cu voce și pe vigil/assistant/reply
    ask: (request: AssistantRequest) => publish('vigil/assistant', { request, ts: Date.now() / 1000 }),
    // repornirea demo-ului pe placa ECU (viteza de croazieră, fără avarii); ML2 face la fel la pornire
    resetCar: () => publish('vigil/car/cmd', { cmd: 'reset', id: 'dashboard' }),
  }
}
