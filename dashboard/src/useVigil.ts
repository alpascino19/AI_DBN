import mqtt from 'mqtt'
import { useEffect, useReducer, useRef, useState } from 'react'
import { dismissEmergency, initialState, reduce, type AssistantRequest, type State } from './state'

const TOPICS = ['vigil/vision', 'vigil/decision', 'vigil/driver_response', 'vigil/alert', 'vigil/ml1/status', 'vigil/ecu',
  'vigil/assistant/reply', 'vigil/car/state', 'vigil/car/event', 'vigil/esp32/status']
// imaginea camerelor (Vision cu --video): JPEG binar, nu JSON; nu trece prin reducer
const VIDEO_TOPICS: Record<string, 'face' | 'body'> = { 'vigil/video/face': 'face', 'vigil/video/body': 'body' }
const VIDEO_STALE_MS = 3000

// brokerul: ?broker=ws://IP:9001, altfel același calculator care servește pagina
export const BROKER =
  new URLSearchParams(location.search).get('broker') ?? `ws://${location.hostname || 'localhost'}:9001`

type Msg = { kind: 'mqtt'; topic: string; payload: string; now: number } | { kind: 'dismiss' }
/** Ultimul cadru al fiecărei camere: URL de obiect (blob:) + momentul sosirii. */
export type VideoFrames = Partial<Record<'face' | 'body', { url: string; at: number }>>

function reducer(state: State, msg: Msg): State {
  return msg.kind === 'dismiss' ? dismissEmergency(state) : reduce(state, msg.topic, msg.payload, msg.now)
}

export function useVigil() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [connected, setConnected] = useState(false)
  const [video, setVideo] = useState<VideoFrames>({})
  const client = useRef<mqtt.MqttClient | null>(null)

  useEffect(() => {
    const c = mqtt.connect(BROKER, { reconnectPeriod: 2000, connectTimeout: 4000 })
    client.current = c
    const urls: Partial<Record<'face' | 'body', string>> = {}
    const stale: Partial<Record<'face' | 'body', ReturnType<typeof setTimeout>>> = {}
    c.on('connect', () => {
      setConnected(true)
      c.subscribe([...TOPICS, ...Object.keys(VIDEO_TOPICS)])
    })
    c.on('close', () => setConnected(false))
    c.on('offline', () => setConnected(false))
    c.on('message', (topic, payload) => {
      const cam = VIDEO_TOPICS[topic]
      if (cam) {
        const url = URL.createObjectURL(new Blob([payload as BlobPart], { type: 'image/jpeg' }))
        const old = urls[cam]
        urls[cam] = url
        setVideo((v) => ({ ...v, [cam]: { url, at: Date.now() } }))
        // cadrul vechi se eliberează după ce <img> a trecut la cel nou (altfel memoria crește la 5 cadre/s)
        if (old) setTimeout(() => URL.revokeObjectURL(old), 1000)
        // fără cadru nou VIDEO_STALE_MS: camera s-a oprit (sau a pierdut Wi-Fi) -> fereastra revine la puncte
        clearTimeout(stale[cam])
        stale[cam] = setTimeout(() => setVideo(({ [cam]: _gone, ...rest }) => rest), VIDEO_STALE_MS)
        return
      }
      dispatch({ kind: 'mqtt', topic, payload: payload.toString(), now: Date.now() })
    })
    return () => {
      c.end(true)
      for (const u of Object.values(urls)) if (u) URL.revokeObjectURL(u)
      for (const h of Object.values(stale)) clearTimeout(h)
    }
  }, [])

  return {
    state,
    video,
    connected,
    broker: BROKER,
    resetAlerts: () => client.current?.publish('vigil/alert', '0', { qos: 1, retain: true }),
    dismiss: () => dispatch({ kind: 'dismiss' }),
    // butoanele asistentului: ML2 (executor.py) răspunde cu voce și pe vigil/assistant/reply
    ask: (request: AssistantRequest) =>
      client.current?.publish('vigil/assistant', JSON.stringify({ request, ts: Date.now() / 1000 }), { qos: 1 }),
    // repornirea demo-ului pe placa ECU (viteza de croazieră, fără avarii); ML2 face la fel la pornire
    resetCar: () => client.current?.publish('vigil/car/cmd', JSON.stringify({ cmd: 'reset', id: 'dashboard' }), { qos: 1 }),
  }
}
