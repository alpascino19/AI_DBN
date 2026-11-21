import mqtt from 'mqtt'
import { useEffect, useReducer, useRef, useState } from 'react'
import { dismissEmergency, initialState, reduce, type AssistantRequest, type State } from './state'

const TOPICS = ['vigil/vision', 'vigil/decision', 'vigil/driver_response', 'vigil/alert', 'vigil/ml1/status', 'vigil/ecu',
  'vigil/assistant/reply']

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
  const client = useRef<mqtt.MqttClient | null>(null)

  useEffect(() => {
    const c = mqtt.connect(BROKER, { reconnectPeriod: 2000, connectTimeout: 4000 })
    client.current = c
    c.on('connect', () => {
      setConnected(true)
      c.subscribe(TOPICS)
    })
    c.on('close', () => setConnected(false))
    c.on('offline', () => setConnected(false))
    c.on('message', (topic, payload) => dispatch({ kind: 'mqtt', topic, payload: payload.toString(), now: Date.now() }))
    return () => {
      c.end(true)
    }
  }, [])

  return {
    state,
    connected,
    broker: BROKER,
    resetAlerts: () => client.current?.publish('vigil/alert', '0', { qos: 1, retain: true }),
    dismiss: () => dispatch({ kind: 'dismiss' }),
    // butoanele asistentului: ML2 (executor.py) răspunde cu voce și pe vigil/assistant/reply
    ask: (request: AssistantRequest) =>
      client.current?.publish('vigil/assistant', JSON.stringify({ request, ts: Date.now() / 1000 }), { qos: 1 }),
  }
}
