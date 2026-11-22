import { describe, expect, it } from 'vitest'
import { HISTORY, dismissEmergency, initialState, reduce } from './state'

const T = 1000
const decision = (d: object) => reduce(initialState, 'vigil/decision', JSON.stringify(d), T)

describe('reduce', () => {
  it('vigil/vision valid -> vision + lastMessageAt', () => {
    const s = reduce(initialState, 'vigil/vision', '{"ear":0.21,"landmarks":[[0.1,0.2]]}', T)
    expect(s.vision?.ear).toBe(0.21)
    expect(s.lastMessageAt).toBe(T)
  })

  it('JSON invalid sau ne-obiect -> starea neschimbată', () => {
    for (const topic of ['vigil/vision', 'vigil/decision', 'vigil/driver_response', 'vigil/ml1/status']) {
      expect(reduce(initialState, topic, '{x', T)).toBe(initialState)
      expect(reduce(initialState, topic, '[1,2]', T)).toBe(initialState)
    }
  })

  it('decizie cu call_112 -> urgență cu raportul', () => {
    const s = decision({ severity: 3, actions: [{ tool: 'call_112', args: { report: 'R' } }] })
    expect(s.decision?.severity).toBe(3)
    expect(s.emergency?.report).toBe('R')
    expect(s.decisions).toHaveLength(1)
  })

  it('severity 0 (revenire la normal) e acceptată', () => {
    expect(decision({ severity: 0, actions: [] }).decision?.severity).toBe(0)
  })

  it('decizie fără actions sau cu severitate invalidă -> ignorată', () => {
    expect(decision({ severity: 2 })).toBe(initialState)
    expect(decision({ severity: 5, actions: [] })).toBe(initialState)
  })

  it(`istoricul păstrează ultimele ${HISTORY}, cea mai nouă prima`, () => {
    let s = initialState
    for (let i = 0; i < 60; i++) s = reduce(s, 'vigil/decision', JSON.stringify({ severity: 1, actions: [], reason: `r${i}` }), i)
    expect(s.decisions).toHaveLength(HISTORY)
    expect(s.decisions[0].reason).toBe('r59')
  })

  it('vigil/driver_response', () => {
    const s = reduce(initialState, 'vigil/driver_response', '{"answered":false,"latency_sec":null,"text":null}', T)
    expect(s.responses[0].answered).toBe(false)
  })

  it('vigil/alert 0-3; restul ignorat', () => {
    expect(reduce(initialState, 'vigil/alert', '2', T).alertLevel).toBe(2)
    expect(reduce(initialState, 'vigil/alert', 'x', T)).toBe(initialState)
    expect(reduce(initialState, 'vigil/alert', '7', T)).toBe(initialState)
    // mesaj gol = ștergerea stării reținute în MQTT, nu nivelul 0
    expect(reduce(initialState, 'vigil/alert', '', T)).toBe(initialState)
    expect(reduce(initialState, 'vigil/alert', '1.5', T)).toBe(initialState)
  })

  it('vigil/assistant/reply: întrebarea auzită rămâne până la răspuns', () => {
    const msg = (o: object) => JSON.stringify({ request: 'listen', ...o })
    let s = reduce(initialState, 'vigil/assistant/reply', msg({ status: 'listening' }), T)
    expect(s.assistant?.status).toBe('listening')
    s = reduce(s, 'vigil/assistant/reply', msg({ status: 'working', heard: 'Cum e vremea?' }), T)
    s = reduce(s, 'vigil/assistant/reply', msg({ status: 'done', text: 'Afară sunt 10 grade.', source: 'Open-Meteo' }), T)
    expect(s.assistant).toMatchObject({ status: 'done', heard: 'Cum e vremea?', text: 'Afară sunt 10 grade.', source: 'Open-Meteo' })
    s = reduce(s, 'vigil/assistant/reply', msg({ status: 'listening' }), T)
    expect(s.assistant?.heard).toBeUndefined()           // întrebare nouă: textul vechi dispare
    expect(reduce(initialState, 'vigil/assistant/reply', '{"status":"done"}', T)).toBe(initialState)
  })

  it('vigil/car/state + event + esp32/status: starea mașinii, evenimente, seria graficului', () => {
    let s = reduce(initialState, 'vigil/esp32/status', 'online', T)
    expect(s.ecuOnline).toBe(true)
    s = reduce(s, 'vigil/car/state', '{"speed_kmh":100,"target_kmh":100,"braking":false,"hazard":false}', T)
    s = reduce(s, 'vigil/car/event', '{"event":"emergency_braking_started","speed_kmh":98.5,"hazard":true,"braking":true,"emergency":true,"id":"ml2-3"}', T + 500)
    expect(s.car).toMatchObject({ speed_kmh: 98.5, braking: true, hazard: true, emergency: true, target_kmh: 100 })
    expect(s.carEvents[0]).toMatchObject({ event: 'emergency_braking_started', id: 'ml2-3' })
    expect(s.series.car.map((p) => p.speed)).toEqual([100, 98.5])
    expect(reduce(s, 'vigil/esp32/status', 'offline', T).ecuOnline).toBe(false)
    expect(reduce(s, 'vigil/car/state', '{"braking":true}', T)).toBe(s)   // fără viteză: ignorat
  })

  it('seria graficului păstrează doar ultimul minut', () => {
    const ecu = JSON.stringify({ mode: 'NORMAL', speed_factor: 1, speed_ref_kmh: 100, target_kmh: 100,
      lka_authority: 0, headway_s: 1.8, inputs: { a: 0.9 } })
    let s = initialState
    for (const t of [0, 30_000, 65_000]) s = reduce(s, 'vigil/ecu', ecu, t)
    expect(s.series.ecu.map((p) => p.t)).toEqual([30_000, 65_000])
    expect(s.series.ecu[1]).toMatchObject({ ref: 100, target: 100, a: 0.9, mode: 'NORMAL' })
  })

  it('vigil/ecu valid; incomplet ignorat', () => {
    const ok = '{"mode":"MRM","speed_factor":0,"speed_ref_kmh":62.5,"lka_authority":1,"headway_s":3.5,"hazards":true}'
    expect(reduce(initialState, 'vigil/ecu', ok, T).ecu?.mode).toBe('MRM')
    expect(reduce(initialState, 'vigil/ecu', '{"mode":"MRM"}', T)).toBe(initialState)
  })

  it('vigil/ml1/status', () => {
    expect(reduce(initialState, 'vigil/ml1/status', '{"a":0.4,"confidence":0.9}', T).status?.a).toBe(0.4)
  })

  it('dismissEmergency', () => {
    const s = decision({ severity: 3, actions: [{ tool: 'call_112', args: { report: 'R' } }] })
    expect(dismissEmergency(s).emergency).toBeNull()
  })
})
