import { describe, expect, it, vi } from 'vitest'
import { ECU_TESTS, resetAll, test112, voiceTest } from './controls'
import { VideoStore } from './videoStore'

// aceeași validare ca ML2/executor.py (TOOLS): unealtă cunoscută + exact argumentele cerute
const TOOLS: Record<string, string[]> = { speak: ['text'], play_alarm: ['level'], ask_driver: ['question'],
  set_esp32: ['level'], call_112: ['report'], suggest_break: [] }
const validForMl2 = (d: { severity: number; actions: { tool: string; args: Record<string, unknown> }[] }) =>
  Number.isInteger(d.severity) && d.severity >= 0 && d.severity <= 3 &&
  d.actions.every((a) => a.tool in TOOLS && JSON.stringify(Object.keys(a.args).sort()) === JSON.stringify([...TOOLS[a.tool]].sort()))

describe('testele din dashboard respectă contractele', () => {
  it('voce și 112 trec validarea ML2; 112 e marcat TEST', () => {
    expect(validForMl2(voiceTest(0))).toBe(true)
    const t = test112(0)
    expect(validForMl2(t)).toBe(true)
    expect(t.severity).toBe(3)
    expect(t.actions[0].args.report).toContain('NU este o urgență reală')
  })

  it('comenzile ECU sunt cele din ECU_Arduino.ino', () => {
    const known = ['emergency', 'cancel_emergency', 'slow_down', 'hazard', 'resume', 'reset']
    for (const t of ECU_TESTS) expect(known).toContain(t.cmd.cmd)
    const slow = ECU_TESTS.filter((t) => t.cmd.cmd === 'slow_down')
    for (const t of slow) expect('decel_mps2' in t.cmd && t.cmd.decel_mps2).toBeLessThanOrEqual(4)  // placa refuză > 4
  })

  it('reset complet: ESP32 0 reținut, mașina reset, ML1 reset, ML2 la normal', () => {
    const sent: [string, unknown, unknown][] = []
    resetAll((t, p, o) => sent.push([t, p, o]), 'dashboard-9', 0)
    expect(sent.map((s) => s[0])).toEqual(['vigil/alert', 'vigil/car/cmd', 'vigil/ml1/cmd', 'vigil/decision'])
    expect(sent[0]).toEqual(['vigil/alert', '0', { retain: true }])
    expect(sent[1][1]).toEqual({ cmd: 'reset', id: 'dashboard-9' })
    expect(sent[2][1]).toEqual({ cmd: 'reset', id: 'dashboard-9' })
    expect((sent[3][1] as { severity: number }).severity).toBe(0)
  })
})

describe('VideoStore', () => {
  it('fiecare cameră își anunță doar abonații ei; fps; cadrul vechi eliberat; camera oprită dispare', () => {
    vi.useFakeTimers()
    let n = 0
    const revoked: string[] = []
    const store = new VideoStore(() => `blob:${++n}`, (u) => revoked.push(u))
    const face = vi.fn()
    const body = vi.fn()
    store.subscribe('face', face)
    store.subscribe('body', body)
    store.push('face', new Uint8Array([1]), 1000)
    store.push('face', new Uint8Array([2]), 1050)
    store.push('face', new Uint8Array([3]), 1100)
    expect(face).toHaveBeenCalledTimes(3)
    expect(body).not.toHaveBeenCalled()                 // fereastra CORP nu se re-randează
    expect(store.get('face')).toMatchObject({ url: 'blob:3', fps: 20 })
    vi.advanceTimersByTime(1000)
    expect(revoked).toEqual(['blob:1', 'blob:2'])
    vi.advanceTimersByTime(3000)                        // 3 s fără cadre: camera oprită
    expect(store.get('face')).toBeUndefined()
    vi.useRealTimers()
  })
})
