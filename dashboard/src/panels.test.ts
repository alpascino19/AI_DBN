import { describe, expect, it } from 'vitest'
import { bodyView, faceView, grade } from './panels'
import { headCenter } from './skeleton'
import type { Vision } from './state'

// forma trimisă de Vision/vigil_vision/panel.py (prin merge.py)
const FACE_PANEL = {
  fps: 30, level: 0, calibration: null,
  eyes: { visible: true, closed: false, closed_sec: 0, closure: 0.03, blink_rate: 0, perclos: 0, perclos_recent: 0, blinks_recent: 0, recovered: false },
  head: { visible: true, direction: 'center', yaw: -1, pitch: -0, eye_down: -0.03, off_road_sec: 0, off_road_ratio: 0.18 },
  unconscious: { state: 'ok', score: 0.02, since_sec: 0, dropped: false, still_sec: 1.1 },
  yawn: { yawning: false, open_sec: 0, recent: 0, total: 0 },
  alerts: [],
}
const BODY_PANEL = {
  fps: 11, level: 0, visible: true, chest: { on: false, src: null, reject: null }, hand_scale: null,
  breathing: { state: 'normal', score: 0.18, rate_bpm: 14, calibrated: true },
  phone: { score: 0.01, active: false, usage: null }, posture: { state: 'upright', calibrated: true },
  chest_zone: [0, 0.77, 1, 1], neck: [0.5, 0.95], alerts: [],
}
const row = (v: ReturnType<typeof faceView>, label: string) =>
  v.sections.flatMap((s) => s.rows).find((r) => r.label === label)

describe('faceView din panoul Vision (ca fereastra VIGIL - FATA)', () => {
  const v = faceView({ face_panel: FACE_PANEL, face_detected: true, fps: 11 } as Vision)

  it('aceleași secțiuni și rânduri', () => {
    expect(v.sections.map((s) => s.title)).toEqual(['OCHI', 'PRIVIRE', 'CONȘTIENȚĂ', 'GURĂ'])
    expect(row(v, 'Stare')?.value).toBe('deschiși')
    expect(row(v, 'Închidere')?.value).toBe('0.03')
    expect(row(v, 'Ultimele 15 s')?.value).toBe('0% închiși, 0 clipiri')
    expect(row(v, 'Direcție')?.value).toBe('la drum')
    expect(row(v, 'Cap stg/dr  sus/jos')?.value).toBe('-1 / 0 grade')   // -0 din Python apare „0”
    expect(row(v, 'Ultimele 10 s')?.value).toBe('18%')
    expect(row(v, 'Cap căzut / nemișcat')?.value).toBe('nu / 1.1s')
    expect(row(v, 'Căscate (5 min)')?.value).toBe('0   (total 0)')
  })

  it('insigna OK, fps-ul camerei feței (nu minimul combinat)', () => {
    expect(v.badge).toEqual({ text: 'OK', tone: 'ok' })
    expect(v.fps).toBe(30)
    expect(v.full).toBe(true)
  })

  it('ochi închiși, calibrare și avertizări', () => {
    const closed = faceView({
      face_panel: { ...FACE_PANEL, level: 2, eyes: { ...FACE_PANEL.eyes, closed: true, closed_sec: 1.6 },
        alerts: [{ level: 2, message: 'Ochi inchisi', sec: 2 }] },
    } as Vision)
    expect(row(closed, 'Stare')).toMatchObject({ value: 'ÎNCHIȘI 1.6s', tone: 'bad' })
    expect(closed.badge).toEqual({ text: 'NIVEL 2', tone: 'level' })
    expect(closed.alerts).toEqual([{ level: 2, message: 'Ochi inchisi', sec: 2 }])
    expect(closed.eyesClosed).toBe(true)
    const calib = faceView({ face_panel: { ...FACE_PANEL, calibration: 0.4 } } as Vision)
    expect(calib.badge).toEqual({ text: 'CALIBRARE 40%', tone: 'calib' })
  })
})

describe('faceView fără panou', () => {
  it('derivă din câmpurile de bază', () => {
    const v = faceView({
      face_detected: true, perclos: 0.12, blink_rate: 14, gaze_on_road: false, head_drop: false,
      events: [{ event: 'gaze_off_road', duration_sec: 2.4, confidence: 0.8, direction: 'down' } as never],
    } as Vision)
    expect(v.full).toBe(false)
    expect(row(v, 'Direcție')?.value).toBe('JOS')
    expect(row(v, 'În afara drumului')).toMatchObject({ value: '2.4s', tone: 'bad' })
    expect(row(v, 'PERCLOS')?.bar?.tone).toBe('warn')
    expect(v.alerts[0]).toEqual({ level: 1, message: 'Privire în afara drumului', sec: 2.4 })
  })

  it('fără niciun mesaj: FĂRĂ DATE', () => {
    expect(faceView(null).badge.tone).toBe('off')
    expect(bodyView(null).badge.tone).toBe('off')
  })
})

describe('bodyView (ca fereastra VIGIL - CORP)', () => {
  it('din panou', () => {
    const v = bodyView({ body_panel: BODY_PANEL } as Vision)
    const rows = Object.fromEntries(v.sections[0].rows.map((r) => [r.label, r.value]))
    expect(rows).toEqual({
      'Mână pe piept': 'nu', 'Mărime mână': '—', Respirație: 'normal', 'Ritm respirație': '14 / min',
      Telefon: 'nu', Postură: 'drept', 'Înclinare laterală': '—', 'Cap coborât': '—', Aplecare: '—',
      'Unghi cap (aplecat)': '—', 'Prăbușire (scor)': '—', Nemișcat: '0.0s',
    })
    expect(v.fps).toBe(11)
    expect(v.chestZone).toEqual([0, 0.77, 1, 1])
  })

  it('postura nouă din Vision: nu se vede, valori de reglaj, fără respirație', () => {
    const v = bodyView({ body_panel: { ...BODY_PANEL, visible: false, breathing: null,
      posture: { state: 'upright', out: 'lost', side_deg: 27, drop: 0.31, lean: null, angle: 14, score: 0.52, still_sec: 2 } },
    } as Vision)
    const r = Object.fromEntries(v.sections[0].rows.map((x) => [x.label, x]))
    expect(r.Corp).toMatchObject({ value: 'nu se vede', tone: 'warn' })        // ca rândul din fereastra CORP
    expect(r['Postură']).toMatchObject({ value: 'nu se vede', tone: 'warn' })
    expect(r['Respirație']).toBeUndefined()                                     // breathing.enabled: false
    expect(r['Înclinare laterală']).toMatchObject({ value: '27 grade', bar: { tone: 'bad' } })
    expect(r['Unghi cap (aplecat)']).toMatchObject({ value: '14 grade', tone: 'warn' })
    expect(r['Prăbușire (scor)'].bar?.tone).toBe('bad')
    expect(r['Nemișcat'].value).toBe('2.0s')
  })

  it('mână pe piept + respirație grea, fără panou', () => {
    const v = bodyView({
      body_detected: true, phone: 0, breathing: { state: 'heavy', rate_bpm: 31 }, posture: { state: 'slumped' },
      events: [{ event: 'hand_on_chest', duration_sec: 4, confidence: 0.8 }],
    } as Vision)
    const r = Object.fromEntries(v.sections[0].rows.map((x) => [x.label, x]))
    expect(r['Mână pe piept']).toMatchObject({ value: 'DA', tone: 'bad' })
    expect(r['Respirație']).toMatchObject({ value: 'GREA', tone: 'bad' })
    expect(r['Postură']).toMatchObject({ value: 'PRĂBUȘIT', tone: 'bad' })
    expect(v.level).toBe(3)
  })
})

describe('utilitare', () => {
  it('grade ca în monitor.py', () => {
    expect([grade(null, 0.5, 0.72), grade(0.4, 0.5, 0.72), grade(0.6, 0.5, 0.72), grade(0.8, 0.5, 0.72)])
      .toEqual(['ok', 'ok', 'warn', 'bad'])
  })

  it('centrul capului din punctele Pose', () => {
    const body = Array.from({ length: 25 }, () => [0, 0])
    for (const i of [0, 2, 5, 7, 8]) body[i] = [0.5, 0.2]
    expect(headCenter(body)).toEqual([0.5, 0.2])
    expect(headCenter([])).toBeNull()
  })
})
