// Conținutul ferestrelor „VIGIL - FAȚA” și „VIGIL - CORP” (aceleași rânduri ca în ferestrele Vision,
// Vision/vigil_vision/face/monitor.py). Sursa: `face_panel` / `body_panel` din vigil/vision (Vision/vigil_vision/panel.py).
// Fără panou (Vision mai vechi, date false, cameră oprită) se derivă ce se poate din câmpurile de bază.

import type { Vision, VisionEvent } from './state'

export type Tone = 'normal' | 'ok' | 'warn' | 'bad'
export type Row = { label: string; value: string; tone?: Tone; bar?: { frac: number | null; tone: Tone } }
export type Section = { title: string; rows: Row[] }
export type WindowAlert = { level: number; message: string; sec: number | null }
export type WindowView = {
  title: string
  level: number
  badge: { text: string; tone: 'ok' | 'calib' | 'level' | 'off' }
  calibration: number | null
  sections: Section[]
  alerts: WindowAlert[]
  fps: number | null
  /** true = panoul complet de la Vision; false = derivat din câmpurile de bază */
  full: boolean
  eyesClosed?: boolean
  chestZone?: number[] | null
  chestOn?: boolean
  neck?: number[] | null
  posture?: string | null
}

type Obj = Record<string, unknown>
const obj = (v: unknown): Obj | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : null)
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const f = (v: unknown, d = 2) => (num(v) === null ? '—' : (v as number).toFixed(d))
const pct = (v: unknown) => (num(v) === null ? '—' : `${Math.round((v as number) * 100)}%`)

/** Ca `grade` din monitor.py: verde sub `warn`, portocaliu sub `bad`, roșu peste. */
export function grade(v: number | null, warn: number, bad: number): Tone {
  if (v === null || v < warn) return 'ok'
  return v < bad ? 'warn' : 'bad'
}

const DIRECTIONS: Record<string, string> = { center: 'la drum', left: 'STÂNGA', right: 'DREAPTA', down: 'JOS', up: 'SUS' }
const POSTURE: Record<string, [string, Tone]> = {
  upright: ['drept', 'normal'],
  leaning: ['aplecat', 'warn'],
  slumped: ['PRĂBUȘIT', 'bad'],
}
const CHEST_WHY: Record<string, string> = {
  face: 'la față', partial: 'parțial', fingers_up: 'degete sus', moving: 'în mișcare',
  depth: 'în aer', phone: 'telefon', close: 'lângă cameră',
}
const PHONE_USAGE: Record<string, string> = { call: 'apel', looking: 'se uită', holding: 'în mână' }
const CHEST_SRC: Record<string, string> = { hands: 'mână', pose: 'schelet' }

// evenimentele fiecărei camere, cu nivelul și textul folosite când lipsește panoul
const FACE_EVENTS: Record<string, [number, string]> = {
  eyes_closed: [2, 'Ochi închiși'], microsleep: [2, 'Micro-somn'], head_drop: [3, 'Cap căzut'],
  gaze_off_road: [1, 'Privire în afara drumului'], yawn: [1, 'Căscat'], frequent_yawning: [2, 'Căscat des'],
  low_blink_rate: [1, 'Clipit rar'], hyper_blinking: [1, 'Clipit foarte des'],
}
const BODY_EVENTS: Record<string, [number, string]> = {
  hand_on_chest: [3, 'Mână pe piept'], heavy_breathing: [2, 'Respirație grea'],
  phone_use: [1, 'Telefon în mână'], head_on_wheel: [3, 'Prăbușit pe volan'],
}

function events(v: Vision | null): VisionEvent[] {
  return Array.isArray(v?.events) ? (v.events as VisionEvent[]).filter((e) => obj(e) && typeof e.event === 'string') : []
}
const findEvent = (v: Vision | null, name: string) => events(v).find((e) => e.event === name)

function alertsFrom(raw: unknown): WindowAlert[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map(obj)
    .filter((a): a is Obj => !!a && num(a.level) !== null && typeof a.message === 'string')
    .map((a) => ({ level: a.level as number, message: a.message as string, sec: num(a.sec) }))
}

function eventAlerts(v: Vision | null, table: Record<string, [number, string]>): WindowAlert[] {
  return events(v)
    .filter((e) => e.event in table)
    .map((e) => ({ level: table[e.event][0], message: table[e.event][1], sec: num(e.duration_sec) }))
    .sort((a, b) => b.level - a.level)
    .slice(0, 3)
}

function badge(level: number, calibration: number | null, hasData: boolean): WindowView['badge'] {
  if (!hasData) return { text: 'FĂRĂ DATE', tone: 'off' }
  if (calibration !== null) return { text: `CALIBRARE ${Math.round(calibration * 100)}%`, tone: 'calib' }
  return level === 0 ? { text: 'OK', tone: 'ok' } : { text: `NIVEL ${level}`, tone: 'level' }
}

// ------------------------------------------------------------------ FAȚA

export function faceView(v: Vision | null): WindowView {
  const p = obj(v?.face_panel)
  return p ? faceFromPanel(p, v) : faceFallback(v)
}

function faceFromPanel(p: Obj, v: Vision | null): WindowView {
  const eyes = obj(p.eyes) ?? {}
  const head = obj(p.head) ?? {}
  const us = obj(p.unconscious) ?? {}
  const yawn = obj(p.yawn) ?? {}
  const level = num(p.level) ?? 0
  const calibration = num(p.calibration)
  const closure = num(eyes.closure)
  const perclos = num(eyes.perclos)
  const eyeDown = num(head.eye_down)
  const ratio = num(head.off_road_ratio)
  const offSec = num(head.off_road_sec) ?? 0
  const dir = str(head.direction)
  const state = str(us.state) ?? 'ok'
  const recent = num(yawn.recent) ?? 0

  return {
    title: 'VIGIL - FAȚA',
    level,
    badge: badge(level, calibration, true),
    calibration,
    full: true,
    eyesClosed: eyes.closed === true,
    fps: num(p.fps) ?? num(v?.fps),
    alerts: alertsFrom(p.alerts),
    sections: [
      {
        title: 'OCHI',
        rows: [
          eyes.visible === false
            ? { label: 'Stare', value: 'fără față', tone: 'warn' }
            : eyes.closed === true
              ? { label: 'Stare', value: `ÎNCHIȘI ${f(eyes.closed_sec, 1)}s`, tone: 'bad' }
              : { label: 'Stare', value: 'deschiși' },
          { label: 'Închidere', value: f(closure), bar: { frac: closure, tone: grade(closure, 0.5, 0.72) } },
          { label: 'Clipiri / min', value: f(eyes.blink_rate, 0) },
          {
            label: 'PERCLOS',
            value: pct(perclos),
            bar: { frac: perclos === null ? null : perclos / 0.3, tone: grade(perclos, 0.1, 0.15) },
          },
          {
            label: 'Ultimele 15 s',
            value: num(eyes.perclos_recent) === null ? '—' : `${pct(eyes.perclos_recent)} închiși, ${num(eyes.blinks_recent) ?? 0} clipiri`,
            tone: eyes.recovered === true ? 'ok' : 'normal',
          },
        ],
      },
      {
        title: 'PRIVIRE',
        rows: [
          head.visible === false
            ? { label: 'Direcție', value: 'fără față', tone: 'warn' }
            : { label: 'Direcție', value: dir ? (DIRECTIONS[dir] ?? dir) : '—', tone: !dir || dir === 'center' ? 'normal' : 'warn' },
          { label: 'Cap stg/dr  sus/jos', value: `${f(head.yaw, 0)} / ${f(head.pitch, 0)} grade` },
          { label: 'Ochi în jos', value: f(eyeDown), bar: { frac: eyeDown, tone: grade(eyeDown, 0.25, 0.35) } },
          { label: 'În afara drumului', value: `${offSec.toFixed(1)}s`, tone: offSec >= 2 ? 'bad' : 'normal' },
          { label: 'Ultimele 10 s', value: pct(ratio), bar: { frac: ratio, tone: grade(ratio, 0.3, 0.5) } },
        ],
      },
      {
        title: 'CONȘTIENȚĂ',
        rows: [
          {
            label: 'Stare',
            value: state === 'unconscious' ? `INCONȘTIENT ${f(us.since_sec, 0)}s` : state === 'suspect' ? 'SUSPECT' : 'normal',
            tone: state === 'unconscious' ? 'bad' : state === 'suspect' ? 'warn' : 'normal',
            bar: { frac: num(us.score), tone: grade(num(us.score), 0.45, 0.65) },
          },
          {
            label: 'Cap căzut / nemișcat',
            value: `${us.dropped === true ? 'DA' : 'nu'} / ${f(us.still_sec, 1)}s`,
            tone: us.dropped === true ? 'warn' : 'normal',
          },
        ],
      },
      {
        title: 'GURĂ',
        rows: [
          yawn.yawning === true
            ? { label: 'Căscat', value: `DA ${f(yawn.open_sec, 1)}s`, tone: 'warn' }
            : { label: 'Căscat', value: 'nu' },
          {
            label: 'Căscate (5 min)',
            value: `${recent}   (total ${num(yawn.total) ?? 0})`,
            tone: recent >= 3 ? 'bad' : 'normal',
          },
        ],
      },
    ],
  }
}

function faceFallback(v: Vision | null): WindowView {
  const visible = v?.face_detected === true
  const closed = findEvent(v, 'eyes_closed')
  const off = findEvent(v, 'gaze_off_road') as (VisionEvent & { direction?: unknown }) | undefined
  const yawn = findEvent(v, 'yawn')
  const perclos = num(v?.perclos)
  const dropped = v?.head_drop === true
  const alerts = eventAlerts(v, FACE_EVENTS)
  const level = alerts.reduce((m, a) => Math.max(m, a.level), 0)
  const dir = v?.gaze_on_road === true ? 'center' : off ? (str(off.direction) ?? 'off') : null
  return {
    title: 'VIGIL - FAȚA',
    level,
    badge: badge(level, null, !!v && (visible || v.landmarks !== undefined)),
    calibration: null,
    full: false,
    eyesClosed: !!closed,
    fps: num(v?.fps),
    alerts,
    sections: [
      {
        title: 'OCHI',
        rows: [
          !visible
            ? { label: 'Stare', value: 'fără față', tone: 'warn' }
            : closed
              ? { label: 'Stare', value: `ÎNCHIȘI ${f(closed.duration_sec, 1)}s`, tone: 'bad' }
              : { label: 'Stare', value: 'deschiși' },
          { label: 'EAR', value: f(v?.ear, 3) },
          { label: 'Clipiri / min', value: f(v?.blink_rate, 0) },
          {
            label: 'PERCLOS',
            value: pct(perclos),
            bar: { frac: perclos === null ? null : perclos / 0.3, tone: grade(perclos, 0.1, 0.15) },
          },
        ],
      },
      {
        title: 'PRIVIRE',
        rows: [
          !visible
            ? { label: 'Direcție', value: 'fără față', tone: 'warn' }
            : { label: 'Direcție', value: dir ? (DIRECTIONS[dir] ?? 'în afara drumului') : '—', tone: dir && dir !== 'center' ? 'warn' : 'normal' },
          {
            label: 'În afara drumului',
            value: `${(num(off?.duration_sec) ?? 0).toFixed(1)}s`,
            tone: (num(off?.duration_sec) ?? 0) >= 2 ? 'bad' : 'normal',
          },
        ],
      },
      {
        title: 'CONȘTIENȚĂ',
        rows: [{ label: 'Cap căzut', value: dropped ? 'DA' : 'nu', tone: dropped ? 'warn' : 'normal' }],
      },
      {
        title: 'GURĂ',
        rows: [yawn ? { label: 'Căscat', value: `DA ${f(yawn.duration_sec, 1)}s`, tone: 'warn' } : { label: 'Căscat', value: 'nu' }],
      },
    ],
  }
}

// ------------------------------------------------------------------ CORP

export function bodyView(v: Vision | null): WindowView {
  const p = obj(v?.body_panel)
  const base = bodyRows(v, p)
  const alerts = p ? alertsFrom(p.alerts) : eventAlerts(v, BODY_EVENTS)
  const level = p ? (num(p.level) ?? 0) : alerts.reduce((m, a) => Math.max(m, a.level), 0)
  const hasData = !!p || v?.body_detected === true || v?.body_landmarks !== undefined
  return {
    title: 'VIGIL - CORP',
    level,
    badge: badge(level, null, hasData),
    calibration: null,
    full: !!p,
    fps: num(p?.fps) ?? num(v?.fps),
    alerts,
    sections: [{ title: 'CORP', rows: base.rows }],
    chestZone: Array.isArray(p?.chest_zone) ? (p.chest_zone as number[]) : null,
    chestOn: base.chestOn,
    neck: Array.isArray(p?.neck) ? (p.neck as number[]) : null,
    posture: base.posture,
  }
}

function bodyRows(v: Vision | null, p: Obj | null): { rows: Row[]; chestOn: boolean; posture: string | null } {
  const rows: Row[] = []
  const visible = p ? p.visible !== false : v?.body_detected === true
  if (!visible) rows.push({ label: 'Corp', value: 'nu se vede', tone: 'warn' })

  const chest = obj(p?.chest)
  const chestOn = chest ? chest.on === true : !!findEvent(v, 'hand_on_chest')
  const why = CHEST_WHY[str(chest?.reject) ?? '']
  const src = CHEST_SRC[str(chest?.src) ?? ''] ?? ''
  rows.push({
    label: 'Mână pe piept',
    value: chestOn ? (src ? `DA (${src})` : 'DA') : why ? `nu (${why})` : 'nu',
    tone: chestOn ? 'bad' : 'normal',
  })
  if (p) rows.push({ label: 'Mărime mână', value: f(p.hand_scale), tone: chest?.reject === 'close' ? 'warn' : 'normal' })

  const b = obj(p?.breathing) ?? obj(v?.breathing) ?? {}
  const heavy = b.state === 'heavy'
  const score = num(b.score)
  rows.push({
    label: 'Respirație',
    value: heavy ? 'GREA' : (str(b.state) ?? '—'),
    tone: heavy ? 'bad' : 'normal',
    bar: { frac: score, tone: grade(score, 0.4, 0.55) },
  })
  const rate = num(b.rate_bpm)
  rows.push({
    label: 'Ritm respirație',
    value: (rate === null ? '—' : `${Math.round(rate)} / min`) + (b.calibrated === false ? '  (calibrare)' : ''),
  })

  const ph = obj(p?.phone)
  const phone = num(ph?.score) ?? num(v?.phone) ?? 0
  const active = ph ? ph.active === true : !!findEvent(v, 'phone_use')
  const usage = PHONE_USAGE[str(ph?.usage) ?? '']
  rows.push({
    label: 'Telefon',
    value: active ? (usage ?? 'da') : phone < 0.05 ? 'nu' : phone.toFixed(2),
    tone: active ? 'warn' : 'normal',
    bar: { frac: phone, tone: grade(phone, 0.25, 0.4) },
  })

  const sl = obj(p?.posture) ?? obj(v?.posture) ?? {}
  const st = str(sl.state)
  const [label, tone] = st ? (POSTURE[st] ?? [st, 'normal' as Tone]) : ['—', 'normal' as Tone]
  rows.push({ label: 'Postură', value: label + (sl.calibrated === false ? ' (calib.)' : ''), tone })
  return { rows, chestOn, posture: st }
}
