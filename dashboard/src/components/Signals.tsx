import type { Vision } from '../state'
import { Card, fmt } from './ui'

const state = (v: unknown) => (v && typeof v === 'object' ? (v as { state?: unknown }).state : undefined)

export function Signals({ vision }: { vision: Vision | null }) {
  const v = vision ?? {}
  const rows: [string, string, boolean?][] = [
    ['EAR', fmt(v.ear, 3)],
    ['PERCLOS', fmt(v.perclos, 2), typeof v.perclos === 'number' && v.perclos > 0.3],
    ['clipit / min', fmt(v.blink_rate, 0)],
    ['privire pe drum', fmt(v.gaze_on_road), v.gaze_on_road === false],
    ['cap căzut', fmt(v.head_drop), v.head_drop === true],
    ['telefon', fmt(v.phone, 2), typeof v.phone === 'number' && v.phone >= 0.6],
    ['respirație', fmt(state(v.breathing)), state(v.breathing) === 'heavy'],
    ['postură', fmt(state(v.posture))],
    ['FPS', fmt(v.fps, 0)],
  ]
  return (
    <Card title="Semnale cameră">
      <dl className="grid grid-cols-3 gap-x-4 gap-y-2 text-sm sm:grid-cols-5">
        {rows.map(([k, val, bad]) => (
          <div key={k}>
            <dt className="text-slate-500">{k}</dt>
            <dd className={`font-mono text-base ${bad ? 'text-orange-400' : ''}`}>{val}</dd>
          </div>
        ))}
      </dl>
    </Card>
  )
}
