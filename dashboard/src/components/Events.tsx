import type { Vision } from '../state'
import { fmt } from './format'
import { Card } from './ui'

export function Events({ vision }: { vision: Vision | null }) {
  const events = Array.isArray(vision?.events) ? vision.events : []
  return (
    <Card title="Evenimente active">
      {events.length === 0 ? (
        <p className="text-sm text-slate-500">niciunul</p>
      ) : (
        <ul className="space-y-1.5 text-sm">
          {events.map((e, i) => (
            <li key={`${e.event}-${i}`} className="flex justify-between gap-2">
              <span className="font-medium text-orange-300">{String(e.event).replaceAll('_', ' ')}</span>
              <span className="tabular-nums text-slate-400">
                {fmt(e.duration_sec, 1)} s · {fmt(e.confidence, 2)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
