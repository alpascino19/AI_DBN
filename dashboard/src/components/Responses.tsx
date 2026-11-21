import type { DriverResponse } from '../state'
import { clock, fmt } from './format'
import { Card } from './ui'

export function Responses({ responses }: { responses: DriverResponse[] }) {
  return (
    <Card title="Răspunsul șoferului">
      {responses.length === 0 ? (
        <p className="text-sm text-slate-400">niciun răspuns încă</p>
      ) : (
        <ul className="max-h-60 space-y-2 overflow-y-auto text-sm">
          {responses.map((r) => (
            <li key={r.receivedAt} className="flex gap-2">
              <span className="w-16 shrink-0 text-xs text-slate-400">{clock(r.receivedAt)}</span>
              {r.answered ? (
                <span>
                  „{r.text}” <span className="text-xs text-slate-400">({fmt(r.latency_sec, 1)} s)</span>
                </span>
              ) : (
                <span className="font-medium text-red-400">fără răspuns</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
