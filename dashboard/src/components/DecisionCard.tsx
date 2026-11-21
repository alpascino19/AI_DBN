import type { Decision } from '../state'
import { Card, SEVERITY, clock } from './ui'

function Row({ d }: { d: Decision }) {
  const sev = SEVERITY[d.severity]
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className={`rounded px-2 py-0.5 text-xs font-bold ${sev.cls}`}>SEV {d.severity}</span>
        <span className="text-sm font-medium">{sev.label}</span>
        <span className="ml-auto text-xs text-slate-500">{clock(d.receivedAt)}</span>
      </div>
      {d.reason && <p className="text-sm text-slate-300">{d.reason}</p>}
      {d.actions.length > 0 && (
        <p className="font-mono text-xs text-sky-300">{d.actions.map((a) => a.tool).join(', ')}</p>
      )}
    </div>
  )
}

export function DecisionCard({ decision, history }: { decision: Decision | null; history: Decision[] }) {
  return (
    <Card title="Decizie LLM">
      {decision ? <Row d={decision} /> : <p className="text-sm text-slate-500">nicio decizie încă</p>}
      {decision?.source && (
        <p className="mt-1 text-xs text-slate-500">
          sursă: {decision.source}
          {decision.latency_ms !== undefined && ` · ${Math.round(decision.latency_ms)} ms`}
        </p>
      )}
      {history.length > 1 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-slate-400">istoric ({history.length - 1})</summary>
          <div className="mt-2 max-h-72 space-y-3 overflow-y-auto border-t border-slate-800 pt-2">
            {history.slice(1).map((d) => (
              <Row key={d.receivedAt + (d.reason ?? '')} d={d} />
            ))}
          </div>
        </details>
      )}
    </Card>
  )
}
