import type { State } from '../state'
import { Card, fmt } from './ui'

export function AttentionGauge({ state }: { state: State }) {
  // scorul vine de la ML 1 (vigil/ml1/status, 1/s); altfel din ultima decizie
  const a = state.status?.a ?? state.decision?.a
  const pct = typeof a === 'number' ? Math.max(0, Math.min(1, a)) * 100 : 0
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 40 ? 'bg-orange-500' : 'bg-red-600'
  return (
    <Card title="Atenție">
      <div className="text-5xl font-bold tabular-nums">{fmt(a)}</div>
      <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full ${color} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 text-sm text-slate-400">încredere {fmt(state.status?.confidence)}</div>
      {state.status?.camera_on === false && <div className="mt-1 text-sm text-orange-400">camera oprită</div>}
    </Card>
  )
}
