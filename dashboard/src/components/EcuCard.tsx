import type { Ecu } from '../state'
import { fmt } from './format'
import { Card } from './ui'

const MODE: Record<string, { label: string; cls: string }> = {
  NORMAL: { label: 'normal', cls: 'bg-green-500/15 text-green-300' },
  CAUTION: { label: 'prudent', cls: 'bg-yellow-400/15 text-yellow-200' },
  ASSIST: { label: 'asistență', cls: 'bg-orange-500/15 text-orange-300' },
  MRM: { label: 'oprire pe banda de urgență', cls: 'bg-red-500/20 text-red-300' },
}
const STATES = ['atent', 'distras', 'obosit', 'urgență']
const BAR = ['bg-emerald-500', 'bg-yellow-400', 'bg-orange-500', 'bg-red-500']

/** Comanda fuzzy ML1 -> ECU (vigil/ecu) și probabilitatea fiecărei stări (filtrul HMM). */
export function EcuCard({ ecu }: { ecu: Ecu | null }) {
  if (!ecu) {
    return (
      <Card title="ECU (control fuzzy)">
        <p className="text-sm text-slate-400">nicio comandă încă (vigil/ecu)</p>
      </Card>
    )
  }
  const mode = MODE[ecu.mode] ?? { label: ecu.mode, cls: 'bg-slate-800 text-slate-300' }
  const p = ecu.state?.p_state ?? []
  return (
    <Card title="ECU (control fuzzy)">
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className={`rounded px-2 py-0.5 text-sm font-semibold ${mode.cls}`}>{mode.label}</span>
        {ecu.hazards && <span className="animate-pulse text-sm font-semibold text-orange-400">avarii</span>}
      </div>
      <div className="mb-3 font-mono">
        <span className="text-3xl">{fmt(ecu.speed_ref_kmh, 0)}</span>
        <span className="text-slate-400"> km/h</span>
        <span className="ml-2 text-sm text-slate-400">→ {fmt(ecu.target_kmh, 0)} ({fmt(ecu.speed_factor * 100, 0)}%)</span>
      </div>
      <dl className="mb-3 grid grid-cols-3 gap-2 text-sm">
        <div>
          <dt className="text-slate-400">asistență bandă</dt>
          <dd className="font-mono">{fmt(ecu.lka_authority * 100, 0)}%</dd>
        </div>
        <div>
          <dt className="text-slate-400">distanță</dt>
          <dd className="font-mono">{fmt(ecu.headway_s, 1)} s</dd>
        </div>
        <div>
          <dt className="text-slate-400">frânare</dt>
          <dd className="font-mono">{fmt(ecu.decel_mps2, 1)} m/s²</dd>
        </div>
      </dl>
      {p.length === 4 && (
        <div className="grid gap-1 text-xs">
          {p.map((v, i) => (
            <div key={STATES[i]} className="grid grid-cols-[4.5rem_1fr_2.5rem] items-center gap-2">
              <span className="text-slate-400">{STATES[i]}</span>
              <div className="h-2 rounded bg-slate-800">
                <div className={`h-2 rounded ${BAR[i]}`} style={{ width: `${Math.round(v * 100)}%` }} />
              </div>
              <span className="text-right font-mono">{fmt(v * 100, 0)}%</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
