import type { ReactNode } from 'react'

export const SEVERITY = [
  { label: 'normal', cls: 'bg-emerald-600' },
  { label: 'distragere', cls: 'bg-yellow-500 text-slate-950' },
  { label: 'oboseală', cls: 'bg-orange-500' },
  { label: 'urgență', cls: 'bg-red-600' },
]

export function Card({ title, children, className = '' }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900 p-4 ${className}`}>
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</h2>
      {children}
    </section>
  )
}

/** Valori lipsă (cameră oprită, față nedetectată) -> „—”. */
export function fmt(v: unknown, digits = 2): string {
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(digits)
  if (typeof v === 'boolean') return v ? 'da' : 'nu'
  if (typeof v === 'string' && v) return v
  return '—'
}

export const clock = (t: number) => new Date(t).toLocaleTimeString('ro-RO')
