// Constante și formatare comune (separat de componente: fast refresh-ul Vite cere fișiere doar cu componente).

export const SEVERITY = [
  { label: 'normal', cls: 'bg-emerald-600' },
  { label: 'distragere', cls: 'bg-yellow-500 text-slate-950' },
  { label: 'oboseală', cls: 'bg-orange-500' },
  { label: 'urgență', cls: 'bg-red-600' },
]

/** Valori lipsă (cameră oprită, față nedetectată) -> „—”. */
export function fmt(v: unknown, digits = 2): string {
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(digits)
  if (typeof v === 'boolean') return v ? 'da' : 'nu'
  if (typeof v === 'string' && v) return v
  return '—'
}

export const clock = (t: number) => new Date(t).toLocaleTimeString('ro-RO')
