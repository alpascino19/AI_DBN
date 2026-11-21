import type { State } from '../state'

const LED = ['stins', 'aprins', 'clipește', 'urgență']

export function Header({ state, connected, broker, onReset }: { state: State; connected: boolean; broker: string; onReset: () => void }) {
  const cal = state.status?.calibration as { ready?: boolean } | undefined
  const level = state.alertLevel
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-slate-800 px-4 py-3">
      <h1 className="text-xl font-bold tracking-tight">Vigil</h1>
      <span className="flex items-center gap-2 text-sm" title={broker}>
        <span className={`h-2.5 w-2.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-red-500'}`} />
        {connected ? 'conectat' : 'deconectat — reîncerc…'}
      </span>
      <span className="text-sm text-slate-400">
        calibrare: {cal === undefined ? '—' : cal.ready ? <span className="text-emerald-400">gata</span> : 'în curs'}
      </span>
      <span className="flex items-center gap-2 text-sm text-slate-400">
        ESP32:
        <span
          className={`h-3 w-3 rounded-full ${
            level === null || level === 0 ? 'bg-slate-700' : level === 3 ? 'animate-ping bg-red-500' : level === 2 ? 'animate-pulse bg-orange-400' : 'bg-yellow-300'
          }`}
        />
        {level === null ? '—' : `${level} (${LED[level]})`}
      </span>
      <button
        onClick={onReset}
        disabled={!connected}
        className="ml-auto rounded-lg bg-slate-700 px-3 py-1.5 text-sm font-medium hover:bg-slate-600 disabled:opacity-40"
      >
        Reset alerte
      </button>
    </header>
  )
}
