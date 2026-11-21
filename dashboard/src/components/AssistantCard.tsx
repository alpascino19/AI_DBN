import type { Assistant, AssistantRequest } from '../state'
import { Card } from './ui'

const QUICK: [AssistantRequest, string][] = [
  ['time', 'Ora'],
  ['weather', 'Vremea'],
  ['traffic', 'Traficul'],
]
const STATUS: Record<string, string> = {
  listening: 'Virgil ascultă... puneți întrebarea',
  working: 'caută răspunsul...',
  speaking: 'Virgil răspunde',
  done: 'gata',
  interrupted: 'întrerupt de o alertă',
}

/** Butoanele asistentului vocal (ML2): întrebare liberă la microfon sau răspuns direct la oră / vreme / trafic. */
export function AssistantCard({ assistant, connected, onAsk }: {
  assistant: Assistant | null
  connected: boolean
  onAsk: (r: AssistantRequest) => void
}) {
  const busy = !!assistant && ['listening', 'working', 'speaking'].includes(assistant.status)
  const listening = assistant?.status === 'listening'
  return (
    <Card title="Asistent Virgil">
      <button
        type="button"
        disabled={!connected}
        onClick={() => onAsk('listen')}
        className={`flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-base font-semibold text-white transition
          disabled:cursor-not-allowed disabled:opacity-40 ${listening ? 'animate-pulse bg-red-600' : 'bg-sky-600 hover:bg-sky-500'}`}
      >
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" />
        </svg>
        {listening ? 'Ascult...' : 'Întreabă-l pe Virgil'}
      </button>
      <div className="mt-2 grid grid-cols-3 gap-2">
        {QUICK.map(([req, label]) => (
          <button
            key={req}
            type="button"
            disabled={!connected}
            onClick={() => onAsk(req)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-2 text-sm font-medium hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-3 min-h-16 text-sm" aria-live="polite">
        {!assistant ? (
          <p className="text-slate-500">Apăsați un buton: Virgil răspunde în difuzoare.</p>
        ) : (
          <>
            <p className={`text-xs ${busy ? 'text-sky-300' : 'text-slate-500'}`}>{STATUS[assistant.status] ?? assistant.status}</p>
            {assistant.heard && <p className="mt-1 text-slate-400">Șofer: „{assistant.heard}”</p>}
            {assistant.text && <p className="mt-1 text-slate-100">Virgil: {assistant.text}</p>}
            {assistant.source && <p className="mt-1 text-xs text-slate-500">sursa: {assistant.source}</p>}
          </>
        )}
      </div>
    </Card>
  )
}
