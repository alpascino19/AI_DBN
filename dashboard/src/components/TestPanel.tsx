import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ALERT_STEP_MS, ALERT_STEPS, ECU_TESTS, RESET, calmDecision, resetAll, test112, voiceTest, type Publish } from '../controls'
import { Card } from './ui'

type LogLine = { at: number; text: string }

function Btn({ children, onClick, disabled, tone = 'normal', title }: {
  children: ReactNode; onClick: () => void; disabled?: boolean; tone?: 'normal' | 'warn' | 'armed'; title?: string
}) {
  const cls = tone === 'armed' ? 'border-red-500 bg-red-600 text-white'
    : tone === 'warn' ? 'border-orange-700 bg-orange-950/40 hover:bg-orange-900/50'
      : 'border-slate-700 bg-slate-800 hover:bg-slate-700'
  return (
    <button type="button" onClick={onClick} disabled={disabled} title={title}
      className={`rounded-md border px-2 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40 ${cls}`}>
      {children}
    </button>
  )
}

/** Panoul „Teste & reset”: fiecare componentă poate fi arătată separat la demo, fără camere. */
export function TestPanel({ publish, connected, onDismiss }: { publish: Publish; connected: boolean; onDismiss: () => void }) {
  const [log, setLog] = useState<LogLine[]>([])
  const [armed, setArmed] = useState<string | null>(null)   // acțiunile mari cer a doua apăsare
  const [alertStep, setAlertStep] = useState<number | null>(null)
  const ids = useRef(0)
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])
  const sequence = useRef<ReturnType<typeof setTimeout>[]>([])   // pașii ESP32 1→2→3→0, anulabili
  useEffect(() => () => [...timers.current, ...sequence.current].forEach(clearTimeout), [])

  const later = (fn: () => void, ms: number) => timers.current.push(setTimeout(fn, ms))
  /** Un reset oprește secvența ESP32 în curs: altfel nivelurile 2 și 3 ar reaprinde LED-ul după reset. */
  const stopSequence = () => {
    sequence.current.forEach(clearTimeout)
    sequence.current = []
    setAlertStep(null)
  }
  const note = (text: string) => setLog((l) => [{ at: Date.now(), text }, ...l].slice(0, 5))
  const nextId = () => `dashboard-${++ids.current}`
  const confirm = (key: string, run: () => void) => {
    if (armed === key) {
      setArmed(null)
      run()
    } else {
      setArmed(key)
      later(() => setArmed((a) => (a === key ? null : a)), 3000)
    }
  }
  const label = (key: string, text: string) => (armed === key ? 'Sigur? Apasă din nou' : text)

  const alertSequence = () => {
    stopSequence()
    sequence.current = ALERT_STEPS.map((level, i) => setTimeout(() => {
      publish('vigil/alert', String(level), { retain: true })
      setAlertStep(level === 0 ? null : level)
    }, i * ALERT_STEP_MS))
    note(`ESP32: nivelurile ${ALERT_STEPS.join(' → ')} (câte ${ALERT_STEP_MS / 1000} s)`)
  }

  return (
    <Card title="Teste & reset">
      <h3 className="mb-1.5 text-xs font-semibold text-slate-400">Reset</h3>
      <div className="grid grid-cols-2 gap-1.5">
        <Btn disabled={!connected} onClick={() => { stopSequence(); RESET.alerts(publish); note('Reset alerte: ESP32 stins') }}>
          Alerte (ESP32 stins)
        </Btn>
        <Btn disabled={!connected} onClick={() => { const id = nextId(); RESET.car(publish, id); note(`Reset mașină: croazieră, fără avarii (${id})`) }}>
          Mașina (croazieră)
        </Btn>
        <Btn disabled={!connected} title="Călătorie nouă: calibrarea de 30 s pornește de la zero"
          onClick={() => { const id = nextId(); RESET.ml1(publish, id); note(`Reset ML1: calibrare de la zero (${id})`) }}>
          ML1 (recalibrare)
        </Btn>
        <Btn disabled={!connected} tone={armed === 'all' ? 'armed' : 'warn'}
          onClick={() => confirm('all', () => {
            const id = nextId()
            stopSequence()
            resetAll(publish, id)
            onDismiss()
            note(`Reset complet: alerte, mașină, ML1, ML2 (${id})`)
          })}>
          {label('all', 'Tot')}
        </Btn>
      </div>

      <h3 className="mb-1.5 mt-3 text-xs font-semibold text-slate-400">Componente</h3>
      <div className="grid grid-cols-3 gap-1.5">
        <Btn disabled={!connected || alertStep !== null} onClick={alertSequence}
          title="Placa ESP32: LED aprins, clipește, clipește rapid, stins">
          {alertStep !== null ? `ESP32 nivel ${alertStep}...` : 'ESP32 1→2→3→0'}
        </Btn>
        <Btn disabled={!connected} title="ML2 rostește o frază de test (cel mult o dată la 10 s)"
          onClick={() => { publish('vigil/decision', voiceTest()); note('Voce ML2: frază de test') }}>
          Voce ML2
        </Btn>
        <Btn disabled={!connected} tone={armed === '112' ? 'armed' : 'warn'}
          title="Apel 112 SIMULAT: raport + voce + fereastra roșie; nimic nu pleacă de pe laptop"
          onClick={() => confirm('112', () => {
            publish('vigil/decision', test112())
            later(() => publish('vigil/decision', calmDecision()), 15000)
            note('Apel 112 simulat (TEST); revenire la normal după 15 s')
          })}>
          {label('112', '112 simulat')}
        </Btn>
      </div>

      <h3 className="mb-1.5 mt-3 text-xs font-semibold text-slate-400">ECU (placa mașinii)</h3>
      <div className="grid grid-cols-2 gap-1.5">
        {ECU_TESTS.map((t) => (
          <Btn key={t.key} disabled={!connected}
            onClick={() => { const id = nextId(); publish('vigil/car/cmd', { ...t.cmd, id }); note(`ECU: ${t.label.toLowerCase()} (${id})`) }}>
            {t.label}
          </Btn>
        ))}
      </div>
      <p className="mt-2 text-[11px] leading-snug text-slate-500">
        Răspunsul plăcii apare în graficul mașinii. Cât timp ML1 rulează, ML2 poate suprascrie o comandă ECU manuală la
        următoarea schimbare a comenzii fuzzy.
      </p>

      {log.length > 0 && (
        <ul className="mt-2 space-y-0.5 border-t border-slate-800 pt-2 text-[11px] text-slate-400" aria-live="polite">
          {log.map((l) => (
            <li key={l.at + l.text}>
              <span className="tabular-nums text-slate-500">{new Date(l.at).toLocaleTimeString('ro-RO')}</span> · {l.text}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
