import { useRef, useState, type KeyboardEvent, type ReactNode } from 'react'

export type Tab = { id: string; label: string; panel: ReactNode; alert?: boolean }

/** Coloana din dreapta: un panou pe rând, ales din butoane mari. Panourile rămân montate (doar ascunse),
 *  ca jurnalul din „Teste & reset” să nu se piardă la schimbarea tabului. Săgețile mută selecția. */
export function SideTabs({ tabs }: { tabs: Tab[] }) {
  const [active, setActive] = useState(tabs[0].id)
  const buttons = useRef<(HTMLButtonElement | null)[]>([])
  const key = (e: KeyboardEvent, i: number) => {
    const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0
    if (!step) return
    e.preventDefault()
    const next = (i + step + tabs.length) % tabs.length
    setActive(tabs[next].id)
    buttons.current[next]?.focus()
  }
  return (
    <div className="grid content-start gap-3">
      <div role="tablist" aria-label="Panouri" className="grid grid-cols-2 gap-1.5 rounded-xl border border-slate-800 bg-slate-900 p-1.5">
        {tabs.map((t, i) => {
          const on = t.id === active
          return (
            <button key={t.id} ref={(el) => { buttons.current[i] = el }} type="button" role="tab"
              id={`tab-${t.id}`} aria-selected={on} aria-controls={`panel-${t.id}`} tabIndex={on ? 0 : -1}
              onClick={() => setActive(t.id)} onKeyDown={(e) => key(e, i)}
              className={`flex min-h-11 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors
                ${i === tabs.length - 1 && tabs.length % 2 ? 'col-span-2' : ''}
                ${on ? 'bg-slate-100 text-slate-950' : 'bg-slate-800/60 text-slate-200 hover:bg-slate-700 hover:text-white'}`}>
              {t.label}
              {t.alert && <span className="h-2 w-2 rounded-full bg-red-500" aria-label="necesită atenție" />}
            </button>
          )
        })}
      </div>
      {tabs.map((t) => (
        <div key={t.id} role="tabpanel" id={`panel-${t.id}`} aria-labelledby={`tab-${t.id}`} hidden={t.id !== active}>
          {t.panel}
        </div>
      ))}
    </div>
  )
}
