import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { lastAt, linePath, nearest, runs, spread, type Pt } from '../chart'
import { SERIES_MS, type Car, type CarEvent, type CarSample, type EcuSample } from '../state'
import { Card } from './ui'

// Culorile categoriale (ordine fixă), validate pe suprafața cardului #0f172a (dark): scripts/validate_palette.js
const C = {
  car: '#3987e5', ref: '#d95926', target: '#199e70', a: '#c98500',
  grid: '#1e293b', axis: '#334155', critical: '#d03b3b', warning: '#fab219',
}
const SPEED = [
  { key: 'car', name: 'Viteza reală (ECU)', short: 'ECU', color: C.car },
  { key: 'ref', name: 'Referința ML1', short: 'ML1', color: C.ref },
  { key: 'target', name: 'Ținta fuzzy', short: 'țintă', color: C.target },
] as const
const M = { l: 48, r: 104, t: 28, b: 30 }
const BAND_LABEL_PX = 44  // o bandă mai îngustă nu își scrie numele (s-ar suprapune; legenda îl are)
const ECU_GAP_MS = 3000   // ML1 publică la 5 Hz: o pauză mai lungă = ML1 oprit
const CAR_GAP_MS = 12000  // ECU publică la 5 s când nu frânează
const EVENT_TEXT: Record<string, string> = {
  emergency_braking_started: 'frânare de urgență', emergency_cancelled: 'urgență anulată', braking_started: 'frânează',
  target_reached: 'viteză atinsă', stopped: 'oprit', hazard_on: 'avarii pornite', hazard_off: 'avarii oprite',
  resumed: 'reluat', reset: 'reset', error: 'eroare',
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(600)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Ceasul graficului: fereastra de un minut alunecă și când nu sosesc mesaje. */
function useNow(ms = 1000) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(id)
  }, [ms])
  return now
}

const fmtAgo = (now: number, t: number) => (now - t < 1500 ? 'acum' : `-${Math.round((now - t) / 1000)} s`)

/** Viteza mașinii: ce cere ML1 (fuzzy) vs ce face ECU-ul (ECU_Arduino), plus atenția șoferului, pe ultimul minut. */
export function CarChart({ ecu, car, carNow, events, online, onReset }: {
  ecu: EcuSample[]
  car: CarSample[]
  carNow: Car | null
  events: CarEvent[]
  online: boolean | null
  onReset: () => void
}) {
  const [box, width] = useWidth()
  const tick = useNow()
  const now = Math.max(tick, ecu.at(-1)?.t ?? 0, car.at(-1)?.t ?? 0)
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)

  const t0 = now - SERIES_MS
  const iw = width - M.l - M.r
  const x = (t: number) => M.l + ((t - t0) / SERIES_MS) * iw
  const tAt = (px: number) => t0 + ((px - M.l) / iw) * SERIES_MS
  const ecuIn = ecu.filter((s) => s.t >= t0)
  const carIn = car.filter((s) => s.t >= t0)
  const vmax = Math.max(120, ...ecuIn.map((s) => s.ref), ...carIn.map((s) => s.speed))
  const ymax = Math.ceil(vmax / 40) * 40
  const H1 = 300
  const y1 = (v: number) => M.t + (1 - v / ymax) * (H1 - M.t - M.b)
  const H2 = 130
  const y2 = (v: number) => M.t + (1 - v) * (H2 - M.t - M.b)

  const pts: Record<string, Pt[]> = {
    car: carIn.map((s) => ({ t: s.t, v: s.speed })),
    ref: ecuIn.map((s) => ({ t: s.t, v: s.ref })),
    target: ecuIn.map((s) => ({ t: s.t, v: s.target })),
  }
  const mrm = runs(ecuIn, (s) => s.mode === 'MRM')
  const hazards = runs(ecuIn, (s) => s.hazards && s.mode !== 'MRM')
  const ends = SPEED.map((s) => pts[s.key].filter((p) => p.v !== null).at(-1))
  const endY = spread(ends.map((p) => (p?.v != null ? y1(p.v) : -99)), 18, M.t + 4, H1 - M.b)
  const xticks = [60, 45, 30, 15, 0].map((sec) => ({ sec, px: x(now - sec * 1000) }))
  const empty = ecuIn.length === 0 && carIn.length === 0

  // citirea la poziția cursorului: toate seriile la același moment
  const hv = hover === null ? null : {
    t: hover,
    car: lastAt(car, hover, CAR_GAP_MS),
    ecu: nearest(ecu, hover, ECU_GAP_MS),
  }
  const move = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const t = tAt(e.clientX - r.left)
    setHover(t >= t0 && t <= now ? t : null)
  }
  const key = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft') setHover((h) => Math.max(t0, (h ?? now) - 5000))
    else if (e.key === 'ArrowRight') setHover((h) => Math.min(now, (h ?? now) + 5000))
    else if (e.key === 'Escape') setHover(null)
  }

  return (
    <Card title="Mașina: comanda ML1 → ECU (ultimul minut)">
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <span className="flex items-center gap-1.5">
          <span className={`h-2 w-2 rounded-full ${online ? 'bg-green-500' : online === false ? 'bg-red-500' : 'bg-slate-500'}`} />
          placa ECU {online ? 'conectată' : online === false ? 'deconectată' : '—'}
        </span>
        <span>
          <span className="text-slate-400">viteză ECU </span>
          <b className="text-lg">{carNow ? `${Math.round(carNow.speed_kmh)} km/h` : '—'}</b>
        </span>
        {carNow?.emergency && <span className="rounded bg-red-500/20 px-1.5 text-xs font-semibold text-red-300">urgență: oprire</span>}
        {carNow?.braking && <span className="rounded bg-orange-500/15 px-1.5 text-xs font-semibold text-orange-300">frânează</span>}
        {carNow?.hazard && <span className="rounded bg-yellow-400/15 px-1.5 text-xs font-semibold text-yellow-200">avarii</span>}
        {carNow?.stopped && <span className="rounded bg-slate-800 px-1.5 text-xs font-semibold text-slate-300">oprit</span>}
        <span className="ml-auto flex gap-2">
          <button type="button" onClick={() => setTable((v) => !v)}
            className="rounded border border-slate-700 px-2 py-0.5 text-xs hover:bg-slate-800">
            {table ? 'grafic' : 'tabel'}
          </button>
          <button type="button" onClick={onReset} title="ECU: viteza de croazieră, fără avarii"
            className="rounded border border-slate-700 px-2 py-0.5 text-xs hover:bg-slate-800">
            reset mașină
          </button>
        </span>
      </div>

      {/* legenda: aceeași cheie ca linia (segment colorat), textul în culoarea textului */}
      <ul className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300">
        {SPEED.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5">
            <svg width="16" height="4" aria-hidden="true"><line x1="0" y1="2" x2="16" y2="2" stroke={s.color} strokeWidth="2" /></svg>
            {s.name}
          </li>
        ))}
        <li className="flex items-center gap-1.5"><span className="h-2.5 w-3 rounded-sm" style={{ background: C.critical, opacity: 0.35 }} />MRM (oprire pe banda de urgență)</li>
        <li className="flex items-center gap-1.5"><span className="h-2.5 w-3 rounded-sm" style={{ background: C.warning, opacity: 0.3 }} />avarii</li>
      </ul>

      <div ref={box}>
      {table ? (
        <CarTable ecu={ecuIn} car={car} now={now} />
      ) : (
        <div className="relative" onPointerLeave={() => setHover(null)}>
          {empty && (
            <p className="absolute inset-x-0 top-20 text-center text-sm text-slate-400">
              aștept vigil/ecu (ML1) și vigil/car/state (ECU_Arduino)
            </p>
          )}
          <svg width={width} height={H1} role="img" tabIndex={0} onPointerMove={move} onKeyDown={key}
            onFocus={() => setHover((h) => h ?? now)} onBlur={() => setHover(null)}
            aria-label="Viteza în km/h: viteza reală raportată de ECU, referința și ținta calculate de ML1"
            className="block touch-none outline-none focus-visible:ring-2 focus-visible:ring-sky-500">
            {mrm.map(([a, b]) => (
              <g key={`m${a}`}>
                <rect x={x(a)} y={M.t} width={Math.max(2, x(b) - x(a))} height={H1 - M.t - M.b} fill={C.critical} opacity={0.18} />
                {x(b) - x(a) >= BAND_LABEL_PX && <text x={x(a) + 4} y={M.t + 11} fontSize="13" fill="#fca5a5">MRM</text>}
              </g>
            ))}
            {hazards.map(([a, b]) => (
              <g key={`h${a}`}>
                <rect x={x(a)} y={M.t} width={Math.max(2, x(b) - x(a))} height={H1 - M.t - M.b} fill={C.warning} opacity={0.12} />
                {x(b) - x(a) >= BAND_LABEL_PX && <text x={x(a) + 4} y={M.t + 11} fontSize="13" fill="#fde68a">avarii</text>}
              </g>
            ))}
            {Array.from({ length: ymax / 40 + 1 }, (_, i) => i * 40).map((v) => (
              <g key={v}>
                <line x1={M.l} x2={M.l + iw} y1={y1(v)} y2={y1(v)} stroke={v === 0 ? C.axis : C.grid} />
                <text x={M.l - 8} y={y1(v) + 4} fontSize="13" textAnchor="end" fill="#94a3b8" className="tabular-nums">{v}</text>
              </g>
            ))}
            <text x={M.l - 8} y={14} fontSize="13" textAnchor="end" fill="#94a3b8">km/h</text>
            {xticks.map(({ sec, px }) => (
              <text key={sec} x={px} y={H1 - 8} fontSize="13" textAnchor={sec === 0 ? 'end' : 'middle'} fill="#94a3b8">
                {sec === 0 ? 'acum' : `-${sec} s`}
              </text>
            ))}
            {SPEED.map((s) => (
              <path key={s.key} d={linePath(pts[s.key], x, y1, s.key === 'car' ? CAR_GAP_MS : ECU_GAP_MS, s.key === 'car')}
                fill="none" stroke={s.color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
            ))}
            {SPEED.map((s, i) => ends[i]?.v != null && (
              <text key={s.key} x={M.l + iw + 8} y={endY[i] + 5} fontSize="15" fontWeight="600" fill="#e2e8f0" className="tabular-nums">
                <tspan fill={s.color}>●</tspan> {s.short} {Math.round(ends[i]!.v!)}
              </text>
            ))}
            {hv && <line x1={x(hv.t)} x2={x(hv.t)} y1={M.t} y2={H1 - M.b} stroke="#94a3b8" strokeWidth="1" />}
          </svg>

          <div className="mt-2 text-xs text-slate-400">Atenția șoferului, a (ML1)</div>
          <svg width={width} height={H2} aria-label="Scorul de atenție a, între 0 și 1" role="img"
            onPointerMove={move} className="block touch-none">
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line x1={M.l} x2={M.l + iw} y1={y2(v)} y2={y2(v)} stroke={v === 0 ? C.axis : C.grid} />
                <text x={M.l - 8} y={y2(v) + 4} fontSize="13" textAnchor="end" fill="#94a3b8">{v}</text>
              </g>
            ))}
            <path d={linePath(ecuIn.map((s) => ({ t: s.t, v: s.a })), x, y2, ECU_GAP_MS)} fill="none" stroke={C.a}
              strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
            {hv && <line x1={x(hv.t)} x2={x(hv.t)} y1={M.t} y2={H2 - M.b} stroke="#94a3b8" strokeWidth="1" />}
          </svg>

          {hv && (hv.car || hv.ecu) && (
            <div
              className="pointer-events-none absolute top-6 z-10 min-w-44 rounded-md border border-slate-700 bg-slate-950/95 px-3 py-2 text-xs shadow-lg"
              style={x(hv.t) > width - 200 ? { right: width - x(hv.t) + 8 } : { left: x(hv.t) + 8 }}
            >
              <div className="mb-1 text-slate-400">{fmtAgo(now, hv.t)}{hv.ecu ? ` · mod ${hv.ecu.mode}` : ''}</div>
              {[
                { s: SPEED[0], v: hv.car?.speed },
                { s: SPEED[1], v: hv.ecu?.ref },
                { s: SPEED[2], v: hv.ecu?.target ?? undefined },
              ].map(({ s, v }) => (
                <div key={s.key} className="flex items-center gap-2">
                  <svg width="12" height="4" aria-hidden="true"><line x1="0" y1="2" x2="12" y2="2" stroke={s.color} strokeWidth="2" /></svg>
                  <b className="tabular-nums text-slate-100">{v == null ? '—' : `${Math.round(v)} km/h`}</b>
                  <span className="text-slate-400">{s.name}</span>
                </div>
              ))}
              <div className="flex items-center gap-2">
                <svg width="12" height="4" aria-hidden="true"><line x1="0" y1="2" x2="12" y2="2" stroke={C.a} strokeWidth="2" /></svg>
                <b className="tabular-nums text-slate-100">{hv.ecu?.a == null ? '—' : hv.ecu.a.toFixed(2)}</b>
                <span className="text-slate-400">atenție a</span>
              </div>
            </div>
          )}
        </div>
      )}
      </div>

      {events.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-xs text-slate-400">
          {events.slice(0, 4).map((e) => (
            <li key={`${e.at}-${e.event}`}>
              <span className="tabular-nums">{fmtAgo(now, e.at)}</span> · ECU:{' '}
              <span className="text-slate-200">{EVENT_TEXT[e.event] ?? e.event}</span>
              {e.speed_kmh != null && ` la ${Math.round(e.speed_kmh)} km/h`}
              {e.detail && ` (${e.detail})`}
              {e.id && <span className="text-slate-400"> · comanda {e.id}</span>}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

/** Vederea tabel (accesibilitate): aceleași valori, fără hover. */
function CarTable({ ecu, car, now }: { ecu: EcuSample[]; car: CarSample[]; now: number }) {
  const rows = ecu.filter((_, i) => i % 25 === 0 || i === ecu.length - 1).slice(-10).reverse()  // ~la 5 s
  if (!rows.length) return <p className="py-6 text-center text-sm text-slate-400">nicio dată încă</p>
  return (
    <table className="w-full text-xs tabular-nums">
      <thead className="text-slate-400">
        <tr><th className="text-left font-normal">când</th><th className="text-right font-normal">ECU km/h</th>
          <th className="text-right font-normal">ML1 km/h</th><th className="text-right font-normal">țintă</th>
          <th className="text-right font-normal">a</th><th className="text-right font-normal">mod</th></tr>
      </thead>
      <tbody>
        {rows.map((s) => {
          const c = lastAt(car, s.t, CAR_GAP_MS)
          return (
            <tr key={s.t} className="border-t border-slate-800">
              <td>{fmtAgo(now, s.t)}</td>
              <td className="text-right">{c ? Math.round(c.speed) : '—'}</td>
              <td className="text-right">{Math.round(s.ref)}</td>
              <td className="text-right">{s.target == null ? '—' : Math.round(s.target)}</td>
              <td className="text-right">{s.a == null ? '—' : s.a.toFixed(2)}</td>
              <td className="text-right">{s.mode}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
