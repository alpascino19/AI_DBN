import { useEffect, useRef, useState } from 'react'
import type { Tone, WindowView } from '../panels'

const W = 640
const H = 480

// aceleași culori ca ferestrele Vision (verde / galben / portocaliu / roșu pe nivel)
const LEVEL_BORDER = ['#16a34a', '#eab308', '#f97316', '#dc2626']
const LEVEL_BG = ['bg-green-600', 'bg-yellow-500 text-slate-950', 'bg-orange-500', 'bg-red-600']
const BADGE = { ok: 'bg-green-600', calib: 'bg-orange-500', off: 'bg-slate-400' }
const TEXT: Record<Tone, string> = {
  normal: 'text-slate-900',
  ok: 'text-green-600 font-semibold',
  warn: 'text-orange-500 font-semibold',
  bad: 'text-red-600 font-semibold',
}
const BAR: Record<Tone, string> = { normal: 'bg-slate-500', ok: 'bg-green-600', warn: 'bg-orange-500', bad: 'bg-red-600' }

/** O fereastră ca `vigil - FATA` / `vigil - CORP`: camera (imaginea, dacă Vision rulează cu --video, plus punctele),
 *  coloana cu detalii, banda de avertizări. `draw(ctx, w, h, withVideo)` desenează punctele peste. */
export function CameraWindow({ view, draw, video }: {
  view: WindowView
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number, withVideo: boolean) => void
  video?: string
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [aspect, setAspect] = useState(W / H)   // al imaginii primite (camera poate fi 16:9); punctele urmează la fel
  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (ctx) draw(ctx, W, H, !!video)
  })
  const level = Math.max(0, Math.min(3, view.level))
  const badgeCls = view.badge.tone === 'level' ? LEVEL_BG[level] : BADGE[view.badge.tone]

  return (
    <section className="overflow-hidden rounded-lg border border-slate-300 bg-white text-slate-900 shadow-lg">
      <div className="grid md:grid-cols-[minmax(0,1fr)_17rem]">
        {/* zona camerei se întinde pe toată înălțimea panoului (ca fundalul din fereastra Vision) */}
        <div className="relative bg-slate-900">
          <div className="relative" style={{ aspectRatio: video ? aspect : W / H }}>
            {video && (
              <img
                src={video}
                alt={`camera ${view.title}`}
                className="absolute inset-0 h-full w-full"
                onLoad={(e) => {
                  const { naturalWidth: nw, naturalHeight: nh } = e.currentTarget
                  if (nw && nh && Math.abs(nw / nh - aspect) > 0.01) setAspect(nw / nh)
                }}
              />
            )}
            <canvas ref={canvas} width={W} height={H} className="absolute inset-0 h-full w-full" />
          </div>
          <span className={`absolute bottom-2 left-3 text-[11px] ${video ? 'rounded bg-red-600 px-1.5 font-semibold text-white' : 'text-slate-500'}`}>
            {video ? 'LIVE' : 'privacy: doar punctele, fără imagine'}
          </span>
          {/* rama pe nivel, deasupra imaginii */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{ boxShadow: `inset 0 0 0 4px ${view.badge.tone === 'off' ? '#94a3b8' : LEVEL_BORDER[level]}` }}
          />
        </div>

        <div className="border-t border-slate-200 bg-slate-100 px-4 pb-4 pt-3 md:border-l md:border-t-0">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-lg font-bold tracking-tight">{view.title}</h2>
            <span className={`rounded px-2 py-0.5 text-sm font-bold text-white ${badgeCls}`}>{view.badge.text}</span>
          </div>
          {view.calibration !== null && (
            <div className="mt-2 h-1.5 rounded bg-slate-300">
              <div className="h-1.5 rounded bg-orange-500" style={{ width: `${Math.round(view.calibration * 100)}%` }} />
            </div>
          )}
          {view.sections.map((s) => (
            <div key={s.title} className="mt-4">
              <h3 className="border-b border-slate-300 pb-1 text-sm font-bold text-slate-500">{s.title}</h3>
              <dl className="mt-1.5 space-y-1.5">
                {s.rows.map((r) => (
                  <div key={r.label} className="grid grid-cols-[1fr_auto] items-center gap-2 text-sm">
                    <dt className="flex items-center gap-2 text-slate-500">
                      <span className="whitespace-pre">{r.label}</span>
                      {r.bar && (
                        <span className="ml-auto h-2 w-16 shrink-0 bg-slate-300">
                          <span
                            className={`block h-2 ${BAR[r.bar.tone]}`}
                            style={{ width: `${Math.round(Math.max(0, Math.min(1, r.bar.frac ?? 0)) * 100)}%` }}
                          />
                        </span>
                      )}
                    </dt>
                    <dd className={`whitespace-pre text-right tabular-nums ${TEXT[r.tone ?? 'normal']}`}>{r.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
          {!view.full && view.badge.tone !== 'off' && (
            <p className="mt-4 text-xs text-slate-400">doar câmpurile de bază (Vision nu a trimis panoul complet)</p>
          )}
        </div>
      </div>

      <div className="min-h-24 border-t border-slate-200 px-4 py-3">
        <div className="flex justify-between text-sm">
          <span className="font-bold text-slate-500">AVERTIZĂRI</span>
          {view.fps !== null && <span className="text-slate-500">{Math.round(view.fps)} fps</span>}
        </div>
        {view.badge.tone === 'off' ? (
          <p className="mt-2 text-sm text-slate-500">nu sosesc date de la această cameră</p>
        ) : view.calibration !== null ? (
          <p className="mt-2 text-sm">Calibrare: priviți drept înainte, stați normal, respirați liniștit.</p>
        ) : view.alerts.length === 0 ? (
          <p className="mt-2 font-bold text-green-600">Nicio avertizare - totul în regulă.</p>
        ) : (
          <ul className="mt-2 space-y-1">
            {view.alerts.map((a, i) => (
              <li
                key={`${a.message}-${i}`}
                className={`flex justify-between rounded px-2.5 py-1 text-sm text-white ${LEVEL_BG[Math.max(0, Math.min(3, a.level))]}`}
              >
                <span>
                  <b className="mr-3">NIVEL {a.level}</b>
                  {a.message}
                </span>
                {a.sec !== null && <span className="tabular-nums">{Math.round(a.sec)}s</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
