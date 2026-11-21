import { useEffect, useRef, useState } from 'react'
import type { Tone, WindowView } from '../panels'

const W = 640
const H = 480

// aceleași culori pe nivel ca ferestrele Vision (verde / galben / portocaliu / roșu), ca nuanță pe fundalul închis
const LEVEL_RING = ['#22c55e', '#eab308', '#f97316', '#ef4444']
const LEVEL_PILL = [
  'bg-green-500/15 text-green-300',
  'bg-yellow-400/15 text-yellow-200',
  'bg-orange-500/15 text-orange-300',
  'bg-red-500/20 text-red-300',
]
const BADGE = { ok: LEVEL_PILL[0], calib: LEVEL_PILL[2], off: 'bg-slate-800 text-slate-300' }
const TEXT: Record<Tone, string> = {
  normal: 'text-slate-100',
  ok: 'text-green-400 font-medium',
  warn: 'text-orange-400 font-medium',
  bad: 'text-red-400 font-medium',
}
const BAR: Record<Tone, string> = { normal: 'bg-slate-500', ok: 'bg-green-500', warn: 'bg-orange-500', bad: 'bg-red-500' }
// titlurile vin cu majuscule din panoul Vision („OCHI”); pe pagină le arătăm ca propoziție
const SENTENCE = 'lowercase first-letter:uppercase'

/** O fereastră ca `vigil - FATA` / `vigil - CORP`: camera (imaginea, dacă Vision rulează cu --video, plus punctele),
 *  dedesubt detaliile și avertizările. `draw(ctx, w, h, withVideo)` desenează punctele peste. */
export function CameraWindow({ view, draw, video, videoFps }: {
  view: WindowView
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number, withVideo: boolean) => void
  video?: string
  /** cadre/s primite de dashboard (nu cele ale camerei): arată dacă rețeaua ține pasul */
  videoFps?: number
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [aspect, setAspect] = useState(W / H)   // al imaginii primite (camera poate fi 16:9); punctele urmează la fel
  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (ctx) draw(ctx, W, H, !!video)
  })
  const level = Math.max(0, Math.min(3, view.level))
  const off = view.badge.tone === 'off'
  const badgeCls = view.badge.tone === 'level' ? LEVEL_PILL[level] : BADGE[view.badge.tone]
  const single = view.sections.length === 1

  return (
    <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
      <div className="relative bg-slate-950" style={{ aspectRatio: video ? aspect : W / H }}>
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
        <span className={`absolute bottom-2 left-3 rounded px-1.5 text-xs ${video ? 'bg-red-600 font-semibold text-white' : 'text-slate-400'}`}>
          {video ? `Live${videoFps ? ` · ${Math.round(videoFps)} fps` : ''}` : 'fără imagine: doar punctele'}
        </span>
        {/* rama pe nivel, deasupra imaginii; fără date nu are ramă */}
        {!off && (
          <div className="pointer-events-none absolute inset-0 rounded-t-xl"
            style={{ boxShadow: `inset 0 0 0 3px ${LEVEL_RING[level]}` }} />
        )}
      </div>

      <div className="px-4 pb-4 pt-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className={`text-base font-semibold ${SENTENCE}`}>{view.title}</h2>
          <span className="flex items-center gap-3">
            {view.fps !== null && <span className="text-xs tabular-nums text-slate-400">{Math.round(view.fps)} fps</span>}
            <span className={`rounded px-2 py-0.5 text-xs font-semibold ${badgeCls}`}>{view.badge.text}</span>
          </span>
        </div>
        {view.calibration !== null && (
          <div className="mt-2 h-1 rounded bg-slate-800">
            <div className="h-1 rounded bg-orange-500" style={{ width: `${Math.round(view.calibration * 100)}%` }} />
          </div>
        )}

        <div className="mt-3 grid gap-x-6 gap-y-4 sm:grid-cols-2">
          {view.sections.map((s) => (
            <div key={s.title} className={single ? 'sm:col-span-2' : ''}>
              <h3 className={`border-b border-slate-800 pb-1 text-xs font-medium text-slate-400 ${SENTENCE}`}>{s.title}</h3>
              <dl className={`mt-1.5 gap-x-6 space-y-1 ${single ? 'sm:columns-2' : ''}`}>
                {s.rows.map((r) => (
                  <div key={r.label} className="grid break-inside-avoid grid-cols-[1fr_auto] items-center gap-2 text-sm">
                    <dt className="flex items-center gap-2 text-slate-400">
                      <span className="whitespace-pre">{r.label}</span>
                      {r.bar && (
                        <span className="ml-auto h-1.5 w-14 shrink-0 rounded-full bg-slate-800">
                          <span
                            className={`block h-1.5 rounded-full ${BAR[r.bar.tone]}`}
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
        </div>
        {!view.full && !off && (
          <p className="mt-3 text-xs text-slate-400">doar câmpurile de bază (Vision nu a trimis panoul complet)</p>
        )}

        <div className="mt-4 border-t border-slate-800 pt-3">
          <h3 className="text-xs font-medium text-slate-400">Avertizări</h3>
          {off ? (
            <p className="mt-1.5 text-sm text-slate-400">nu sosesc date de la această cameră</p>
          ) : view.calibration !== null ? (
            <p className="mt-1.5 text-sm">Calibrare: priviți drept înainte, stați normal, respirați liniștit.</p>
          ) : view.alerts.length === 0 ? (
            <p className="mt-1.5 text-sm text-green-400">Nicio avertizare, totul în regulă.</p>
          ) : (
            <ul className="mt-1.5 space-y-1">
              {view.alerts.map((a, i) => (
                <li key={`${a.message}-${i}`}
                  className={`flex justify-between gap-3 rounded px-2.5 py-1 text-sm ${LEVEL_PILL[Math.max(0, Math.min(3, a.level))]}`}>
                  <span><b className="mr-2 font-semibold">Nivel {a.level}</b>{a.message}</span>
                  {a.sec !== null && <span className="tabular-nums">{Math.round(a.sec)}s</span>}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  )
}
