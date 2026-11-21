import { useEffect, useRef } from 'react'
import { drawSkeleton } from '../skeleton'
import { Card } from './ui'

const W = 640
const H = 480

// Imaginea camerei va sta sub canvas când o va trimite CV; acum se vede doar scheletul.
export function CameraPanel({ title, face, body, detected }: { title: string; face?: unknown; body?: unknown; detected?: unknown }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (ctx) drawSkeleton(ctx, face, body, W, H)
  }, [face, body])

  return (
    <Card title={title}>
      <div className="relative aspect-[4/3] overflow-hidden rounded-lg bg-slate-950">
        <span className="absolute inset-0 flex items-center justify-center text-sm text-slate-700">imagine: în curând</span>
        <canvas ref={canvas} width={W} height={H} className="absolute inset-0 h-full w-full" />
        {detected === false && (
          <span className="absolute left-2 top-2 rounded bg-slate-800/80 px-2 py-0.5 text-xs text-orange-300">nedetectat</span>
        )}
      </div>
    </Card>
  )
}
