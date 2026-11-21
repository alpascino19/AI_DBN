// Desenul scheletului din coordonatele trimise de Vision (normalizate 0..1).
// Fața: puncte (Vision trimite o selecție de puncte, fără ordinea conexiunilor).
// Corpul: primele 25 de puncte MediaPipe Pose, deci se pot uni cu liniile standard.

export const POSE_CONNECTIONS: [number, number][] = [
  // față
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  // umeri, brațe, mâini
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 17], [15, 19], [15, 21], [17, 19], [16, 18], [16, 20], [16, 22], [18, 20],
  // trunchi
  [11, 23], [12, 24], [23, 24],
]

const valid = (p: unknown): p is [number, number] =>
  Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])

/** Puncte valide 0..1 -> pixeli; intrările invalide se sar. */
export function toCanvas(points: unknown, w: number, h: number): [number, number][] {
  if (!Array.isArray(points)) return []
  return points.filter(valid).map(([x, y]) => [x * w, y * h])
}

export function drawSkeleton(ctx: CanvasRenderingContext2D, face: unknown, body: unknown, w: number, h: number): void {
  ctx.clearRect(0, 0, w, h)

  // corpul: indicii contează pentru conexiuni, deci nu filtrăm înainte de a uni
  const raw = Array.isArray(body) ? body : []
  ctx.strokeStyle = '#38bdf8'
  ctx.lineWidth = 3
  ctx.beginPath()
  for (const [a, b] of POSE_CONNECTIONS) {
    const p = raw[a]
    const q = raw[b]
    if (!valid(p) || !valid(q)) continue
    ctx.moveTo(p[0] * w, p[1] * h)
    ctx.lineTo(q[0] * w, q[1] * h)
  }
  ctx.stroke()
  ctx.fillStyle = '#7dd3fc'
  for (const [x, y] of toCanvas(raw, w, h)) ctx.fillRect(x - 3, y - 3, 6, 6)

  ctx.fillStyle = '#34d399'
  for (const [x, y] of toCanvas(face, w, h)) {
    ctx.beginPath()
    ctx.arc(x, y, 2, 0, Math.PI * 2)
    ctx.fill()
  }
}
