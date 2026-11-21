// Desenul din ferestrele camerelor, din coordonatele trimise de Vision (normalizate 0..1). Fără imagine
// (privacy mode): doar punctele, ca peste imaginea din ferestrele Vision.
// Fața: o selecție de puncte (contur, ochi, nas, gură, sprâncene), fără ordinea conexiunilor.
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
const HEAD_POINTS = [0, 2, 5, 7, 8] // nas, ochi, urechi: centrul capului pentru linia posturii

const valid = (p: unknown): p is [number, number] =>
  Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])

/** Puncte valide 0..1 -> pixeli; intrările invalide se sar. */
export function toCanvas(points: unknown, w: number, h: number): [number, number][] {
  if (!Array.isArray(points)) return []
  return points.filter(valid).map(([x, y]) => [x * w, y * h])
}

/** Centrul capului (media punctelor capului valide) sau null. */
export function headCenter(body: unknown): [number, number] | null {
  const raw = Array.isArray(body) ? body : []
  const pts = HEAD_POINTS.map((i) => raw[i]).filter(valid)
  if (!pts.length) return null
  return [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length]
}

// Ordinea punctelor feței: SKELETON_IDX din Vision/vigil_vision/face/stack.py (indici MediaPipe, sortați).
const RIGHT_EYE = [33, 160, 158, 133, 153, 144]
const LEFT_EYE = [362, 385, 387, 263, 373, 380]
const NOSE_TIP = 1
export const FACE_SKELETON_IDX = [...new Set([
  ...RIGHT_EYE, ...LEFT_EYE, NOSE_TIP, 152, 10, 61, 291, 13, 14,
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
])].sort((a, b) => a - b)
const EYE_POS = [...RIGHT_EYE, ...LEFT_EYE].map((i) => FACE_SKELETON_IDX.indexOf(i))
const NOSE_POS = FACE_SKELETON_IDX.indexOf(NOSE_TIP)

/** Doar ochii și vârful nasului, ca peste imagine în fereastra FAȚA; null dacă Vision trimite alt set de puncte. */
export function eyesAndNose(face: unknown): { eyes: unknown[]; nose: unknown } | null {
  if (!Array.isArray(face) || face.length !== FACE_SKELETON_IDX.length) return null
  return { eyes: EYE_POS.map((i) => face[i]), nose: face[NOSE_POS] }
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fill()
}

/** withVideo: peste imaginea camerei desenăm doar ochii + nasul (ca Vision); fără imagine, tot conturul. */
export function drawFace(ctx: CanvasRenderingContext2D, face: unknown, closed: boolean, w: number, h: number,
  withVideo = false): void {
  ctx.clearRect(0, 0, w, h)
  const eyeColor = closed ? '#ef4444' : '#22c55e' // ca în fereastra FAȚA: verde = ochi deschiși, roșu = închiși
  const sel = withVideo ? eyesAndNose(face) : null
  if (sel) {
    ctx.fillStyle = eyeColor
    for (const [x, y] of toCanvas(sel.eyes, w, h)) dot(ctx, x, y, 3)
    ctx.fillStyle = '#3b82f6'
    for (const [x, y] of toCanvas([sel.nose], w, h)) dot(ctx, x, y, 3.5)
    return
  }
  ctx.fillStyle = eyeColor
  for (const [x, y] of toCanvas(face, w, h)) dot(ctx, x, y, 2.5)
}

export type BodyOverlay = { chestZone?: number[] | null; chestOn?: boolean; neck?: number[] | null; posture?: string | null }

/** withVideo: peste imagine doar zona pieptului și linia posturii (ca fereastra CORP); fără imagine și scheletul. */
export function drawBody(ctx: CanvasRenderingContext2D, body: unknown, o: BodyOverlay, w: number, h: number,
  withVideo = false): void {
  ctx.clearRect(0, 0, w, h)
  // zona pieptului: verde, roșie când mâna e pe piept
  const z = o.chestZone
  if (Array.isArray(z) && z.length === 4 && z.every(Number.isFinite)) {
    ctx.strokeStyle = o.chestOn ? '#ef4444' : '#22c55e'
    ctx.lineWidth = 3
    ctx.strokeRect(z[0] * w, z[1] * h, (z[2] - z[0]) * w, (z[3] - z[1]) * h)
  }
  // scheletul: indicii contează pentru conexiuni, deci nu filtrăm înainte de a uni
  const raw = Array.isArray(body) ? body : []
  if (!withVideo) {
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 2
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
    for (const [x, y] of toCanvas(raw, w, h)) ctx.fillRect(x - 2.5, y - 2.5, 5, 5)
  }
  // linia posturii: centrul capului -> baza gâtului (gri = drept, portocaliu = aplecat, roșu = prăbușit)
  const head = headCenter(raw)
  if (head && valid(o.neck)) {
    ctx.strokeStyle = o.posture === 'slumped' ? '#ef4444' : o.posture === 'leaning' ? '#f97316' : '#d4d4d8'
    ctx.lineWidth = 4
    ctx.beginPath()
    ctx.moveTo(head[0] * w, head[1] * h)
    ctx.lineTo(o.neck[0] * w, o.neck[1] * h)
    ctx.stroke()
  }
}
