// Geometria graficului vitezei (CarChart): funcții pure, testate în chart.test.ts.

export type Pt = { t: number; v: number | null }

/** Calea SVG a unei serii; o pauză mai lungă decât gapMs rupe linia (sursa s-a oprit, nu inventăm valori).
 *  step: valoarea se păstrează până la următorul punct (starea ECU vine rar, la 5 s, și nu se schimbă între). */
export function linePath(pts: Pt[], x: (t: number) => number, y: (v: number) => number, gapMs: number,
  step = false): string {
  let d = ''
  let prev: Pt | null = null
  for (const p of pts) {
    if (p.v === null) {
      prev = null
      continue
    }
    const px = x(p.t).toFixed(1)
    const py = y(p.v).toFixed(1)
    if (!prev || prev.v === null || p.t - prev.t > gapMs) d += `M${px},${py}`
    else if (step) d += `H${px}V${py}`
    else d += `L${px},${py}`
    prev = p
  }
  return d
}

/** Intervalele [start, end] în care pred e adevărat (benzile MRM / avarii). */
export function runs<T extends { t: number }>(items: T[], pred: (s: T) => boolean): [number, number][] {
  const out: [number, number][] = []
  let start: number | null = null
  let last = 0
  for (const s of items) {
    if (pred(s)) {
      if (start === null) start = s.t
      last = s.t
    } else if (start !== null) {
      out.push([start, s.t])
      start = null
    }
  }
  if (start !== null) out.push([start, last])
  return out
}

/** Punctul cel mai apropiat în timp (seriile sunt ordonate după t). */
export function nearest<T extends { t: number }>(items: T[], t: number, maxMs = Infinity): T | null {
  let best: T | null = null
  for (const s of items) {
    if (!best || Math.abs(s.t - t) < Math.abs(best.t - t)) best = s
  }
  return best && Math.abs(best.t - t) <= maxMs ? best : null
}

/** Ultimul punct de la t înapoi (pentru seriile în trepte). */
export function lastAt<T extends { t: number }>(items: T[], t: number, maxMs = Infinity): T | null {
  let best: T | null = null
  for (const s of items) {
    if (s.t > t) break
    best = s
  }
  return best && t - best.t <= maxMs ? best : null
}

/** Etichetele de la capătul liniilor: aceeași ordine, dar la cel puțin minGap px una de alta. */
export function spread(ys: number[], minGap: number, lo: number, hi: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0])
  const placed: number[] = []
  for (const [y] of order) placed.push(Math.max(y, placed.length ? placed[placed.length - 1] + minGap : lo))
  // dacă ultima a ieșit sub marginea de jos, împingem tot grupul în sus
  const over = placed.length ? placed[placed.length - 1] - hi : 0
  const shifted = placed.map((y) => Math.max(lo, y - Math.max(0, over)))
  const out = new Array<number>(ys.length)
  order.forEach(([, i], k) => (out[i] = shifted[k]))
  return out
}
