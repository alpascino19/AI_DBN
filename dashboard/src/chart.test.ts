import { describe, expect, it } from 'vitest'
import { lastAt, linePath, nearest, runs, spread } from './chart'
import { eyesAndNose, FACE_SKELETON_IDX } from './skeleton'

const id = (v: number) => v

describe('linePath', () => {
  it('linie, trepte și pauză', () => {
    const pts = [{ t: 0, v: 10 }, { t: 1, v: 20 }, { t: 10, v: 30 }]
    expect(linePath(pts, id, id, 5)).toBe('M0.0,10.0L1.0,20.0M10.0,30.0')     // pauza de 9 > 5 rupe linia
    expect(linePath(pts.slice(0, 2), id, id, 5, true)).toBe('M0.0,10.0H1.0V20.0')
    expect(linePath([{ t: 0, v: null }, { t: 1, v: 5 }], id, id, 5)).toBe('M1.0,5.0')
  })
})

describe('runs', () => {
  it('benzile MRM', () => {
    const s = [{ t: 0, m: 0 }, { t: 1, m: 1 }, { t: 2, m: 1 }, { t: 3, m: 0 }, { t: 4, m: 1 }]
    expect(runs(s, (x) => x.m === 1)).toEqual([[1, 3], [4, 4]])
  })
})

describe('nearest / lastAt', () => {
  const s = [{ t: 0 }, { t: 10 }, { t: 20 }]
  it('cel mai apropiat și ultimul înainte', () => {
    expect(nearest(s, 12)?.t).toBe(10)
    expect(nearest(s, 100, 5)).toBeNull()
    expect(lastAt(s, 19)?.t).toBe(10)
    expect(lastAt(s, -1)).toBeNull()
    expect(lastAt(s, 40, 5)).toBeNull()
  })
})

describe('spread', () => {
  it('etichetele nu se suprapun și rămân în ordine', () => {
    const out = spread([50, 50, 51], 12, 0, 200)
    const sorted = [...out].sort((a, b) => a - b)
    expect(sorted[1] - sorted[0]).toBeGreaterThanOrEqual(12)
    expect(sorted[2] - sorted[1]).toBeGreaterThanOrEqual(12)
    expect(spread([195, 198], 12, 0, 200).every((y) => y <= 200)).toBe(true)
  })
})

describe('punctele feței peste imagine', () => {
  it('același set ca SKELETON_IDX din Vision; ochii și nasul se găsesc', () => {
    expect(FACE_SKELETON_IDX.length).toBe(63)            // = len(SKELETON_IDX) în Vision/vigil_vision/face/stack.py
    const face = FACE_SKELETON_IDX.map((i) => [i / 1000, 0])
    const sel = eyesAndNose(face)!
    expect(sel.eyes.map((p) => Math.round((p as number[])[0] * 1000))).toEqual([33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380])
    expect(Math.round((sel.nose as number[])[0] * 1000)).toBe(1)
    expect(eyesAndNose(face.slice(1))).toBeNull()      // alt set de puncte: desenăm tot, nu ghicim
  })
})
