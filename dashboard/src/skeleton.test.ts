import { describe, expect, it } from 'vitest'
import { POSE_CONNECTIONS, toCanvas } from './skeleton'

describe('toCanvas', () => {
  it('coordonate 0..1 -> pixeli', () => {
    expect(toCanvas([[0.5, 0.25]], 200, 100)).toEqual([[100, 25]])
  })

  it('date lipsă -> nimic', () => {
    expect(toCanvas(null, 10, 10)).toEqual([])
    expect(toCanvas(undefined, 10, 10)).toEqual([])
  })

  it('intrările invalide se sar', () => {
    expect(toCanvas([[1], 'x', [0.1, 0.2], [Number.NaN, 0.5]], 10, 10)).toEqual([[1, 2]])
  })
})

describe('POSE_CONNECTIONS', () => {
  it('folosesc doar primele 25 de puncte MediaPipe Pose (cât trimite Vision)', () => {
    for (const [a, b] of POSE_CONNECTIONS) {
      expect(a).toBeLessThan(25)
      expect(b).toBeLessThan(25)
    }
  })
})
