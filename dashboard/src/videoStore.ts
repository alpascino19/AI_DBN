// Cadrele live ale camerelor (vigil/video/face, vigil/video/body), ținute în afara stării React: la 2 x 20
// cadre/s, un useState ar re-randa tot dashboard-ul (inclusiv graficul) de 40 de ori pe secundă. Fiecare
// fereastră se abonează doar la camera ei (useSyncExternalStore), deci un cadru re-randează doar fereastra lui.

export type Cam = 'face' | 'body'
/** Ultimul cadru: URL de obiect (blob:), momentul sosirii, cadre/s primite (ultimele 2 s). */
export type VideoFrame = { url: string; at: number; fps: number }

export const VIDEO_STALE_MS = 3000   // fără cadru nou atât timp: camera s-a oprit, fereastra revine la puncte
const FPS_WINDOW_MS = 2000
const REVOKE_AFTER_MS = 1000         // cadrul vechi se eliberează după ce <img> a trecut la cel nou

export class VideoStore {
  private frames: Partial<Record<Cam, VideoFrame>> = {}
  private listeners: Record<Cam, Set<() => void>> = { face: new Set(), body: new Set() }
  private arrivals: Record<Cam, number[]> = { face: [], body: [] }
  private stale: Partial<Record<Cam, ReturnType<typeof setTimeout>>> = {}
  private makeUrl: (data: Uint8Array) => string
  private revoke: (url: string) => void

  /** makeUrl / revoke: înlocuibile în teste (implicit URL-uri blob: din browser). */
  constructor(
    makeUrl: (data: Uint8Array) => string = (d) => URL.createObjectURL(new Blob([d as BlobPart], { type: 'image/jpeg' })),
    revoke: (url: string) => void = (u) => URL.revokeObjectURL(u),
  ) {
    this.makeUrl = makeUrl
    this.revoke = revoke
  }

  get = (cam: Cam): VideoFrame | undefined => this.frames[cam]

  subscribe = (cam: Cam, fn: () => void): (() => void) => {
    this.listeners[cam].add(fn)
    return () => this.listeners[cam].delete(fn)
  }

  /** Un cadru JPEG nou pentru camera `cam`. */
  push(cam: Cam, data: Uint8Array, at = Date.now()): void {
    const old = this.frames[cam]?.url
    const recent = (this.arrivals[cam] = [...this.arrivals[cam].filter((x) => at - x < FPS_WINDOW_MS), at])
    const fps = recent.length > 1 ? ((recent.length - 1) * 1000) / (at - recent[0]) : 0
    this.frames = { ...this.frames, [cam]: { url: this.makeUrl(data), at, fps } }
    this.emit(cam)
    if (old) setTimeout(() => this.revoke(old), REVOKE_AFTER_MS)
    clearTimeout(this.stale[cam])
    this.stale[cam] = setTimeout(() => this.drop(cam), VIDEO_STALE_MS)
  }

  /** Camera nu mai trimite: fereastra revine la puncte. */
  drop(cam: Cam): void {
    const f = this.frames[cam]
    if (!f) return
    const { [cam]: _gone, ...rest } = this.frames
    this.frames = rest
    this.arrivals[cam] = []
    this.emit(cam)
    setTimeout(() => this.revoke(f.url), REVOKE_AFTER_MS)
  }

  dispose(): void {
    for (const cam of ['face', 'body'] as Cam[]) {
      clearTimeout(this.stale[cam])
      const f = this.frames[cam]
      if (f) this.revoke(f.url)
    }
    this.frames = {}
  }

  private emit(cam: Cam) {
    for (const fn of this.listeners[cam]) fn()
  }
}
