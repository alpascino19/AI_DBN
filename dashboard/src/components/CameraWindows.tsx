import { useCallback, useSyncExternalStore } from 'react'
import { bodyView, faceView, type WindowView } from '../panels'
import { drawBody, drawFace } from '../skeleton'
import type { Vision } from '../state'
import type { Cam, VideoStore } from '../videoStore'
import { CameraWindow } from './CameraWindow'

type Draw = (ctx: CanvasRenderingContext2D, w: number, h: number, withVideo: boolean) => void

/** O fereastră abonată doar la imaginea camerei ei: un cadru nou re-randează doar fereastra asta. */
function LiveWindow({ cam, video, view, draw }: { cam: Cam; video?: VideoStore; view: WindowView; draw: Draw }) {
  const subscribe = useCallback((fn: () => void) => (video ? video.subscribe(cam, fn) : () => {}), [video, cam])
  const snapshot = useCallback(() => video?.get(cam), [video, cam])
  const frame = useSyncExternalStore(subscribe, snapshot)
  return <CameraWindow view={view} draw={draw} video={frame?.url} videoFps={frame?.fps} />
}

/** Fereastra FAȚA (laptopul 2) și fereastra CORP (laptopul 3), ca pe laptopurile cu camere. */
export function CameraWindows({ vision, video }: { vision: Vision | null; video?: VideoStore }) {
  const face = faceView(vision)
  const body = bodyView(vision)
  return (
    <div className="grid items-start gap-4 xl:grid-cols-2">
      <LiveWindow cam="face" video={video} view={face}
        draw={(ctx, w, h, withVideo) => drawFace(ctx, vision?.landmarks, !!face.eyesClosed, w, h, withVideo)} />
      <LiveWindow cam="body" video={video} view={body}
        draw={(ctx, w, h, withVideo) => drawBody(ctx, vision?.body_landmarks, body, w, h, withVideo)} />
    </div>
  )
}
