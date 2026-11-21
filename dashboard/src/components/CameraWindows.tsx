import { bodyView, faceView } from '../panels'
import { drawBody, drawFace } from '../skeleton'
import type { Vision } from '../state'
import type { VideoFrames } from '../useVigil'
import { CameraWindow } from './CameraWindow'

/** Fereastra FAȚA (laptopul 2) și fereastra CORP (laptopul 3), ca pe laptopurile cu camere.
 *  video: ultimul cadru al fiecărei camere (useVigil îl scoate singur după 3 s fără cadre noi). */
export function CameraWindows({ vision, video = {} }: { vision: Vision | null; video?: VideoFrames }) {
  const face = faceView(vision)
  const body = bodyView(vision)
  return (
    <div className="grid items-start gap-4 2xl:grid-cols-2">
      <CameraWindow
        view={face}
        video={video.face?.url}
        draw={(ctx, w, h, withVideo) => drawFace(ctx, vision?.landmarks, !!face.eyesClosed, w, h, withVideo)}
      />
      <CameraWindow
        view={body}
        video={video.body?.url}
        draw={(ctx, w, h, withVideo) => drawBody(ctx, vision?.body_landmarks, body, w, h, withVideo)}
      />
    </div>
  )
}
