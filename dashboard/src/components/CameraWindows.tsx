import { bodyView, faceView } from '../panels'
import { drawBody, drawFace } from '../skeleton'
import type { Vision } from '../state'
import { CameraWindow } from './CameraWindow'

/** Fereastra FAȚA (laptopul 2) și fereastra CORP (laptopul 3), ca pe laptopurile cu camere. */
export function CameraWindows({ vision }: { vision: Vision | null }) {
  const face = faceView(vision)
  const body = bodyView(vision)
  return (
    <div className="grid items-start gap-4 2xl:grid-cols-2">
      <CameraWindow view={face} draw={(ctx, w, h) => drawFace(ctx, vision?.landmarks, !!face.eyesClosed, w, h)} />
      <CameraWindow view={body} draw={(ctx, w, h) => drawBody(ctx, vision?.body_landmarks, body, w, h)} />
    </div>
  )
}
