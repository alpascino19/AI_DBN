import { AttentionGauge } from './components/AttentionGauge'
import { CameraPanel } from './components/CameraPanel'
import { DecisionCard } from './components/DecisionCard'
import { EcuCard } from './components/EcuCard'
import { EmergencyOverlay } from './components/EmergencyOverlay'
import { Events } from './components/Events'
import { Header } from './components/Header'
import { Responses } from './components/Responses'
import { Signals } from './components/Signals'
import { useVigil } from './useVigil'

export default function App() {
  const { state, connected, broker, resetAlerts, dismiss } = useVigil()
  const v = state.vision
  return (
    <div className="min-h-screen">
      <Header state={state} connected={connected} broker={broker} onReset={resetAlerts} />
      <main className="grid gap-4 p-4 lg:grid-cols-[1fr_22rem]">
        <div className="grid gap-4">
          <div className="grid gap-4 md:grid-cols-[12rem_1fr_1fr]">
            <AttentionGauge state={state} />
            <CameraPanel title="Camera față" face={v?.landmarks} detected={v?.face_detected} />
            <CameraPanel title="Camera corp" body={v?.body_landmarks} detected={v?.body_detected} />
          </div>
          <div className="grid gap-4 md:grid-cols-[1fr_18rem]">
            <Signals vision={v} />
            <Events vision={v} />
          </div>
        </div>
        <div className="grid content-start gap-4">
          <DecisionCard decision={state.decision} history={state.decisions} />
          <EcuCard ecu={state.ecu} />
          <Responses responses={state.responses} />
        </div>
      </main>
      {state.emergency && <EmergencyOverlay emergency={state.emergency} onClose={dismiss} />}
    </div>
  )
}
