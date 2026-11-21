import { AssistantCard } from './components/AssistantCard'
import { AttentionGauge } from './components/AttentionGauge'
import { CameraWindows } from './components/CameraWindows'
import { CarChart } from './components/CarChart'
import { DecisionCard } from './components/DecisionCard'
import { EcuCard } from './components/EcuCard'
import { EmergencyOverlay } from './components/EmergencyOverlay'
import { Events } from './components/Events'
import { Header } from './components/Header'
import { Responses } from './components/Responses'
import { useVigil } from './useVigil'

export default function App() {
  const { state, video, connected, broker, resetAlerts, dismiss, ask, resetCar } = useVigil()
  return (
    <div className="min-h-screen">
      <Header state={state} connected={connected} broker={broker} onReset={resetAlerts} />
      <main className="grid gap-4 p-4 lg:grid-cols-[1fr_22rem]">
        <div className="grid content-start gap-4">
          <CameraWindows vision={state.vision} video={video} />
          <CarChart ecu={state.series.ecu} car={state.series.car} carNow={state.car} events={state.carEvents}
            online={state.ecuOnline} onReset={resetCar} />
          <div className="grid gap-4 md:grid-cols-[12rem_1fr]">
            <AttentionGauge state={state} />
            <Events vision={state.vision} />
          </div>
        </div>
        <div className="grid content-start gap-4">
          <AssistantCard assistant={state.assistant} connected={connected} onAsk={ask} />
          <DecisionCard decision={state.decision} history={state.decisions} />
          <EcuCard ecu={state.ecu} />
          <Responses responses={state.responses} />
        </div>
      </main>
      {state.emergency && <EmergencyOverlay emergency={state.emergency} onClose={dismiss} />}
    </div>
  )
}
