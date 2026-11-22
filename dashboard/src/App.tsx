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
import { SideTabs } from './components/SideTabs'
import { TestPanel } from './components/TestPanel'
import { useVigil } from './useVigil'

export default function App() {
  const { state, video, connected, broker, publish, resetAlerts, dismiss, ask, resetCar } = useVigil()
  return (
    <div className="min-h-screen">
      <Header state={state} connected={connected} broker={broker} onReset={resetAlerts} />
      <main className="grid gap-4 p-4 lg:grid-cols-[1fr_24rem]">
        <div className="grid min-w-0 content-start gap-4">
          <CameraWindows vision={state.vision} video={video} />
          <Events vision={state.vision} />
          <CarChart ecu={state.series.ecu} car={state.series.car} carNow={state.car} events={state.carEvents}
            online={state.ecuOnline} onReset={resetCar} />
        </div>
        <div className="grid min-w-0 content-start gap-4">
          <AttentionGauge state={state} />
          <SideTabs tabs={[
            { id: 'assistant', label: 'Asistent Virgil', panel: <AssistantCard assistant={state.assistant} connected={connected} onAsk={ask} /> },
            { id: 'tests', label: 'Teste & reset', panel: <TestPanel publish={publish} connected={connected} onDismiss={dismiss} /> },
            // punctul roșu: o decizie de severitate 3 sau ECU în manevra de oprire, chiar dacă tabul nu e deschis
            { id: 'decision', label: 'Decizie LLM', alert: (state.decision?.severity ?? 0) >= 3,
              panel: <DecisionCard decision={state.decision} history={state.decisions} /> },
            { id: 'ecu', label: 'ECU', alert: state.ecu?.mode === 'MRM', panel: <EcuCard ecu={state.ecu} /> },
            { id: 'responses', label: 'Răspunsul șoferului', panel: <Responses responses={state.responses} /> },
          ]} />
        </div>
      </main>
      {state.emergency && <EmergencyOverlay emergency={state.emergency} onClose={dismiss} />}
    </div>
  )
}
