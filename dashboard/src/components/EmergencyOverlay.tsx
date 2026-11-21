import { clock } from './format'

export function EmergencyOverlay({ emergency, onClose }: { emergency: { report: string; at: number }; onClose: () => void }) {
  return (
    <div role="alertdialog" aria-labelledby="em-title" className="fixed inset-0 z-50 flex items-center justify-center bg-red-950/90 p-4">
      <div className="w-full max-w-2xl rounded-2xl border-2 border-red-500 bg-slate-950 p-6 shadow-2xl">
        <h2 id="em-title" className="text-2xl font-bold text-red-400">
          URGENȚĂ — apel 112 simulat
        </h2>
        <p className="mt-1 text-sm text-slate-400">{clock(emergency.at)} · avarii pornite (simulat)</p>
        <pre className="mt-4 whitespace-pre-wrap rounded-lg bg-slate-900 p-4 font-sans text-base">{emergency.report}</pre>
        <button onClick={onClose} autoFocus className="mt-4 rounded-lg bg-red-600 px-4 py-2 font-medium hover:bg-red-500">
          Închide
        </button>
      </div>
    </div>
  )
}
