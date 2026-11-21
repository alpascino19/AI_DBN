// Mesajele panoului „Teste & reset”: funcții pure (testate în controls.test.ts), aceleași contracte ca restul
// sistemului (ML2/executor.py validează deciziile, ECU_Arduino înțelege comenzile de pe vigil/car/cmd).

export type Publish = (topic: string, payload: string | object, opts?: { qos?: 0 | 1; retain?: boolean }) => unknown

/** ESP32: nivelurile de alertă pe rând, apoi stins (reținut, ca la ML2). */
export const ALERT_STEPS = [1, 2, 3, 0] as const
export const ALERT_STEP_MS = 2500

/** Test voce: o decizie de severitate 1 cu o frază; ML2 o rostește (debounce ML2: o dată la 10 s). */
export function voiceTest(now = Date.now()) {
  return {
    severity: 1, source: 'test', reason: 'TEST din dashboard: vocea ML2', ts: now / 1000,
    actions: [{ tool: 'speak', args: { text: 'Test de voce Vigil. Sistemul vă aude și vă poate vorbi.' } }],
  }
}

/** Test 112: severitate 3 cu call_112. ML2 rulează apelul SIMULAT (raport în ML2/reports + voce), iar
 *  dashboard-ul deschide fereastra roșie cu raportul. Nimic nu pleacă de pe laptop. */
export function test112(now = Date.now()) {
  const when = new Date(now).toLocaleTimeString('ro-RO')
  return {
    severity: 3, source: 'test', reason: 'TEST din dashboard: apelul 112 simulat', ts: now / 1000,
    actions: [{
      tool: 'call_112',
      args: { report: `TEST din dashboard (${when}) - NU este o urgență reală.\nVerifică lanțul ML2: apel 112 simulat, ` +
        'raport salvat, mesaj vocal, fereastra de urgență din dashboard.' },
    }],
  }
}

/** Revenire la normal pentru ML2 (stinge ESP32), după testul de 112. */
export function calmDecision(now = Date.now()) {
  return { severity: 0, source: 'test', reason: 'TEST din dashboard: revenire la normal', ts: now / 1000, actions: [] }
}

/** Comenzile plăcii ECU (ECU_Arduino), una câte una. */
export const ECU_TESTS = [
  { key: 'emergency', label: 'Frânare de urgență', cmd: { cmd: 'emergency', decel_mps2: 2.5 } },
  { key: 'slow60', label: 'Încetinește la 60', cmd: { cmd: 'slow_down', target_kmh: 60, decel_mps2: 2.0 } },
  { key: 'slow30', label: 'Încetinește la 30', cmd: { cmd: 'slow_down', target_kmh: 30, decel_mps2: 2.0 } },
  { key: 'hazard_on', label: 'Avarii pornite', cmd: { cmd: 'hazard', on: true } },
  { key: 'hazard_off', label: 'Avarii oprite', cmd: { cmd: 'hazard', on: false } },
  { key: 'cancel', label: 'Anulează urgența', cmd: { cmd: 'cancel_emergency' } },
  { key: 'resume', label: 'Reia (fără frânare)', cmd: { cmd: 'resume' } },
] as const

export const RESET = {
  alerts: (p: Publish) => p('vigil/alert', '0', { retain: true }),
  car: (p: Publish, id: string) => p('vigil/car/cmd', { cmd: 'reset', id }),
  ml1: (p: Publish, id: string) => p('vigil/ml1/cmd', { cmd: 'reset', id }),
}

/** Reset complet: ESP32 stins, mașina la croazieră, ML1 de la zero (calibrare nouă), ML2 la normal. */
export function resetAll(p: Publish, id: string, now = Date.now()) {
  RESET.alerts(p)
  RESET.car(p, id)
  RESET.ml1(p, id)
  p('vigil/decision', calmDecision(now))
}
