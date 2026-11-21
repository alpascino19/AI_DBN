"""Pierderea cunoștinței din semnalele feței: cap căzut + ochi închiși + nemișcare. [CV1]

Un om care leșină la volan:
    1. capul CADE: în față (pitch mare în jos) sau într-o parte (roll mare),
       de obicei BRUSC (viteză mare a unghiului), nu lent ca la privitul în jos;
    2. ochii sunt ÎNCHIȘI prelungit;
    3. NU SE MAI MIȘCĂ: capul stă nemișcat câteva secunde;
    4. adesea fața dispare din cadru (cade pe volan) -> păstrăm starea `hold_sec`
       (CV2 / body/slump.py vede atunci corpul prăbușit).

Decizia e un SCOR 0..1, nu "și/și/și":
    cap căzut   0.35   (pitch / roll peste prag, rampă)
    ochi închiși 0.35  (closed_sec, rampă 1 -> 3 s)
    nemișcare   0.20   (still_sec, rampă 1 -> 3 s)
    cădere bruscă 0.10 (viteza pitch/roll în ultimele 2 s)
Inconștient = scor >= enter (0.65); iese sub exit (0.40). Fără ochi închiși
scorul nu trece de ~0.65, deci o privire în jos cu ochii deschiși nu ajunge acolo.

Utilizare:
    unc = UnconsciousAnalyzer()
    st = unc.update(eye_state, head_state, t)
    st.state ("ok" / "suspect" / "unconscious"), st.score, st.head_drop
    unc.detections()   # {"head_drop": (..), "unconscious": (..)}
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .eyes import EyeState
from .head_pose import HeadState

OK, SUSPECT, UNCONSCIOUS = "ok", "suspect", "unconscious"


@dataclass
class UnconsciousConfig:
    pitch_drop: tuple[float, float] = (25.0, 45.0)   # grade în jos față de neutru: [începe, maxim]
    roll_drop: tuple[float, float] = (25.0, 45.0)    # grade înclinat lateral
    closed_ramp: tuple[float, float] = (1.0, 3.0)    # secunde cu ochii închiși
    still_ramp: tuple[float, float] = (1.0, 3.0)     # secunde nemișcat
    still_deg: float = 3.0           # variația unghiurilor sub atât (pe 1 s) = nemișcat
    sudden_deg_per_sec: float = 40.0 # viteză de cădere peste atât = bruscă
    sudden_window_sec: float = 2.0   # cât timp ținem minte o cădere bruscă
    head_drop_closed_sec: float = 1.0  # head_drop = cap căzut + ochi închiși atâtea secunde
    enter: float = 0.65
    exit: float = 0.40
    suspect: float = 0.45
    hold_sec: float = 10.0           # fața dispare după leșin -> păstrăm starea


@dataclass
class UnconsciousState:
    state: str = OK
    score: float = 0.0
    head_drop: bool = False          # cap căzut + ochi închiși (eveniment pentru ML1)
    dropped: bool = False            # doar capul căzut (fără condiția ochilor)
    sudden: bool = False             # căderea a fost bruscă
    still_sec: float = 0.0
    since_sec: float = 0.0           # de cât timp e "unconscious"
    parts: Optional[dict] = None     # contribuția fiecărui semnal (pentru reglaj)

    def to_dict(self) -> dict:
        return asdict(self)


def _ramp(x: float, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class UnconsciousAnalyzer:
    def __init__(self, config: Optional[UnconsciousConfig] = None) -> None:
        self.cfg = config or UnconsciousConfig()
        self.state = UnconsciousState()
        self._hist: deque[tuple[float, float, float]] = deque()   # (t, pitch, roll) ~2 s
        self._still_since: Optional[float] = None
        self._sudden_at: Optional[float] = None
        self._uncon_since: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._roll0: Optional[float] = None                         # roll neutru (învățat lent)

    def update(self, eyes: EyeState, head: HeadState, t: float) -> UnconsciousState:
        c, s = self.cfg, self.state

        if not head.face_visible or head.pitch is None:
            # fața a dispărut: dacă era inconștient, păstrăm starea hold_sec
            if s.state == UNCONSCIOUS and self._last_seen is not None and t - self._last_seen <= c.hold_sec:
                s.since_sec = round(t - (self._uncon_since or t), 1)
                return s
            self._reset_motion()
            self._set(OK, 0.0, t)
            s.head_drop = s.dropped = s.sudden = False
            s.still_sec = 0.0
            return s
        self._last_seen = t

        pitch = float(head.pitch)
        roll = float(head.roll or 0.0)
        # roll neutru: oamenii țin capul ușor înclinat; îl învățăm lent cât timp nu e căzut
        if self._roll0 is None:
            self._roll0 = roll
        elif abs(roll - self._roll0) < 10:
            self._roll0 += 0.01 * (roll - self._roll0)
        roll_rel = abs(roll - self._roll0)

        # ----- istoric 2 s: viteză și nemișcare -----
        self._hist.append((t, pitch, roll))
        while self._hist and self._hist[0][0] < t - c.sudden_window_sec:
            self._hist.popleft()
        recent = [(tt, p, r) for tt, p, r in self._hist if tt >= t - 1.0]
        if len(recent) >= 3:
            spread = max(np.ptp([p for _, p, _ in recent]), np.ptp([r for _, _, r in recent]))
        else:
            spread = 99.0
        if spread < c.still_deg:
            self._still_since = self._still_since if self._still_since is not None else t
        else:
            self._still_since = None
        s.still_sec = 0.0 if self._still_since is None else round(t - self._still_since, 2)

        # viteza pe ultima ~0.5 s (o cădere durează sub o secundă)
        past = next(((tt, p, r) for tt, p, r in self._hist if tt >= t - 0.5), None)
        if past is not None and t - past[0] >= 0.15:
            t0, p0, r0 = past
            dt = max(1e-3, t - t0)
            speed = max(pitch - p0, abs(roll - r0)) / dt     # doar căderea (în jos / lateral)
            if speed > c.sudden_deg_per_sec:
                self._sudden_at = t
        s.sudden = self._sudden_at is not None and t - self._sudden_at <= c.sudden_window_sec + c.still_ramp[1]

        # ----- scor -----
        drop = max(_ramp(pitch, c.pitch_drop), _ramp(roll_rel, c.roll_drop))
        closed = _ramp(eyes.closed_sec if eyes.eyes_closed else 0.0, c.closed_ramp)
        still = _ramp(s.still_sec, c.still_ramp)
        score = 0.35 * drop + 0.35 * closed + 0.20 * still + 0.10 * float(s.sudden)
        s.parts = {"drop": round(drop, 2), "closed": round(closed, 2), "still": round(still, 2),
                   "sudden": s.sudden}
        s.dropped = drop > 0.0
        s.head_drop = s.dropped and eyes.eyes_closed and eyes.closed_sec >= c.head_drop_closed_sec

        if s.state == UNCONSCIOUS:
            new = UNCONSCIOUS if score >= c.exit else (SUSPECT if score >= c.suspect else OK)
        else:
            new = UNCONSCIOUS if score >= c.enter else (SUSPECT if score >= c.suspect else OK)
        self._set(new, score, t)
        return s

    def detections(self) -> dict[str, tuple[bool, float]]:
        s = self.state
        return {
            "head_drop": (s.head_drop, round(max(0.6, s.score), 2) if s.head_drop else 0.0),
            "unconscious": (s.state == UNCONSCIOUS, round(s.score, 2) if s.state == UNCONSCIOUS else 0.0),
        }

    # ---------- intern ----------
    def _set(self, state: str, score: float, t: float) -> None:
        s = self.state
        if state == UNCONSCIOUS and s.state != UNCONSCIOUS:
            self._uncon_since = t
        elif state != UNCONSCIOUS:
            self._uncon_since = None
        s.state, s.score = state, round(score, 3)
        s.since_sec = 0.0 if self._uncon_since is None else round(t - self._uncon_since, 1)

    def _reset_motion(self) -> None:
        self._hist.clear()
        self._still_since = None
        self._sudden_at = None
