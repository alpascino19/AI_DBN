"""Netezirea landmark-urilor (MediaPipe tremură puțin de la un cadru la altul). [CV2]

OneEuroFilter (Casiez et al., 2012): netezește puternic când punctul stă pe loc
și reacționează repede când se mișcă — exact ce trebuie pentru umeri/mâini.
PoseSmoother aplică filtrul pe toate cele 33 de puncte Pose; e folosit în
pipeline imediat după MediaPipe, deci toți analizorii primesc puncte stabile.
"""
from __future__ import annotations

import math
from typing import Optional

from .context import Landmark


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self) -> None:
        self._x, self._t, self._dx = None, None, 0.0


class Ema:
    """Medie exponențială cu constantă de timp `tau` secunde."""

    def __init__(self, tau: float) -> None:
        self.tau = tau
        self.value: Optional[float] = None
        self._t: Optional[float] = None

    def __call__(self, x: float, t: float) -> float:
        if self.value is None or self._t is None or t <= self._t:
            self.value, self._t = x, t
            return x
        a = 1.0 - math.exp(-(t - self._t) / self.tau)
        self._t = t
        self.value += a * (x - self.value)
        return self.value

    def reset(self) -> None:
        self.value, self._t = None, None


class PoseSmoother:
    """Netezește x, y (One Euro) și visibility (EMA) pentru fiecare punct Pose.
    Dacă persoana dispare mai mult de `reset_after` secunde, filtrele pornesc de la zero."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.5,
                 vis_tau: float = 0.3, reset_after: float = 0.5) -> None:
        self._args = (min_cutoff, beta)
        self._vis_tau = vis_tau
        self.reset_after = reset_after
        self._fx: list[OneEuroFilter] = []
        self._fy: list[OneEuroFilter] = []
        self._fv: list[Ema] = []
        self._last_t = -1e9

    def __call__(self, pose: Optional[list[Landmark]], t: float) -> Optional[list[Landmark]]:
        if pose is None:
            return None
        if t - self._last_t > self.reset_after or len(self._fx) != len(pose):
            n = len(pose)
            self._fx = [OneEuroFilter(*self._args) for _ in range(n)]
            self._fy = [OneEuroFilter(*self._args) for _ in range(n)]
            self._fv = [Ema(self._vis_tau) for _ in range(n)]
        self._last_t = t
        return [Landmark(self._fx[i](l.x, t), self._fy[i](l.y, t), l.z, self._fv[i](l.visibility, t))
                for i, l in enumerate(pose)]
