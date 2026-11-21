"""Avertizări (alerte) locale din semnalele feței. [CV1]

Reguli simple, cu durată minimă, repetare și cooldown, peste EyeState + HeadState:
    ochi închiși prea mult, privire stânga/dreapta/jos prea mult, distragere
    (priviri dese în afara drumului), oboseală (PERCLOS / clipit rar), fără față.

Rolul lor: feedback imediat pentru demo și plan B dacă LLM-ul (ML1) e lent.
Decizia "oficială" rămâne la LLM; avertizările pot fi trimise și ele în `vigil/vision`.

Utilizare:
    am = AlertManager()
    for w in am.update(eye_state, head_state, eyes.detections(), t):
        print(w.level, w.message)      # doar avertizările NOI / repetate
    am.active                          # toate cele active acum
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .eyes import EyeState
from .head_pose import CENTER, DOWN, LEFT, RIGHT, HeadState


@dataclass
class Alert:
    code: str
    level: int          # 1 = redusă, 2 = medie, 3 = ridicată (ca în TASKS.md)
    message: str
    active_sec: float   # de cât timp e adevărată condiția

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Rule:
    code: str
    level: int
    message: str
    condition: Callable[["Signals"], bool]
    min_sec: float = 0.0        # condiția trebuie să țină atât până avertizăm
    repeat_sec: float = 5.0     # cât de des repetăm cât timp e activă
    cooldown_sec: float = 0.0   # după ce se oprește, cât așteptăm înainte de o nouă avertizare
    hides: tuple[str, ...] = () # când e activă, ascunde aceste avertizări (mai puțin grave / redundante)


@dataclass
class Signals:
    eyes: EyeState
    head: HeadState
    detections: dict[str, tuple[bool, float]]

    def det(self, name: str) -> bool:
        return self.detections.get(name, (False, 0.0))[0]


def default_rules() -> list[Rule]:
    return [
        # --- ochi ---
        Rule("eyes_closed_critical", 3, "ALERTĂ: ochii sunt închiși de prea mult timp! Opriți în siguranță!",
             lambda s: s.eyes.eyes_closed and s.eyes.closed_sec >= 3.0, repeat_sec=2.0, hides=("eyes_closed",)),
        Rule("eyes_closed", 2, "Ochii sunt închiși! Treziți-vă și priviți la drum!",
             # la căscat ochii se strâng natural: nu dăm alarmă de nivel 2 (critica rămâne)
             # privind în jos (telefon) pleoapele coboară și par închise: acolo avertizăm "privire în jos"
             lambda s: s.eyes.eyes_closed and s.eyes.closed_sec >= 1.5 and not s.det("yawning")
             and s.head.direction != DOWN,
             repeat_sec=3.0),
        Rule("fatigue", 2, "Semne de oboseală. Vă recomandăm o pauză.",
             # declanșare pe ultimele 60 s; se stinge când ultimele 15 s arată normal
             lambda s: ((s.eyes.perclos or 0) >= 0.15 or s.det("low_blink_rate")) and not s.eyes.recovered,
             min_sec=5.0, repeat_sec=60.0, cooldown_sec=60.0),
        Rule("hyper_blinking", 2, "Clipiți foarte des. Vă simțiți bine?",
             lambda s: s.det("hyper_blinking"), min_sec=5.0, repeat_sec=30.0, cooldown_sec=30.0),
        # --- căscat (detecții din yawn.py) ---
        Rule("frequent_yawning", 2, "Căscați des. Vă recomandăm o pauză.",
             lambda s: s.det("frequent_yawning"), repeat_sec=120.0, cooldown_sec=60.0),
        Rule("yawn", 1, "Ați căscat: semn de oboseală. Rămâneți atent la drum.",
             lambda s: s.det("yawning"), repeat_sec=600.0, cooldown_sec=10.0),  # o dată pe căscat
        # --- privire ---
        Rule("look_left", 1, "Priviți spre stânga de prea mult timp. Vă rugăm să priviți la drum.",
             lambda s: s.head.direction == LEFT, min_sec=2.0),
        Rule("look_right", 1, "Priviți spre dreapta de prea mult timp. Vă rugăm să priviți la drum.",
             lambda s: s.head.direction == RIGHT, min_sec=2.0),
        Rule("look_down", 1, "Priviți în jos (telefon?). Vă rugăm să priviți la drum.",
             lambda s: s.head.direction == DOWN, min_sec=2.0),
        Rule("distracted", 2, "Sunteți distras: priviți des în afara drumului.",
             lambda s: (s.head.off_road_ratio or 0) >= 0.5, min_sec=2.0, repeat_sec=15.0, cooldown_sec=10.0),
        Rule("no_face", 1, "Nu vă pot vedea fața. Vă rugăm să priviți spre drum.",
             lambda s: not s.head.face_visible, min_sec=3.0, repeat_sec=10.0),
        # --- pierderea cunoștinței (face/unconscious.py) ---
        Rule("unconscious", 3, "URGENȚĂ: posibilă pierdere a cunoștinței! Răspundeți!",
             # telefonul în mână explică privirea în jos: nu e leșin
             lambda s: s.det("unconscious") and not s.det("phone_use"), repeat_sec=3.0,
             hides=("eyes_closed_critical", "eyes_closed", "head_drop", "look_down", "no_face",
                    "distracted", "fatigue")),
        Rule("head_drop", 2, "Capul a căzut și ochii sunt închiși. Sunteți bine?",
             lambda s: s.det("head_drop") and not s.det("phone_use"), repeat_sec=5.0,
             hides=("look_down", "eyes_closed")),
        # --- corp (detecții CV2 din pipeline: body/chest, breathing, phone, slump) ---
        Rule("chest_and_breathing", 3, "URGENȚĂ: mâna pe piept și respirație grea! Sunteți bine?",
             lambda s: s.det("hand_on_chest") and s.det("heavy_breathing"),
             min_sec=2.0, repeat_sec=3.0, hides=("hand_on_chest", "heavy_breathing")),
        Rule("head_on_wheel", 3, "URGENȚĂ: șoferul pare prăbușit (capul pe volan)! Răspundeți!",
             lambda s: s.det("head_on_wheel"),
             min_sec=2.0, repeat_sec=3.0, hides=("no_face", "look_down", "eyes_closed", "distracted")),
        Rule("hand_on_chest", 2, "Mâna pe piept. Vă simțiți bine?",
             lambda s: s.det("hand_on_chest"), min_sec=2.0, repeat_sec=8.0),
        Rule("heavy_breathing", 2, "Respirație grea. Vă simțiți bine?",
             lambda s: s.det("heavy_breathing"), min_sec=3.0, repeat_sec=10.0),
        Rule("phone_looking", 2, "Folosiți telefonul și nu priviți drumul! Lăsați telefonul jos.",
             lambda s: s.det("phone_use") and s.head.face_visible and s.head.direction != CENTER,
             min_sec=1.0, repeat_sec=8.0, hides=("phone_use", "look_down", "look_left", "look_right")),
        Rule("phone_use", 1, "Telefon în mână. Vă rugăm să-l lăsați jos.",
             lambda s: s.det("phone_use"), min_sec=1.0, repeat_sec=10.0),
    ]


class AlertManager:
    def __init__(self, rules: Optional[list[Rule]] = None) -> None:
        self.rules = rules or default_rules()
        self._since: dict[str, float] = {}       # de când e adevărată condiția
        self._last_emit: dict[str, float] = {}
        self._ended: dict[str, float] = {}       # când s-a oprit ultima oară
        self.active: list[Alert] = []

    def update(self, eyes: EyeState, head: HeadState,
               detections: dict[str, tuple[bool, float]], t: float) -> list[Alert]:
        sig = Signals(eyes, head, detections)
        new: list[Alert] = []
        self.active = []
        for r in self.rules:
            if not r.condition(sig):
                if r.code in self._since:
                    self._ended[r.code] = t
                self._since.pop(r.code, None)
                self._last_emit.pop(r.code, None)
                continue

            start = self._since.setdefault(r.code, t)
            dur = t - start
            if dur < r.min_sec:
                continue
            if t - self._ended.get(r.code, -1e9) < r.cooldown_sec and r.code not in self._last_emit:
                continue

            w = Alert(r.code, r.level, r.message, round(dur, 1))
            self.active.append(w)
            last = self._last_emit.get(r.code)
            if last is None or t - last >= r.repeat_sec:
                self._last_emit[r.code] = t
                new.append(w)

        # avertizările grave le ascund pe cele redundante (ex. critic -> "ochi închiși")
        by_code = {r.code: r for r in self.rules}
        hidden = {h for w in self.active for h in by_code[w.code].hides}
        self.active = [w for w in self.active if w.code not in hidden]
        new = [w for w in new if w.code not in hidden]
        # cel mai sever primul
        self.active.sort(key=lambda w: -w.level)
        new.sort(key=lambda w: -w.level)
        return new

    @property
    def level(self) -> int:
        """Cel mai mare nivel activ (0 = nimic)."""
        return max((w.level for w in self.active), default=0)
