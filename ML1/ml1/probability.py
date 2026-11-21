"""ML1 — matricea de probabilitate a stărilor șoferului (filtru Bayes / HMM) și statisticile de eficiență.

Stări ascunse S = severitatea REALĂ 0..3 (atent, distras, obosit, urgență).
Observația pe cadru O = (severitatea dată de reguli, treapta scorului a) -> 4 x 3 = 12 simboluri.

    E[s, o]    = P(O = o | S = s)                 matricea de emisie (cât de des „spun” regulile o când starea e s)
    T(dt)[i,j] = P(S(t+dt) = j | S(t) = i)        matricea de tranziție Markov, din generatorul Q (pe secundă)
    b_t        ∝ E[:, o_t] ⊙ (T(dt)ᵀ · b_{t-1})   filtrul înainte: 16 + 4 înmulțiri pe cadru, fără numpy

Cadrele fără măsurare de încredere (camera oprită) fac doar pasul de predicție: lipsa observației nu e
„șofer atent”. Decizia Bayes cu cost asimetric (Curs 4/5): sev* = argmin_p Σ_s b[s] · COST[s, p].

Matricele se învață pe TRAIN în `ml1.evaluate` și se salvează în `prob_matrix.json` lângă acest fișier.
Fără fișier se folosesc valori implicite prudente (DEFAULT_*).

    python -m ml1.probability            # matricele curente + statisticile pe confuzia salvată
    python -m ml1.probability --fit      # re-învață matricele pe șoferii de train (simulare, ~20 s)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

K = 4
STATE_NAMES = ("atent", "distras", "obosit", "urgență")
A_CUTS = (0.75, 0.5)        # treptele scorului a: >= 0.75 | 0.5..0.75 | < 0.5
N_OBS = K * 3
MIN_CONF = 0.3              # sub atât (camera oprită, doar voce) cadrul nu e o observație
MAX_DT = 120.0              # pauze mai lungi: predicția se plafonează (oricum aproape de staționar)
B_FLOOR = 1e-6              # nicio stare nu devine imposibilă (altfel filtrul nu mai poate reveni)
MATRIX_PATH = Path(__file__).with_name("prob_matrix.json")

# Cost asimetric (același ca în evaluare): subestimare cu k trepte = 5k (x2 dacă era urgență),
# supraestimare = k. COST[adevăr][predicție].
COST = [[0.0 if p == t else float(5 * (t - p) * (2 if t == 3 else 1) if p < t else (p - t))
         for p in range(K)] for t in range(K)]


def a_bin(a: float | None) -> int:
    if a is None:
        return 0
    return 0 if a >= A_CUTS[0] else (1 if a >= A_CUTS[1] else 2)


def obs_symbol(rule_sev: int, a: float | None) -> int:
    return max(0, min(3, int(rule_sev))) * 3 + a_bin(a)


def _normalize(v: list[float]) -> list[float]:
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


# Valori implicite (înainte de prima învățare): regulile sunt corecte ~85% pe diagonală, iar starea se
# schimbă rar (durata medie: distras ~10 s, obosit ~40 s, urgență ~5 min).
DEFAULT_EMISSION = [
    # obs = sev_reguli * 3 + treapta a      (sev 0 | sev 1 | sev 2 | sev 3), fiecare cu a mare / mediu / mic
    [0.80, 0.06, 0.01, 0.05, 0.03, 0.01, 0.015, 0.01, 0.005, 0.005, 0.003, 0.002],
    [0.20, 0.10, 0.02, 0.30, 0.25, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005],
    [0.06, 0.08, 0.04, 0.03, 0.04, 0.03, 0.15, 0.25, 0.25, 0.02, 0.02, 0.03],
    [0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.03, 0.04, 0.08, 0.10, 0.15, 0.50],
]
DEFAULT_RATES = [  # Q[i][j], 1/s; diagonala se completează astfel încât rândul să dea 0
    [0.0, 1 / 120, 1 / 900, 1 / 3600],
    [1 / 10, 0.0, 1 / 900, 1 / 3600],
    [1 / 40, 1 / 600, 0.0, 1 / 1800],
    [1 / 600, 1 / 3600, 1 / 1200, 0.0],
]
DEFAULT_PRIOR = [0.97, 0.01, 0.01, 0.01]


def _generator(rates: list[list[float]]) -> list[list[float]]:
    q = [[max(0.0, float(x)) for x in row] for row in rates]
    for i in range(K):
        q[i][i] = 0.0
        q[i][i] = -sum(q[i])
    return q


@dataclass
class ProbModel:
    emission: list[list[float]]
    q: list[list[float]]
    prior: list[float]
    meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    # --------------------------------------------------------------- I/O
    @classmethod
    def default(cls) -> "ProbModel":
        return cls([_normalize(r) for r in DEFAULT_EMISSION], _generator(DEFAULT_RATES), list(DEFAULT_PRIOR),
                   {"source": "implicit (neînvățat)"})

    @classmethod
    def load(cls, path: Path | None = None) -> "ProbModel":
        path = path or MATRIX_PATH
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            m = cls([_normalize(r) for r in d["emission"]], _generator(d["q"]), _normalize(d["prior"]), d.get("meta", {}))
            if len(m.emission) != K or any(len(r) != N_OBS for r in m.emission):
                raise ValueError("dimensiuni greșite")
            return m
        except (OSError, ValueError, KeyError, TypeError):
            return cls.default()

    def save(self, path: Path | None = None) -> Path:
        path = path or MATRIX_PATH
        d = {"states": list(STATE_NAMES), "a_cuts": list(A_CUTS),
             "emission": [[round(x, 6) for x in r] for r in self.emission],
             "q": [[round(x, 7) for x in r] for r in self.q],
             "prior": [round(x, 6) for x in self.prior], "meta": self.meta}
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    # --------------------------------------------------------- tranziție
    def transition(self, dt: float) -> list[list[float]]:
        """T(dt) = exp(Q·dt), aproximat prin (I + Q·dt/n)^n cu pași de cel mult 0.1 s. Rezultatul e pus în
        cache pe dt rotunjit la 10 ms: la 5–10 Hz se calculează practic o singură dată."""
        dt = min(MAX_DT, max(0.0, dt))
        key = round(dt, 2)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = max(1, math.ceil(key / 0.1))
        h = key / n
        step = [[(1.0 if i == j else 0.0) + self.q[i][j] * h for j in range(K)] for i in range(K)]
        out = [[1.0 if i == j else 0.0 for j in range(K)] for i in range(K)]
        base, e = step, n
        while e:                       # ridicare la putere prin pătrate repetate: O(log n) produse 4x4
            if e & 1:
                out = _matmul(out, base)
            base, e = _matmul(base, base), e >> 1
        out = [_normalize([max(0.0, x) for x in row]) for row in out]
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = out
        return out

    def stationary(self) -> list[float]:
        b = [1.0 / K] * K
        T = self.transition(60.0)
        for _ in range(200):
            b = _normalize([sum(b[i] * T[i][j] for i in range(K)) for j in range(K)])
        return b

    def mean_dwell_sec(self) -> list[float]:
        return [(-1.0 / self.q[i][i]) if self.q[i][i] < 0 else math.inf for i in range(K)]


# ------------------------------------------------------------------ filtru

class StateFilter:
    """Filtrul înainte, pe cadru. Pur Python: ~20 µs/cadru, memorie constantă (doar vectorul b)."""

    def __init__(self, model: ProbModel | None = None):
        self.model = model or ProbModel.load()
        self.reset()

    def reset(self) -> None:
        self.b = list(self.model.prior)
        self._ts: float | None = None
        self.observed = False

    def update(self, rule_sev: int, a: float | None, confidence: float, ts: float) -> list[float]:
        dt = 0.0 if self._ts is None else max(0.0, ts - self._ts)
        self._ts = ts if self._ts is None else max(self._ts, ts)
        b = self.b
        if dt > 0:
            T = self.model.transition(dt)
            b = [b[0] * T[0][j] + b[1] * T[1][j] + b[2] * T[2][j] + b[3] * T[3][j] for j in range(K)]
        self.observed = confidence >= MIN_CONF
        if self.observed:
            o = obs_symbol(rule_sev, a)
            E = self.model.emission
            b = [b[s] * E[s][o] for s in range(K)]
        b = _normalize(b)
        if min(b) < B_FLOOR:
            b = _normalize([max(B_FLOOR, x) for x in b])
        self.b = b
        return b

    # ---------------------------------------------------- mărimi derivate
    @property
    def p_risk(self) -> float:
        """P(obosit sau urgență) — intrarea „risc” a controlerului fuzzy."""
        return self.b[2] + self.b[3]

    @property
    def expected_severity(self) -> float:
        return sum(s * p for s, p in enumerate(self.b))

    @property
    def map_state(self) -> int:
        return max(range(K), key=lambda s: self.b[s])

    def bayes_severity(self, cost: list[list[float]] = COST) -> int:
        return bayes_decision(self.b, cost)

    def to_dict(self) -> dict:
        return {"p_state": [round(x, 4) for x in self.b], "p_risk": round(self.p_risk, 4),
                "expected_severity": round(self.expected_severity, 3), "map_state": self.map_state,
                "bayes_severity": self.bayes_severity(), "observed": self.observed}


def bayes_decision(b: list[float], cost: list[list[float]] = COST) -> int:
    """Severitatea cu riscul așteptat minim (nu cea mai probabilă): la cost asimetric, o urgență cu
    probabilitate 15% poate justifica deja o alarmă."""
    return min(range(K), key=lambda p: sum(b[t] * cost[t][p] for t in range(K)))


# ------------------------------------------------------------ învățare

def fit_model(seqs: list[dict], hz: float, alpha_emission: float = 1.0, pseudo_transition: float = 0.5) -> ProbModel:
    """Învață E și Q din secvențe etichetate (doar TRAIN).

    Fiecare secvență: {"truth": [-1..3], "rule": [0..3], "a": [...], "conf": [...]}; truth = -1 (oboseală
    intermediară) e ignorat la emisie și completat cu starea anterioară la tranziții.
    Netezire Laplace (alpha) la emisie și pseudo-numărători mici la tranziții: nicio celulă nu e 0."""
    em = [[alpha_emission] * N_OBS for _ in range(K)]
    tr = [[pseudo_transition if i != j else 0.0 for j in range(K)] for i in range(K)]
    first = [1.0] * K
    for sq in seqs:
        prev = None
        for t, r, a, c in zip(sq["truth"], sq["rule"], sq["a"], sq["conf"]):
            if t >= 0 and c >= MIN_CONF:
                em[t][obs_symbol(r, a)] += 1
            cur = t if t >= 0 else prev
            if cur is None:
                continue
            if prev is None:
                first[cur] += 1
            else:
                tr[prev][cur] += 1
            prev = cur
    emission = [_normalize(r) for r in em]
    p_frame = [_normalize(r) for r in tr]
    # Q ≈ hz · (P_cadru − I): exact la ordinul 1, iar probabilitățile pe cadru sunt mici (≤ 1%)
    rates = [[hz * p_frame[i][j] if i != j else 0.0 for j in range(K)] for i in range(K)]
    n_frames = int(sum(sum(r) for r in em) - alpha_emission * K * N_OBS)
    return ProbModel(emission, _generator(rates), _normalize(first),
                     {"source": "învățat pe train", "frames": n_frames, "trips": len(seqs), "hz": hz})


# ------------------------------------------------ statistici de eficiență

def matrix_stats(cm, prior: list[float] | None = None) -> dict:
    """Statistici din matricea de confuzie C[adevăr][predicție] (orice listă / numpy 4x4).

    * P(predicție | adevăr): rândurile normalizate (sensibilitatea și confuziile pe fiecare stare);
    * P(adevăr | predicție): coloanele, prin Bayes cu prevalența dată (implicit cea din date) — cât
      de credibilă e o alarmă de nivel p;
    * pe clasă: sensibilitate, specificitate, VPP, VPN, LR+ = sens/(1−spec), LR− = (1−sens)/spec;
    * global: acuratețe, acuratețe echilibrată, kappa Cohen, MCC multiclasă, cost mediu pe cadru."""
    C = [[float(cm[i][j]) for j in range(K)] for i in range(K)]
    n = sum(map(sum, C))
    if n <= 0:
        return {"n": 0}
    rows = [sum(r) for r in C]
    cols = [sum(C[i][j] for i in range(K)) for j in range(K)]
    p_pred_true = [[C[i][j] / rows[i] if rows[i] else 0.0 for j in range(K)] for i in range(K)]
    prev = _normalize(prior) if prior else [r / n for r in rows]
    p_true_pred = []
    for j in range(K):
        col = [prev[i] * p_pred_true[i][j] for i in range(K)]
        s = sum(col)
        p_true_pred.append([x / s if s else 0.0 for x in col])          # [predicție][adevăr]
    per = []
    for k in range(K):
        tp, fn, fp = C[k][k], rows[k] - C[k][k], cols[k] - C[k][k]
        tn = n - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else None
        spec = tn / (tn + fp) if tn + fp else None
        ppv = tp / (tp + fp) if tp + fp else None
        npv = tn / (tn + fn) if tn + fn else None
        # LR+ nedefinit (None) când specificitatea e 1: nicio alarmă falsă pentru clasa asta
        lr_pos = (sens / (1 - spec)) if sens is not None and spec is not None and spec < 1 else None
        lr_neg = ((1 - sens) / spec) if sens is not None and spec else None
        per.append({"state": STATE_NAMES[k], "support": int(rows[k]), "sensitivity": sens, "specificity": spec,
                    "ppv": ppv, "npv": npv, "lr_pos": lr_pos, "lr_neg": lr_neg})
    acc = sum(C[k][k] for k in range(K)) / n
    present = [p["sensitivity"] for p in per if p["sensitivity"] is not None]
    pe = sum(rows[k] * cols[k] for k in range(K)) / n ** 2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    c = sum(C[k][k] for k in range(K))
    mcc_den = math.sqrt((n ** 2 - sum(x * x for x in cols)) * (n ** 2 - sum(x * x for x in rows)))
    mcc = (c * n - sum(rows[k] * cols[k] for k in range(K))) / mcc_den if mcc_den else 0.0
    cost = sum(C[i][j] * COST[i][j] for i in range(K) for j in range(K)) / n
    return {"n": int(n), "accuracy": acc, "balanced_accuracy": sum(present) / len(present) if present else 0.0,
            "kappa": kappa, "mcc": mcc, "cost_per_frame": cost, "per_class": per,
            "p_pred_given_true": p_pred_true, "p_true_given_pred": p_true_pred, "prevalence": prev}


def format_matrix(M: list[list[float]], rows: list[str], cols: list[str], nd: int = 3) -> list[str]:
    w = max(8, nd + 4)
    out = [" " * 12 + "".join(f"{c:>{w}}" for c in cols)]
    for name, r in zip(rows, M):
        out.append(f"{name:<12}" + "".join(f"{x:>{w}.{nd}f}" for x in r))
    return out


# ----------------------------------------------------------------- CLI

def _fit_from_simulation(quick: bool) -> ProbModel:
    import numpy as np
    from . import evaluate as ev
    rng = np.random.default_rng(0)
    drivers = ev.make_drivers(4 if quick else 20, rng, "tr")
    sims = [ev.TripSim(d, ev.make_trip(rng), rng) for d in drivers]
    runs = [ev.run_system(s) for s in sims]
    return fit_model(ev.hmm_sequences(sims, runs), ev.HZ)


def main(argv=None) -> None:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fit", action="store_true", help="re-învață matricele pe șoferii de train și le salvează")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    m = _fit_from_simulation(args.quick) if args.fit else ProbModel.load()
    if args.fit:
        print(f"salvat: {m.save()}")
    print(f"Sursa: {m.meta}")
    obs = [f"r{s}/{x}" for s in range(K) for x in ("H", "M", "L")]
    print("\nEmisie P(observație | stare)   (r = severitate reguli, H/M/L = a mare/mediu/mic):")
    print("\n".join(format_matrix(m.emission, list(STATE_NAMES), obs, 2)))
    print("\nTranziție pe 1 s P(S(t+1) = j | S(t) = i):")
    print("\n".join(format_matrix(m.transition(1.0), list(STATE_NAMES), list(STATE_NAMES), 4)))
    print("\nDurata medie în stare (s): " + ", ".join(f"{n} {d:.0f}" for n, d in zip(STATE_NAMES, m.mean_dwell_sec())))
    res = ROOT_DOCS / "evaluation_results.json"
    if res.exists():
        st = json.loads(res.read_text(encoding="utf-8")).get("prob_stats", {}).get("test", {}).get("rules+calibrare")
        if st:
            print("\nP(starea reală | severitatea publicată), test, rules+calibrare:")
            print("\n".join(format_matrix(st["p_true_given_pred"], [f"publicat {s}" for s in range(K)], list(STATE_NAMES))))
            print(f"kappa {st['kappa']:.3f} · MCC {st['mcc']:.3f} · acuratețe echilibrată {st['balanced_accuracy']:.3f}")


ROOT_DOCS = Path(__file__).resolve().parent.parent / "docs"

if __name__ == "__main__":
    main()
