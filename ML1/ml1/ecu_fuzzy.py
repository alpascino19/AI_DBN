"""ML1 -> ECU: controler fuzzy (Mamdani) pentru viteză, asistența de bandă și distanța de siguranță.

Intrări (fuzzificate):
    a      scorul de atenție (fusion.py)                                         mic / mediu / mare
    risk   0.5·P(obosit) + P(urgență), din matricea de probabilitate (probability.py)   mic / mediu / mare
    conf   încrederea măsurării (camera, vocea)                                  mică / mare
Ieșiri (centroid):
    speed_factor  fracția din viteza setată (0..1)
    authority     cât preia asistența de bandă / MPC-ul: 0 = șoferul, 1 = sistemul
    headway_s     distanța de siguranță față de mașina din față, în secunde

Peste fuzzy stă un SUPERVIZOR determinist (aceeași idee ca în rules.py: regulile sunt podeaua):
    * severitatea 3 publicată -> viteză <= 50% + avarii + asistența de bandă preia (imediat, fără fuzzy);
    * manevra de risc minim (MRM: bandă de urgență, viteză -> 0) după apelul 112 sau după MRM_HOLD_SEC de
      severitate 3 CONFIRMATĂ de matricea de probabilitate (P(urgență) >= 0.5); rămâne activă până la
      revenirea la severitatea 0;
    * referința de viteză se schimbă cu decelerație / accelerație limitată (fără frânări bruște).

De ce fuzzy și nu un prag: tranziții line între „atent” și „obosit”, reguli citibile de un inginer auto,
iar inferența se poate compila într-un tabel (LUT) care rulează pe orice ECU, fără LLM și fără numpy:

    python -m ml1.ecu_fuzzy                 # tabelul de comportament pe situații tipice
    python -m ml1.ecu_fuzzy --bench         # timpul pe pas (Mamdani exact vs. LUT)
    python -m ml1.ecu_fuzzy --export-c      # ML1/ecu/vigil_fuzzy_lut.h (C99, LUT uint8 + interpolare)
"""

from __future__ import annotations

import math
from pathlib import Path

ECU_DIR = Path(__file__).resolve().parent.parent / "ecu"

# ------------------------------------------------------------ funcții de apartenență

def trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapez (a, b, c, d); a == b sau c == d dau umeri verticali."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def tri(x: float, a: float, b: float, c: float) -> float:
    return trap(x, a, b, b, c)


INPUTS = {
    "a": {"mic": (0.0, 0.0, 0.40, 0.60), "mediu": (0.45, 0.65, 0.65, 0.85), "mare": (0.75, 0.90, 1.0, 1.0)},
    "risk": {"mic": (0.0, 0.0, 0.15, 0.35), "mediu": (0.20, 0.45, 0.45, 0.70), "mare": (0.55, 0.80, 1.0, 1.0)},
    "conf": {"mică": (0.0, 0.0, 0.20, 0.45), "mare": (0.30, 0.55, 1.0, 1.0)},
}

# Universurile ieșirilor sunt extinse peste [0, 1] cu triunghiuri simetrice la capete: centroidul poate
# ajunge exact la 1.0 (viteză normală) și 0.0 (fără intervenție), apoi se decupează.
OUTPUTS = {
    "speed": {"range": (-0.1, 1.1), "terms": {"târâș": (0.15, 0.30, 0.45), "lent": (0.45, 0.60, 0.75),
                                              "redus": (0.70, 0.82, 0.94), "normal": (0.90, 1.00, 1.10)}},
    "authority": {"range": (-0.2, 1.2), "terms": {"șofer": (-0.2, 0.0, 0.2), "împărțit": (0.25, 0.5, 0.75),
                                                  "sistem": (0.8, 1.0, 1.2)}},
    "headway": {"range": (1.0, 4.0), "terms": {"normal": (1.4, 1.8, 2.2), "mare": (2.0, 2.5, 3.0),
                                               "maxim": (2.8, 3.3, 3.8)}},
}
CLIP = {"speed": (0.0, 1.0), "authority": (0.0, 1.0), "headway": (1.5, 3.5)}

# Baza de reguli (citită ca: DACĂ antecedent ATUNCI viteză, autoritate, distanță). Acoperă orice combinație
# de intrări: pentru fiecare (a, risk, conf) se aprinde cel puțin o regulă.
RULES: list[tuple[dict[str, str], tuple[str, str, str]]] = [
    ({"a": "mare", "risk": "mic", "conf": "mare"}, ("normal", "șofer", "normal")),
    ({"a": "mediu", "risk": "mic"}, ("redus", "împărțit", "mare")),
    ({"a": "mic", "risk": "mic"}, ("lent", "împărțit", "mare")),
    ({"a": "mare", "risk": "mediu"}, ("redus", "împărțit", "mare")),
    ({"a": "mediu", "risk": "mediu"}, ("lent", "împărțit", "maxim")),
    ({"a": "mic", "risk": "mediu"}, ("lent", "sistem", "maxim")),
    ({"risk": "mare"}, ("târâș", "sistem", "maxim")),
    ({"conf": "mică", "risk": "mic"}, ("redus", "împărțit", "mare")),   # camera oprită: prudență
]

N_GRID = 121


def _grid(lo: float, hi: float) -> list[float]:
    return [lo + (hi - lo) * i / (N_GRID - 1) for i in range(N_GRID)]


_GRIDS = {k: _grid(*v["range"]) for k, v in OUTPUTS.items()}
_MF = {k: {t: [tri(x, *p) for x in _GRIDS[k]] for t, p in v["terms"].items()} for k, v in OUTPUTS.items()}
_OUT_KEYS = ("speed", "authority", "headway")


def fuzzify(a: float, risk: float, conf: float) -> dict[str, dict[str, float]]:
    x = {"a": a, "risk": risk, "conf": conf}
    return {var: {t: trap(min(1.0, max(0.0, x[var])), *p) for t, p in terms.items()} for var, terms in INPUTS.items()}


def infer(a: float, risk: float, conf: float, explain: bool = False) -> dict:
    """Mamdani: ȘI = min, implicație = min (tăiere), agregare = max, defuzzificare = centroid."""
    mu = fuzzify(a, risk, conf)
    fired = []
    for ante, cons in RULES:
        w = min(mu[v][t] for v, t in ante.items())
        if w > 0:
            fired.append((w, cons))
    out: dict = {}
    for k_idx, key in enumerate(_OUT_KEYS):
        xs = _GRIDS[key]
        agg = [0.0] * N_GRID
        for w, cons in fired:
            mf = _MF[key][cons[k_idx]]
            for i in range(N_GRID):
                v = mf[i] if mf[i] < w else w
                if v > agg[i]:
                    agg[i] = v
        area = sum(agg)
        val = sum(x * m for x, m in zip(xs, agg)) / area if area > 0 else xs[0]
        lo, hi = CLIP[key]
        out[key] = min(hi, max(lo, val))
    if explain:
        out["rules"] = [{"if": ante, "then": cons, "w": round(w, 3)}
                        for (ante, cons), w in ((r, min(mu[v][t] for v, t in r[0].items())) for r in RULES) if w > 0]
    return out


# --------------------------------------------------------------------- LUT

def _axis(var: str, n_uniform: int) -> list[float]:
    """Nodurile tabelului: toate colțurile funcțiilor de apartenență + o grilă uniformă. Ieșirea are
    „genunchi” exact la colțuri (o regulă începe / încetează să se aprindă); cu nodurile acolo, eroarea
    interpolării rămâne doar din curbura centroidului, care e mică."""
    pts = {round(i / (n_uniform - 1), 4) for i in range(n_uniform)}
    pts |= {round(p, 4) for mf in INPUTS[var].values() for p in mf if 0.0 <= p <= 1.0}
    return sorted(pts)


LUT_AXES = {"a": _axis("a", 11), "risk": _axis("risk", 11), "conf": _axis("conf", 6)}


def build_lut(axes: dict = LUT_AXES) -> list:
    return [[[tuple(infer(a, r, c)[o] for o in _OUT_KEYS) for c in axes["conf"]] for r in axes["risk"]]
            for a in axes["a"]]


def quantize(lut) -> list[int]:
    """LUT -> uint8 [a][risk][conf][ieșire], ieșirea scalată pe intervalul ei (CLIP)."""
    flat = []
    for plane in lut:
        for row in plane:
            for cell in row:
                for key, v in zip(_OUT_KEYS, cell):
                    lo, hi = CLIP[key]
                    flat.append(int(round(255 * (v - lo) / (hi - lo))))
    return flat


def _locate(axis: list[float], x: float) -> tuple[int, float]:
    x = min(1.0, max(0.0, x))
    i = 0
    while i < len(axis) - 2 and x > axis[i + 1]:
        i += 1
    return i, (x - axis[i]) / (axis[i + 1] - axis[i])


def lut_eval(q: list[int], a: float, risk: float, conf: float, axes: dict = LUT_AXES) -> dict:
    """Interpolarea triliniară pe tabelul uint8 — EXACT ce face funcția C exportată (pentru verificare)."""
    nr, nc = len(axes["risk"]), len(axes["conf"])
    (i, fa), (j, fr), (k, fc) = _locate(axes["a"], a), _locate(axes["risk"], risk), _locate(axes["conf"], conf)
    out = {}
    for o, key in enumerate(_OUT_KEYS):
        acc = 0.0
        for di, wa in ((0, 1 - fa), (1, fa)):
            for dj, wr in ((0, 1 - fr), (1, fr)):
                for dk, wc in ((0, 1 - fc), (1, fc)):
                    acc += wa * wr * wc * q[(((i + di) * nr + (j + dj)) * nc + (k + dk)) * 3 + o]
        lo, hi = CLIP[key]
        out[key] = lo + (hi - lo) * acc / 255.0
    return out


def lut_error(q: list[int], n: int = 4000, seed: int = 1) -> dict:
    """Eroarea LUT față de inferența exactă pe n puncte aleatoare: {ieșire: (medie, maximă)}.
    Maximul apare doar în puncte izolate unde două reguli se predau una alteia abrupt (centroidul e
    neliniar acolo); 6 KB țin media sub 1%, iar pentru max < 0.04 ar trebui ~70 KB."""
    import random
    rnd = random.Random(seed)
    mx = {k: 0.0 for k in _OUT_KEYS}
    mean = {k: 0.0 for k in _OUT_KEYS}
    for _ in range(n):
        a, r, c = rnd.random(), rnd.random(), rnd.random()
        ex, lu = infer(a, r, c), lut_eval(q, a, r, c)
        for k in _OUT_KEYS:
            d = abs(ex[k] - lu[k])
            mx[k], mean[k] = max(mx[k], d), mean[k] + d / n
    return {k: (mean[k], mx[k]) for k in _OUT_KEYS}


def _c_floats(xs: list[float]) -> str:
    return ", ".join(f"{x:.4f}f" for x in xs)


def export_c(path: Path | None = None, axes: dict = LUT_AXES) -> tuple[Path, dict]:
    path = path or ECU_DIR / "vigil_fuzzy_lut.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    q = quantize(build_lut(axes))
    err = lut_error(q)
    na, nr, nc = len(axes["a"]), len(axes["risk"]), len(axes["conf"])
    body = ",\n".join("    " + ", ".join(str(v) for v in q[i:i + 15]) for i in range(0, len(q), 15))
    rules_txt = "\n".join(f" *   DACĂ {' ȘI '.join(f'{v} {t}' for v, t in ante.items())} -> "
                          f"viteză {c[0]}, autoritate {c[1]}, distanță {c[2]}" for ante, c in RULES)
    lim = {k: CLIP[k] for k in _OUT_KEYS}
    text = f"""/* vigil_fuzzy_lut.h — generat de `python -m ml1.ecu_fuzzy --export-c`. NU edita manual.
 *
 * Controlerul fuzzy Mamdani al Vigil (ML1/ml1/ecu_fuzzy.py) compilat într-un tabel: {na}x{nr}x{nc} puncte
 * (a, risc, încredere) x 3 ieșiri, uint8, {len(q)} octeți în flash. Interpolare triliniară, fără alocare,
 * fără recursivitate, timp mărginit. Eroarea față de inferența exactă (4000 de puncte aleatoare), medie / maximă:
 * viteză {err['speed'][0]:.4f} / {err['speed'][1]:.3f}, autoritate {err['authority'][0]:.4f} / {err['authority'][1]:.3f},
 * distanță {err['headway'][0]:.3f} / {err['headway'][1]:.3f} s.
 *
 * Intrări în [0, 1]: a = scorul de atenție, risk = 0.5 * P(obosit) + P(urgență) (matricea de probabilitate),
 * conf = încrederea măsurării.
 * Ieșiri: speed_factor [{lim['speed'][0]}, {lim['speed'][1]}], authority [{lim['authority'][0]}, {lim['authority'][1]}],
 *         headway_s [{lim['headway'][0]}, {lim['headway'][1]}].
 * Supervizorul determinist (plafoane pe severitate, MRM, limitarea ratei) NU e în acest fișier: el
 * aparține stratului de siguranță al ECU (vezi EcuController în ecu_fuzzy.py).
 *
 * Reguli:
{rules_txt}
 */
#ifndef VIGIL_FUZZY_LUT_H
#define VIGIL_FUZZY_LUT_H

#include <stdint.h>

#define VIGIL_LUT_NA {na}
#define VIGIL_LUT_NR {nr}
#define VIGIL_LUT_NC {nc}

static const uint8_t VIGIL_FUZZY_LUT[VIGIL_LUT_NA * VIGIL_LUT_NR * VIGIL_LUT_NC * 3] = {{
{body}
}};

static const float VIGIL_OUT_LO[3] = {{{lim['speed'][0]:.2f}f, {lim['authority'][0]:.2f}f, {lim['headway'][0]:.2f}f}};
static const float VIGIL_OUT_HI[3] = {{{lim['speed'][1]:.2f}f, {lim['authority'][1]:.2f}f, {lim['headway'][1]:.2f}f}};

/* Nodurile (neuniforme: includ colțurile funcțiilor de apartenență). */
static const float VIGIL_AX_A[VIGIL_LUT_NA] = {{{_c_floats(axes['a'])}}};
static const float VIGIL_AX_R[VIGIL_LUT_NR] = {{{_c_floats(axes['risk'])}}};
static const float VIGIL_AX_C[VIGIL_LUT_NC] = {{{_c_floats(axes['conf'])}}};

static inline void vigil_lut_idx(const float *ax, int n, float x, int *i, float *f)
{{
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    *i = 0;
    while (*i < n - 2 && x > ax[*i + 1]) ++*i;
    *f = (x - ax[*i]) / (ax[*i + 1] - ax[*i]);
}}

/* out[0] = speed_factor, out[1] = authority, out[2] = headway_s */
static inline void vigil_fuzzy_eval(float a, float risk, float conf, float out[3])
{{
    int i, j, k, o, di, dj, dk;
    float fa, fr, fc;
    vigil_lut_idx(VIGIL_AX_A, VIGIL_LUT_NA, a, &i, &fa);
    vigil_lut_idx(VIGIL_AX_R, VIGIL_LUT_NR, risk, &j, &fr);
    vigil_lut_idx(VIGIL_AX_C, VIGIL_LUT_NC, conf, &k, &fc);
    for (o = 0; o < 3; ++o) {{
        float acc = 0.0f;
        for (di = 0; di < 2; ++di)
            for (dj = 0; dj < 2; ++dj)
                for (dk = 0; dk < 2; ++dk) {{
                    float w = (di ? fa : 1.0f - fa) * (dj ? fr : 1.0f - fr) * (dk ? fc : 1.0f - fc);
                    acc += w * (float)VIGIL_FUZZY_LUT[((((i + di) * VIGIL_LUT_NR) + (j + dj)) * VIGIL_LUT_NC + (k + dk)) * 3 + o];
                }}
        out[o] = VIGIL_OUT_LO[o] + (VIGIL_OUT_HI[o] - VIGIL_OUT_LO[o]) * acc / 255.0f;
    }}
}}

#endif /* VIGIL_FUZZY_LUT_H */
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path, err


# --------------------------------------------------------------- supervizor

# Plafoane pe severitatea PUBLICATĂ. La severitatea 2 viteza o decide fuzzy-ul (riscul din matricea de
# probabilitate include deja regulile): o alarmă falsă costă puțin ca avertizare, dar mult ca frânare
# (evaluarea: plafonul fix la 85% încetinea mașina ~10% din condusul normal). Severitatea 3 rămâne podea.
SPEED_CAP = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5}
AUTHORITY_FLOOR = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}   # asistența de bandă: ieftină, nu frânează
HEADWAY_FLOOR = {0: 0.0, 1: 0.0, 2: 2.5, 3: 3.3}
MRM_HOLD_SEC = 10.0      # urgență confirmată atât timp (fără 112) -> manevra de risc minim
MRM_P_EMERG = 0.5        # confirmarea: P(urgență) din matricea de probabilitate, nu doar severitatea publicată
                         # (care are histerezis: rămâne 3 câteva secunde după un micro-somn de 3 s)
DECEL_NORMAL = 0.5       # m/s², încetinire blândă (cât frâna de motor): o privire scurtă abia se simte
DECEL_URGENT = 2.0       # m/s², la severitate 3 sau risc mare
DECEL_MRM = 2.5         # m/s², oprire controlată pe banda de urgență (sub ~3 m/s², fără risc de impact din spate)
ACCEL = 1.0              # m/s², revenirea la viteza setată (ca un ACC obișnuit)
FUZZY_TAU_DOWN = 1.5     # s: o privire de o secundă nu frânează mașina; plafoanele pe severitate rămân instantanee


def risk_from_state(p_state: list[float]) -> float:
    """Riscul ponderat cu gravitatea: un șofer sigur obosit = 0.5 (viteză ~60%, nu oprire pe autostradă),
    o urgență sigură = 1 (spre MRM). Distragerea (stare 1) intră prin scorul a, nu prin risc."""
    return min(1.0, max(0.0, 0.5 * p_state[2] + p_state[3]))


class EcuController:
    """Un pas = intrările ML1 la momentul ts -> comanda pentru ECU (dict publicat pe `vigil/ecu`)."""

    def __init__(self, v_set_kmh: float = 100.0, mrm_hold_sec: float = MRM_HOLD_SEC):
        self.v_set = float(v_set_kmh)
        self.mrm_hold = mrm_hold_sec
        self.speed_ref = self.v_set
        self.mrm = False
        self._fz_speed = 1.0
        self._ts: float | None = None
        self._sev3_since: float | None = None
        self.last: dict | None = None

    def step(self, ts: float, a: float, p_state: list[float], confidence: float, severity: int,
             called_112: bool = False, v_set_kmh: float | None = None) -> dict:
        if v_set_kmh is not None and v_set_kmh > 0:
            self.v_set = float(v_set_kmh)
        first = self._ts is None
        dt = 0.0 if first else max(0.0, ts - self._ts)
        self._ts = ts if first else max(self._ts, ts)
        sev = max(0, min(3, int(severity)))
        risk = risk_from_state(p_state)
        fz = infer(a, risk, confidence)
        # netezire asimetrică a vitezei fuzzy: coboară cu constanta FUZZY_TAU_DOWN, urcă imediat
        if fz["speed"] < self._fz_speed and not first:
            self._fz_speed += (1.0 - math.exp(-dt / FUZZY_TAU_DOWN)) * (fz["speed"] - self._fz_speed)
        else:
            self._fz_speed = fz["speed"]

        # supervizor --------------------------------------------------------
        confirmed = sev == 3 and p_state[3] >= MRM_P_EMERG
        self._sev3_since = (self._sev3_since if self._sev3_since is not None else ts) if confirmed else None
        if called_112 or (self._sev3_since is not None and ts - self._sev3_since >= self.mrm_hold):
            self.mrm = True
        elif sev == 0:
            self.mrm = False
        factor = min(self._fz_speed, SPEED_CAP[sev])
        authority = max(fz["authority"], AUTHORITY_FLOOR[sev])
        headway = max(fz["headway"], HEADWAY_FLOOR[sev])
        if self.mrm:
            factor, authority, headway = 0.0, 1.0, CLIP["headway"][1]
        target = factor * self.v_set

        # referința de viteză cu rată limitată (km/h; 1 m/s² = 3.6 km/h pe secundă)
        decel = 0.0
        if target < self.speed_ref:
            urgent = sev >= 3 or risk >= INPUTS["risk"]["mare"][0]
            lim = (DECEL_MRM if self.mrm else DECEL_URGENT if urgent else DECEL_NORMAL) * 3.6 * dt
            step = min(self.speed_ref - target, lim)
            self.speed_ref -= step
            decel = step / 3.6 / dt if dt > 0 else 0.0
        elif target > self.speed_ref:
            self.speed_ref += min(target - self.speed_ref, ACCEL * 3.6 * dt)

        mode = ("MRM" if self.mrm else "ASSIST" if authority >= 0.6 else
                "CAUTION" if factor < 0.95 or headway > 2.2 else "NORMAL")
        self.last = {
            "ts": round(ts, 3), "mode": mode,
            "speed_factor": round(factor, 3), "target_kmh": round(target, 1), "speed_ref_kmh": round(self.speed_ref, 1),
            "decel_mps2": round(decel, 2), "lka_authority": round(authority, 3), "headway_s": round(headway, 2),
            "hazards": bool(self.mrm or sev >= 3), "lane": "shoulder" if self.mrm else "keep",
            "inputs": {"a": round(a, 3), "risk": round(risk, 3), "confidence": round(confidence, 3), "severity": sev},
            "fuzzy": {k: round(v, 3) for k, v in fz.items()},
        }
        return self.last


# --------------------------------------------------------------------- CLI

SITUATIONS = [   # nume, a, P(stare), încredere, severitate publicată
    ("atent, cameră OK", 0.95, [0.97, 0.01, 0.01, 0.01], 0.6, 0),
    ("privire scurtă", 0.78, [0.90, 0.08, 0.01, 0.01], 0.6, 0),
    ("telefon în mână", 0.55, [0.10, 0.85, 0.04, 0.01], 0.6, 1),
    ("debut oboseală", 0.65, [0.45, 0.05, 0.48, 0.02], 0.6, 1),
    ("somnolent", 0.40, [0.05, 0.02, 0.90, 0.03], 0.6, 2),
    ("urgență medicală", 0.15, [0.01, 0.01, 0.03, 0.95], 0.6, 3),
    ("cameră oprită", 0.90, [0.95, 0.02, 0.02, 0.01], 0.2, 0),
]


def _table() -> None:
    print(f"{'situație':<20}{'a':>6}{'risc':>6}{'încr.':>6}{'sev':>4} | {'viteză':>7}{'autorit.':>9}{'distanță':>9}  mod")
    for name, a, p, c, s in SITUATIONS:
        out = EcuController().step(0.0, a, p, c, s)
        print(f"{name:<20}{a:>6.2f}{risk_from_state(p):>6.2f}{c:>6.2f}{s:>4} | {out['speed_factor']:>6.0%} "
              f"{out['lka_authority']:>8.2f}{out['headway_s']:>8.1f} s  {out['mode']}{'  + avarii' if out['hazards'] else ''}")


def _bench() -> None:
    import random
    import time
    rnd = random.Random(0)
    pts = [(rnd.random(), rnd.random(), rnd.random()) for _ in range(2000)]
    t0 = time.perf_counter()
    for p in pts:
        infer(*p)
    t1 = time.perf_counter()
    q = quantize(build_lut())
    t2 = time.perf_counter()
    for p in pts:
        lut_eval(q, *p)
    t3 = time.perf_counter()
    print(f"Mamdani exact: {(t1 - t0) / len(pts) * 1e6:.0f} µs/pas · LUT (Python): {(t3 - t2) / len(pts) * 1e6:.0f} µs/pas "
          f"· LUT: {len(q)} octeți")
    print("eroare LUT (medie / max):", ", ".join(f"{k} {m:.4f} / {x:.3f}" for k, (m, x) in lut_error(q).items()))


def main(argv=None) -> None:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description="controlerul fuzzy ML1 -> ECU")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--export-c", nargs="?", const="", metavar="CALE")
    args = p.parse_args(argv)
    if args.export_c is not None:
        path, err = export_c(Path(args.export_c) if args.export_c else None)
        print(f"scris: {path}   eroare medie / max: {', '.join(f'{k} {m:.4f} / {x:.3f}' for k, (m, x) in err.items())}")
        return
    if args.bench:
        _bench()
        return
    _table()


if __name__ == "__main__":
    main()
