"""Evaluarea ML1 după protocolul din Cursul 5 (Evaluation & Generalization).

    python -m ml1.evaluate                 # ~1-2 min, scrie docs/EVALUATION_RESULTS.md + docs/figures/
    python -m ml1.evaluate --llm           # include și varianta cu LLM (Ollama) pe test
    python -m ml1.evaluate --quick         # puțini șoferi (pentru teste)

Ce face și de ce (trimiteri la cursuri):
  * Date etichetate: călătorii simulate în FORMA REALĂ a mesajelor CV (phone_use, head_on_wheel,
    breathing/posture, face_detected), cu eticheta adevărată 0-3 pe cadru. Buclă închisă: șoferul
    simulat răspunde la întrebările motorului după starea lui reală (normal/obosit/urgență).
  * Împărțire pe ȘOFER, nu pe cadre (C5: cadre vecine din același video = scurgere de informație).
    train (învățare modele ML) / val (alegere praguri) / test (o singură dată) / shift (alt domeniu:
    ochi mici, zgomot mare, fața pierdută des — C5: distribution shift, raportat separat).
  * Metrici pe clasă + macro-F1, nu doar acuratețe (C2/C5: clase dezechilibrate — severitatea 3 e rară);
    cost asimetric (a rata o urgență costă mult mai mult decât o alarmă falsă, C4/C5);
    latența de detecție, alarme false pe oră, apeluri 112 corecte / ratate / false.
  * Baseline-uri (C5): clasa majoritară, prag simplu pe scorul a (praguri alese pe val, C4).
  * Modele clasice (C4, C6): regresie softmax și arbore de decizie (numpy), antrenate pe train,
    standardizare învățată doar pe train (C2: fără scurgere), ponderi pe clase pentru dezechilibru.
  * Incertitudine: interval de încredere bootstrap 95% pe călătorii (C5).
  * Curbe ROC / PR pentru scorul 1-a și diagramă de calibrare + Brier pentru probabilitățile softmax (C5).

LIMITĂ IMPORTANTĂ: etichetele vin din același simulator care generează datele. Rezultatele arată
regresii, compromisuri și efectul modificărilor, NU performanța pe șoferi reali. Pentru asta e
nevoie de înregistrări reale etichetate (vezi docs/EVALUATION.md, secțiunea „Date reale”).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import profile
from .engine import DecisionEngine

HZ = 5
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
CV_MIN_DUR = {"hand_on_chest": 2.0, "heavy_breathing": 3.0, "phone_use": 1.0, "head_on_wheel": 2.0, "default": 1.0}
CV_GAP_SEC = 1.0   # Vision/config.yaml events.gap_sec
K = 4
SEV_NAMES = ["0 OK", "1 Redusă", "2 Medie", "3 Ridicată"]
# Cost asimetric: subestimare cu k trepte = 5k (×2 dacă era urgență), supraestimare = 1k.
COST = np.array([[0 if p == t else (5 * (t - p) * (2 if t == 3 else 1) if p < t else (p - t)) for p in range(K)]
                 for t in range(K)], dtype=float)


# ------------------------------------------------------------ simulare ----

@dataclass
class Driver:
    id: str
    ear0: float
    perclos0: float
    blink0: float
    noise: float
    face_drop: float
    conf_scale: float
    latency: float
    domain: str


def make_drivers(n: int, rng: np.random.Generator, prefix: str, shift: bool = False) -> list[Driver]:
    out = []
    for i in range(n):
        if shift:  # ochi mici / PERCLOS măsurat mare / zgomot / fața pierdută des
            d = Driver(f"{prefix}{i}", rng.uniform(0.20, 0.24), rng.uniform(0.07, 0.11), rng.uniform(8, 26),
                       rng.uniform(1.4, 2.0), rng.uniform(0.05, 0.10), rng.uniform(0.7, 0.9), rng.uniform(1.0, 2.5), "shift")
        else:
            d = Driver(f"{prefix}{i}", float(np.clip(rng.normal(0.29, 0.02), 0.25, 0.34)), rng.uniform(0.02, 0.05),
                       rng.uniform(10, 22), rng.uniform(0.6, 1.2), rng.uniform(0.0, 0.02), rng.uniform(0.85, 1.0),
                       rng.uniform(0.8, 2.0), "standard")
        out.append(d)
    return out


def make_trip(rng: np.random.Generator) -> list[tuple[str, float, dict]]:
    segs: list[tuple[str, float, dict]] = [("normal", 35.0, {})]
    middle = ["glance", "glance", "yawn", "touch", "phone", "fatigue", "privacy"]
    rng.shuffle(middle)
    for kind in middle:
        segs.append(("normal", float(rng.uniform(20, 35)), {}))
        if kind == "glance":
            segs.append(("glance", 3.0, {"g": float(rng.uniform(0.6, 1.6))}))
        elif kind == "yawn":
            segs.append(("yawn", 4.0, {}))
        elif kind == "touch":
            segs.append(("touch", 3.0, {}))
        elif kind == "phone":
            segs.append(("phone", float(rng.uniform(5, 12)), {}))
        elif kind == "fatigue":
            rise = float(rng.uniform(25, 40))
            segs.append(("fatigue", rise + 30.0, {"rise": rise, "hold": 15.0, "rec": 15.0}))
        elif kind == "privacy":
            segs.append(("privacy", 15.0, {}))
    segs.append(("normal", float(rng.uniform(25, 35)), {}))
    end = rng.choice(["chest", "collapse", "none"])
    if end == "chest":
        segs.append(("chest", 20.0, {"progress": bool(rng.random() < 0.5)}))
        if segs[-1][2]["progress"]:
            segs.append(("collapse", 15.0, {}))
    elif end == "collapse":
        segs.append(("collapse", 20.0, {}))
    else:
        segs.append(("normal", 20.0, {}))
    return segs


def _fatigue_level(tau: float, p: dict) -> float:
    if tau < p["rise"]:
        return 0.85 * tau / p["rise"]
    if tau < p["rise"] + p["hold"]:
        return 0.85
    return max(0.0, 0.85 * (1 - (tau - p["rise"] - p["hold"]) / p["rec"]))


class TripSim:
    """Generează cadrele (forma CV) + eticheta adevărată + răspunsurile șoferului."""

    def __init__(self, driver: Driver, segs, rng: np.random.Generator):
        self.d, self.segs, self.rng = driver, segs, rng
        self.active_since: dict[str, float] = {}
        self.frames: list[dict] = []
        self.truth: list[int] = []
        self.kind: list[str] = []
        self.onsets: list[tuple[float, int, str]] = []
        self.emergency_onset: float | None = None
        self.seg_idx: list[int] = []
        # Comportamentul șoferului (răspunde / tace) e fixat per episod și tip de întrebare, NU tras
        # din fluxul aleator comun: două variante comparate primesc exact aceleași răspunsuri
        # (comparație pereche, Curs 5: modele comparate în aceleași condiții).
        self.behavior_seed = int(rng.integers(1 << 30))
        self._build()
        self._responsiveness()

    def _events(self, t: float, active: dict[str, dict]) -> list[dict]:
        """Emulează FIDEL tracker-ul CV (Vision/vigil_vision/events/tracker.py):
        * pauzele mai scurte de CV_GAP_SEC NU întrerup evenimentul (durata = episodul, nu închiderea continuă);
        * evenimentul apare doar după durata minimă (CV_MIN_DUR)."""
        for name, attrs in active.items():
            st = self.active_since.get(name)
            if st is None:
                self.active_since[name] = [t, t, attrs]
            else:
                st[1], st[2] = t, attrs
        for name in list(self.active_since):
            if t - self.active_since[name][1] > CV_GAP_SEC:
                del self.active_since[name]
        out = []
        for name, (start, last, attrs) in self.active_since.items():
            dur = last - start
            if dur >= CV_MIN_DUR.get(name, CV_MIN_DUR["default"]):
                conf = float(np.clip(self.rng.normal(0.82, 0.05) * self.d.conf_scale, 0.3, 0.99))
                out.append({"event": name, "duration_sec": round(dur, 1), "confidence": round(conf, 2), **attrs})
        return out

    def _build(self) -> None:
        """Semantica reală CV1/CV2 (Vision/vigil_vision/face/eyes.py, stack.py, pipeline.py):
        * închiderea ochilor e simulată cadru cu cadru (reprize, nu valori independente);
        * PERCLOS = fracția cadrelor „închise” pe ultimele 60 s, None în primele 10 s;
        * artefacte: pleoapele PAR închise la privit în jos (telefon) și la căscat — CV le măsoară ca închise;
        * evenimente CV1: eyes_closed, microsleep (închis >= 0.5 s), gaze_off_road (cu direcție),
          low_blink_rate, yawn, head_drop; CV2: phone_use, hand_on_chest, heavy_breathing, head_on_wheel."""
        d, rng, t0 = self.d, self.rng, 0.0
        prev_label = 0
        window: deque = deque()          # (t, închis măsurat) pentru PERCLOS pe 60 s
        run_left = 0                     # cadre rămase din repriza de închidere curentă
        closed_since = None
        for si, (kind, dur, p) in enumerate(self.segs):
            n_frames = int(round(dur * HZ))   # ts = index / HZ exact (fără derivă între segmente)
            seg_rng = np.random.default_rng([self.behavior_seed, si, 7])
            phone_lowered = seg_rng.random() < 0.6          # privește în jos cu pleoapele coborâte
            glance_dir = "left" if seg_rng.random() < 0.5 else "right"
            for i in range(n_frames):
                tau, t = i / HZ, round(t0 + i / HZ, 2)
                D, phone, gaze_off, chest, collapse, yawn, eyes_burst, cam = 0.0, False, False, False, False, False, False, True
                gaze_dir, artifact_closed = None, False
                label = 0
                if kind == "glance":
                    gaze_off, gaze_dir = tau < p["g"], glance_dir
                elif kind == "yawn":
                    yawn = tau < 2.5
                    artifact_closed = 0.8 <= tau < 2.0             # ochi strânși în timpul căscatului
                elif kind == "touch":
                    chest = tau < 1.5          # sub pragul CV de 2 s -> fără eveniment
                elif kind == "phone":
                    phone, label = True, 1
                    gaze_off, gaze_dir = rng.random() < 0.7, "down"
                    artifact_closed = phone_lowered and 1.0 <= tau < 3.5   # pleoape coborâte privind telefonul
                    gaze_off = gaze_off or artifact_closed
                elif kind == "fatigue":
                    D = _fatigue_level(tau, p)
                    in_hold = p["rise"] <= tau < p["rise"] + p["hold"]
                    eyes_burst = in_hold and (tau - p["rise"]) % 6 < 2.4
                    yawn = in_hold and 3 <= (tau - p["rise"]) % 6 < 5
                    label = 2 if D >= 0.6 else (0 if D < 0.4 else -1)
                elif kind == "privacy":
                    cam = False
                elif kind == "chest":
                    chest, label = True, 3
                elif kind == "collapse":
                    collapse, label = True, 3
                if label == 3 and self.emergency_onset is None:
                    self.emergency_onset = t
                if label > prev_label and label >= 1:
                    self.onsets.append((t, label, kind))
                if label >= 0:
                    prev_label = label

                if not cam:
                    window.clear()
                    closed_since = None
                    msg = {"ts": t, "camera_on": False, "events": [], "phone": 0.0}
                else:
                    n = d.noise
                    face = not (collapse and tau >= 2 and rng.random() < 0.7) and rng.random() >= d.face_drop
                    # reprize de închidere: clipiri lungi tot mai dese și mai lungi odată cu oboseala
                    if run_left <= 0 and rng.random() < (0.15 + 0.30 * D) / HZ:
                        run_left = max(1, int(round((0.15 + 0.9 * D ** 1.3) * HZ)))
                    blink_closed = run_left > 0
                    run_left -= 1
                    closed_true = collapse or eyes_burst or blink_closed
                    measured_closed = closed_true or artifact_closed
                    if face:
                        window.append((t, measured_closed))
                    while window and t - window[0][0] > 60.0:
                        window.popleft()
                    closed_since = (closed_since if closed_since is not None else t) if measured_closed else None
                    elapsed = t - (window[0][0] if window else t)
                    perclos = None
                    if window and t >= 10.0 and elapsed >= 9.8:
                        perclos = sum(1 for _, c in window if c) / len(window)
                        perclos = float(np.clip(perclos + (d.perclos0 - 0.03) + rng.normal(0, 0.005 * n), 0, 1))
                    ear = d.ear0 * (0.25 if measured_closed else (1 - 0.15 * D)) + rng.normal(0, 0.012 * n)
                    blink = max(0.0, d.blink0 * (1 - 0.6 * D) + rng.normal(0, 1.5 * n))

                    active: dict[str, dict] = {}
                    if measured_closed:
                        active["eyes_closed"] = {}
                        if closed_since is not None and t - closed_since >= 0.5:
                            active["microsleep"] = {}
                    if gaze_off or collapse:
                        active["gaze_off_road"] = {"direction": gaze_dir or "down"}
                    if yawn:
                        active["yawn"] = {}
                    if t >= 10.0 and blink < 8.0:
                        active["low_blink_rate"] = {}
                    if phone:
                        active["phone_use"] = {"in_hand": True, "usage": "looking"}
                    if chest:
                        active["hand_on_chest"] = {"side": "left"}
                        if kind == "chest":
                            active["hand_on_chest"]["breathing"] = "heavy"
                            active["heavy_breathing"] = {"rate_bpm": 32.0}
                    if collapse:
                        active["head_drop"] = {}
                        if tau >= 3:
                            active["head_on_wheel"] = {"face_visible": False}
                    if not face:   # fără față CV1 nu produce evenimente de față
                        for k in ("eyes_closed", "microsleep", "gaze_off_road", "yawn", "low_blink_rate"):
                            active.pop(k, None)
                    msg = {
                        "ts": t, "camera_on": True, "face_detected": face, "body_detected": True,
                        "ear": round(float(np.clip(ear, 0.03, 0.45)), 3) if face else None,
                        "perclos": round(perclos, 3) if (face and perclos is not None) else None,
                        "blink_rate": round(float(blink), 1) if (face and t >= 10.0) else None,
                        "gaze_on_road": (not gaze_off and not collapse) if face else None,
                        # CV: head_drop de la față SAU de la corpul prăbușit (pipeline._head_drop)
                        "head_drop": True if collapse else (False if face else None),
                        "phone": round(float(np.clip(rng.normal(0.82, 0.06), 0, 1)) if phone else abs(rng.normal(0.02, 0.02)), 2),
                        "breathing": {"state": "heavy" if kind == "chest" else "normal"},
                        "posture": {"state": "slumped" if collapse and tau >= 3 else "upright",
                                    "still_sec": round(tau, 1) if collapse else 0.0},
                        "events": self._events(t, active),
                    }
                self.frames.append(msg)
                self.truth.append(label)
                self.kind.append(kind)
                self.seg_idx.append(si)
            t0 += n_frames / HZ
        self._grace()

    def _answers(self, si: int, ok_q: bool) -> bool:
        """Același tragere ca în respond(): șoferul răspunde la acest tip de întrebare în acest episod?"""
        kind = self.segs[si][0]
        r = np.random.default_rng([self.behavior_seed, si, int(ok_q)]).random()
        if kind == "collapse":
            return False
        if kind == "chest":
            return r < 0.5
        if kind == "fatigue":
            return r < (0.9 if ok_q else 0.7)
        return True

    def _responsiveness(self) -> None:
        """Tabelul de severitate (TASKS.md): ochi închiși / oboseală + FĂRĂ răspuns vocal = 3. Un episod de
        oboseală în care șoferul simulat tace e deci ambiguu între 2 și 3: cadrele lui de severitate 2 nu
        intră în matrice, iar un 112 după un „Sunteți bine?” fără răspuns e acceptabil (nici corect, nici fals)."""
        self.unresponsive = {si for si, (kind, _, _) in enumerate(self.segs)
                             if kind == "fatigue" and not (self._answers(si, False) and self._answers(si, True))}
        for i, si in enumerate(self.seg_idx):
            if si in self.unresponsive and self.labels[i] == 2:
                self.labels[i] = -1

    def _grace(self) -> None:
        """Ferestre „nu contează”: 4 s după o escaladare reală, 12 s după o de-escaladare
        (sistemul are histerezis intenționat). Etichetele -1 (oboseală intermediară) la fel."""
        lab = list(self.truth)
        last_up, last_down, prev = -1e9, -1e9, 0
        for i, v in enumerate(self.truth):
            t = i / HZ
            if v >= 0:
                if v > prev:
                    last_up = t
                elif v < prev:
                    last_down = t
                prev = v
            if v < 0 or t - last_up < 4.0 or t - last_down < 12.0:
                lab[i] = -1
        self.labels = lab

    def respond(self, question: str, t: float) -> tuple[float, dict] | None:
        """Răspunsul șoferului simulat la o întrebare, după starea lui REALĂ la momentul t."""
        i = min(len(self.kind) - 1, int(t * HZ))
        ok_q = "bine" in question
        kind, L = self.kind[i], self.d.latency
        rng = np.random.default_rng([self.behavior_seed, self.seg_idx[i], int(ok_q)])
        if kind == "collapse":
            return None
        if kind == "chest":
            return (float(rng.uniform(3, 5)), {"answered": True, "latency_sec": 4.0, "text": "nu mă simt bine"}) \
                if rng.random() < 0.5 else None
        if kind == "fatigue":
            if rng.random() < (0.9 if ok_q else 0.7):
                lat = L + float(rng.uniform(1.0, 2.5))
                return lat, {"answered": True, "latency_sec": round(lat, 1), "text": "da, sunt bine" if ok_q else "da, opresc"}
            return None
        lat = L + float(rng.uniform(0, 0.5))
        return lat, {"answered": True, "latency_sec": round(lat, 1), "text": "da, sunt bine"}


# ---------------------------------------------------------- rulare ----

def run_system(sim: TripSim, llm=None, calibrate: bool = True) -> dict:
    eng = DecisionEngine(llm_fn=llm, calibrate=calibrate)
    pending: list[tuple[float, dict]] = []
    pred, a_vals, feats, decisions = [], [], [], []
    cur = 0
    for msg in sim.frames:
        t = msg["ts"]
        for item in [p for p in pending if p[0] <= t]:
            eng.on_driver_response(item[1], ts=item[0])
            pending.remove(item)
        for d in eng.on_vision(msg):
            cur = d["severity"]
            decisions.append(d)
            for a in d["actions"]:
                if a["tool"] == "ask_driver":
                    r = sim.respond(a["args"]["question"], t)
                    if r is not None:
                        pending.append((t + r[0], r[1]))
        pred.append(cur)
        r = eng.last_result
        a_vals.append(r.a)
        feats.append(features(msg, r, eng))
    t112 = next((d["ts"] for d in decisions if any(a["tool"] == "call_112" for a in d["actions"])), None)
    return {"pred": np.array(pred), "a": np.array(a_vals), "X": np.array(feats, dtype=float),
            "decisions": decisions, "t112": t112, "llm_calls": eng.stats["llm_calls"]}


EV_FEATS = ["hand_on_chest", "heavy_breathing", "phone_in_hand", "head_on_wheel", "head_drop", "eyes_closed", "yawn"]
FEATURE_NAMES = ["a", "a_raw", "perclos", "ear_ratio", "blink", "gaze_off", "face_missing", "camera_off", "phone",
                 "breathing_heavy", "slumped"] + [f"dur_{e}" for e in EV_FEATS]


def features(msg: dict, r, eng: DecisionEngine) -> list[float]:
    """Vectorul de trăsături pe cadru (Curs 2: reprezentarea contează). Doar informație disponibilă
    la momentul predicției; valorile lipsă primesc valori neutre + indicator explicit (face_missing)."""
    from .cv_adapter import normalize_vision
    m = normalize_vision(msg)
    evs = {e["event"]: e for e in m.get("events", []) if (e.get("confidence") or 0) >= 0.5}
    base_ear = eng.fusion.baseline_ear or 0.30
    ear = m.get("ear")
    return [
        r.a, r.a_raw if r.a_raw is not None else r.a,
        m.get("perclos") if m.get("perclos") is not None else 0.1,
        (ear / base_ear) if ear is not None else 1.0,
        m.get("blink_rate") if m.get("blink_rate") is not None else 15.0,
        1.0 if m.get("gaze_on_road") is False else (0.5 if m.get("gaze_on_road") is None else 0.0),
        1.0 if m.get("face_detected") is False else 0.0,
        1.0 if m.get("camera_on") is False else 0.0,
        m.get("phone") or 0.0,
        1.0 if m.get("breathing_state") == "heavy" else 0.0,
        1.0 if m.get("posture_state") == "slumped" else 0.0,
    ] + [min(10.0, float(evs[e]["duration_sec"])) if e in evs else 0.0 for e in EV_FEATS]


# ------------------------------------------------------- modele ML ----

class Standardizer:
    """Fit doar pe train (Curs 2: fit_transform pe train, transform pe rest)."""

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=K).astype(float)
    w = np.where(counts > 0, len(y) / (K * np.maximum(counts, 1)), 0.0)
    return w[y]


def fit_softmax(X, y, epochs=500, lr=0.5, l2=1e-3):
    """Regresie softmax (Curs 4) cu gradient descent pe tot setul (Curs 3), ponderi pe clase."""
    n, d = X.shape
    W, b = np.zeros((d, K)), np.zeros(K)
    Y, sw = np.eye(K)[y], class_weights(y)
    for _ in range(epochs):
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None] / sw.sum()
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def softmax_proba(X, Wb):
    W, b = Wb
    Z = X @ W + b
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


class Tree:
    """Arbore de decizie (Curs 6): Gini ponderat, adâncime limitată, praguri din cuantile."""

    def __init__(self, depth=4, min_leaf=40):
        self.depth, self.min_leaf = depth, min_leaf

    def fit(self, X, y):
        self.root = self._grow(X, y, class_weights(y), 0)
        return self

    def _leaf(self, y, w):
        dist = np.bincount(y, weights=w, minlength=K)
        return {"leaf": int(dist.argmax()), "dist": (dist / max(dist.sum(), 1e-9)).round(3).tolist(), "n": int(len(y))}

    def _grow(self, X, y, w, depth):
        if depth >= self.depth or len(y) < 2 * self.min_leaf or len(set(y.tolist())) == 1:
            return self._leaf(y, w)

        def gini(wc):
            s = wc.sum()
            return 0.0 if s <= 0 else 1 - ((wc / s) ** 2).sum()

        best = None
        parent = gini(np.bincount(y, weights=w, minlength=K)) * w.sum()
        for j in range(X.shape[1]):
            for thr in np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 19))):
                left = X[:, j] <= thr
                nl = left.sum()
                if nl < self.min_leaf or len(y) - nl < self.min_leaf:
                    continue
                g = gini(np.bincount(y[left], weights=w[left], minlength=K)) * w[left].sum() + \
                    gini(np.bincount(y[~left], weights=w[~left], minlength=K)) * w[~left].sum()
                if best is None or g < best[0]:
                    best = (g, j, float(thr), left)
        if best is None or best[0] >= parent - 1e-9:
            return self._leaf(y, w)
        _, j, thr, left = best
        return {"f": j, "thr": thr, "l": self._grow(X[left], y[left], w[left], depth + 1),
                "r": self._grow(X[~left], y[~left], w[~left], depth + 1)}

    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            n = self.root
            while "leaf" not in n:
                n = n["l"] if x[n["f"]] <= n["thr"] else n["r"]
            out[i] = n["leaf"]
        return out

    def rules(self, names, node=None, ind="") -> list[str]:
        node = node or self.root
        if "leaf" in node:
            return [f"{ind}-> severitate {node['leaf']}  (n={node['n']}, distribuție {node['dist']})"]
        f, t = names[node["f"]], node["thr"]
        return ([f"{ind}dacă {f} <= {t:.3g}:"] + self.rules(names, node["l"], ind + "    ")
                + [f"{ind}altfel ({f} > {t:.3g}):"] + self.rules(names, node["r"], ind + "    "))


# ------------------------------------------------------- metrici ----

def confusion(y, p) -> np.ndarray:
    m = np.zeros((K, K), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    n = cm.sum()
    return {"accuracy": float(tp.sum() / n) if n else 0.0, "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "precision": prec.round(3).tolist(), "recall": rec.round(3).tolist(), "f1": f1.round(3).tolist(),
            "cost_per_frame": float((cm * COST).sum() / n) if n else 0.0, "n": int(n)}


def bootstrap(per_trip_cms: list[np.ndarray], fn, rng, n=400) -> tuple[float, float]:
    vals = []
    m = len(per_trip_cms)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(fn(sum(per_trip_cms[i] for i in idx)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def roc_pr(y_bin: np.ndarray, s: np.ndarray) -> dict:
    order = np.argsort(-s)
    y = y_bin[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    P, N = max(y.sum(), 1), max((1 - y).sum(), 1)
    tpr, fpr = np.concatenate([[0], tp / P]), np.concatenate([[0], fp / N])
    prec = np.concatenate([[1], tp / np.maximum(tp + fp, 1)])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return {"auc": float(trap(tpr, fpr)), "ap": float(np.sum(np.diff(np.concatenate([[0], tp / P])) * prec[1:])),
            "fpr": fpr, "tpr": tpr, "recall": tpr, "precision": prec}


def trip_metrics(sim: TripSim, run: dict) -> dict:
    lab, pred = np.array(sim.labels), run["pred"]
    mask = lab >= 0
    cm = confusion(lab[mask], pred[mask])
    # latența: de la debutul real până când sistemul ajunge la severitatea reală
    lat = []
    for t, level, kind in sim.onsets:
        i0 = int(t * HZ)
        hit = next((j for j in range(i0, min(len(pred), i0 + 30 * HZ)) if pred[j] >= level), None)
        lat.append({"kind": kind, "level": level, "latency": None if hit is None else (hit - i0) / HZ})
    # alarme false: decizii cu severitate >= 1 publicate când adevărul e 0
    lab_raw = np.array(sim.truth)
    fa = sum(1 for d in run["decisions"] if d["severity"] >= 1 and lab_raw[min(len(lab_raw) - 1, int(d["ts"] * HZ))] == 0)
    normal_hours = (lab_raw == 0).sum() / HZ / 3600
    emerg = sim.emergency_onset is not None
    calls = [d["ts"] for d in run["decisions"] if any(a["tool"] == "call_112" for a in d["actions"])]
    seg_at = lambda t: sim.seg_idx[min(len(sim.seg_idx) - 1, int(round(t * HZ)))]
    false_calls = [t for t in calls if not (emerg and t >= sim.emergency_onset)
                   and seg_at(t) not in sim.unresponsive]
    good_calls = [t for t in calls if emerg and t >= sim.emergency_onset]
    t112 = good_calls[0] if good_calls else (false_calls[0] if false_calls else None)
    return {
        "cm": cm, "latency": lat, "false_alarms": fa, "normal_hours": normal_hours,
        "112": ("fp" if false_calls else "tp" if good_calls else "fn" if emerg else "tn"),
        "t112_delay": (t112 - sim.emergency_onset) if (emerg and t112 is not None and t112 >= sim.emergency_onset) else None,
    }


def summarize(name: str, sims, runs, rng) -> dict:
    tm = [trip_metrics(s, r) for s, r in zip(sims, runs)]
    cm = sum(t["cm"] for t in tm)
    sc = scores(cm)
    lo, hi = bootstrap([t["cm"] for t in tm], lambda c: scores(c)["macro_f1"], rng)
    e_lo, e_hi = bootstrap([t["cm"] for t in tm], lambda c: scores(c)["recall"][3], rng)
    lats = {}
    for lv in (1, 2, 3):
        xs = [l["latency"] for t in tm for l in t["latency"] if l["level"] == lv]
        got = [x for x in xs if x is not None]
        lats[lv] = {"n": len(xs), "missed": len(xs) - len(got),
                    "median_s": float(np.median(got)) if got else None, "p90_s": float(np.percentile(got, 90)) if got else None}
    c112 = {k: sum(1 for t in tm if t["112"] == k) for k in ("tp", "fn", "fp", "tn")}
    delays = [t["t112_delay"] for t in tm if t["t112_delay"] is not None]
    hours = sum(t["normal_hours"] for t in tm)
    return {"name": name, "scores": sc, "cm": cm.tolist(), "macro_f1_ci": [lo, hi], "recall3_ci": [e_lo, e_hi],
            "latency": lats, "false_alarms_per_hour": sum(t["false_alarms"] for t in tm) / hours if hours else None,
            "calls_112": c112, "t112_median_s": float(np.median(delays)) if delays else None, "trips": len(sims)}


def frame_summary(name: str, sims, preds, rng) -> dict:
    cms = []
    for s, p in zip(sims, preds):
        lab = np.array(s.labels)
        m = lab >= 0
        cms.append(confusion(lab[m], p[m]))
    cm = sum(cms)
    lo, hi = bootstrap(cms, lambda c: scores(c)["macro_f1"], rng)
    e_lo, e_hi = bootstrap(cms, lambda c: scores(c)["recall"][3], rng)
    return {"name": name, "scores": scores(cm), "cm": cm.tolist(), "macro_f1_ci": [lo, hi], "recall3_ci": [e_lo, e_hi],
            "trips": len(sims)}


def a_threshold_predict(a: np.ndarray, cuts) -> np.ndarray:
    c1, c2, c3 = cuts
    return np.where(a < c3, 3, np.where(a < c2, 2, np.where(a < c1, 1, 0)))


def tune_a_thresholds(sims, runs) -> tuple:
    """Alegerea pragurilor pe VALIDARE prin cost minim (Curs 4/5), niciodată pe test."""
    y = np.concatenate([np.array(s.labels) for s in sims])
    a = np.concatenate([r["a"] for r in runs])
    m = y >= 0
    y, a = y[m], a[m]
    best = None
    for c1 in np.arange(0.95, 0.55, -0.05):
        for c2 in np.arange(c1 - 0.05, 0.25, -0.05):
            for c3 in np.arange(c2 - 0.05, 0.0, -0.05):
                cm = confusion(y, a_threshold_predict(a, (c1, c2, c3)))
                cost = (cm * COST).sum() / cm.sum()
                if best is None or cost < best[0]:
                    best = (cost, (round(float(c1), 2), round(float(c2), 2), round(float(c3), 2)))
    return best[1]


# ---------------------------------------------------------- grafice ----

STATUS = ["#0ca30c", "#fab219", "#ec835a", "#d03b3b"]


def plot_all(test_sims, res_runs: dict, roc: dict, rel: dict, example_idx: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    files = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8})

    # 1. matrici de confuzie
    names = [n for n in ("rules+calibrare", "rules fără calibrare", "prag pe a", "softmax", "arbore") if n in res_runs]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.8))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = np.array(res_runs[n]["cm"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b")
        ax.set_title(n, fontsize=10)
        ax.set_xticks(range(K), [str(k) for k in range(K)])
        ax.set_yticks(range(K), [str(k) for k in range(K)])
        ax.set_xlabel("prezis")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("adevărat")
    fig.suptitle("Matrici de confuzie pe test (rânduri normalizate = recall pe clasă)", fontsize=11)
    fig.tight_layout()
    f = FIG / "confusion_test.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)

    # 2. ROC + PR pentru scorul 1 - a (afectat = severitate >= 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(roc["fpr"], roc["tpr"], color="#2a78d6", lw=2)
    a1.plot([0, 1], [0, 1], color="#85847f", lw=1)
    a1.set_xlabel("FPR")
    a1.set_ylabel("TPR (recall)")
    a1.set_title(f"ROC: 1 - a  (AUC = {roc['auc']:.3f})")
    a2.plot(roc["recall"], roc["precision"], color="#2a78d6", lw=2)
    a2.set_xlabel("recall")
    a2.set_ylabel("precizie")
    a2.set_title(f"Precizie-recall  (AP = {roc['ap']:.3f})")
    a2.set_ylim(0, 1.02)
    fig.tight_layout()
    f = FIG / "roc_pr_attention.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)

    # 3. diagramă de calibrare softmax P(sev >= 2)
    if rel:
        fig, ax = plt.subplots(figsize=(4.4, 4))
        ax.plot([0, 1], [0, 1], color="#85847f", lw=1)
        ax.plot(rel["conf"], rel["freq"], marker="o", color="#2a78d6", lw=2, ms=5)
        ax.set_xlabel("probabilitate prezisă P(severitate ≥ 2)")
        ax.set_ylabel("frecvență observată")
        ax.set_title(f"Calibrare softmax (Brier = {rel['brier']:.3f})", fontsize=10)
        fig.tight_layout()
        f = FIG / "reliability_softmax.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(f.name)

    # 4. exemplu de călătorie: adevăr vs decizie + scor a
    s = test_sims[example_idx]
    run = res_runs["rules+calibrare"]["_runs"][example_idx]
    t = np.arange(len(s.truth)) / HZ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.plot(t, run["a"], color="#2a78d6", lw=1.5)
    ax1.set_ylabel("scor a")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Exemplu de călătorie din test ({s.d.id}, {s.d.domain})", fontsize=10)
    truth = np.clip(np.array(s.truth), 0, 3)
    ax2.step(t, truth, where="post", color="#85847f", lw=4, alpha=0.5, label="adevăr")
    ax2.step(t, run["pred"], where="post", color="#d03b3b", lw=1.6, label="decizia ML1")
    ax2.set_yticks(range(4))
    ax2.set_ylabel("severitate")
    ax2.set_xlabel("timp (s)")
    ax2.legend(loc="upper left", frameon=False, ncol=2)
    for d in run["decisions"]:
        if any(a["tool"] == "call_112" for a in d["actions"]):
            ax2.axvline(d["ts"], color="#d03b3b", lw=1)
            ax2.text(d["ts"] + 1, 3.2, "112", color="#d03b3b", fontsize=9)
    fig.tight_layout()
    f = FIG / "trip_example.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)
    return files


# ------------------------------------------------------------ main ----

def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    n = {"train": 4, "val": 3, "test": 3, "shift": 3} if args.quick else {"train": 20, "val": 8, "test": 10, "shift": 8}
    drivers = {k: make_drivers(v, rng, k[:2], shift=(k == "shift")) for k, v in n.items()}
    sims = {k: [TripSim(d, make_trip(rng), rng) for d in ds] for k, ds in drivers.items()}

    # Profilul de praguri (ml1/profile.py) ales pe VALIDARE: costul minim (Curs 5)
    profile_val = {}
    runs_legacy = {}
    for name in ("legacy", "cv", "cv_signs"):
        profile.use(name)
        rv = [run_system(s) for s in sims["val"]]
        sv = summarize(name, sims["val"], rv, rng)
        profile_val[name] = {"macro_f1": sv["scores"]["macro_f1"], "cost": sv["scores"]["cost_per_frame"],
                             "false_alarms_per_hour": sv["false_alarms_per_hour"], "calls_112": sv["calls_112"]}
        if name == "legacy":
            runs_legacy = {k: [run_system(s) for s in sims[k]] for k in ("test", "shift")}
    chosen = min(profile_val, key=lambda n: profile_val[n]["cost"])
    profile.use(chosen)

    runs = {k: [run_system(s) for s in sims[k]] for k in sims}
    runs_nocal = {k: [run_system(s, calibrate=False) for s in sims[k]] for k in ("test", "shift")}

    results: dict = {"config": {"seed": args.seed, "drivers": n, "hz": HZ,
                                "frames": {k: int(sum(len(s.frames) for s in v)) for k, v in sims.items()}},
                     "profile_val": profile_val, "profile_chosen": chosen}

    # pragurile pe a: alese pe validare
    cuts = tune_a_thresholds(sims["val"], runs["val"])
    results["a_cuts_from_val"] = cuts

    # modele ML: antrenare pe train, standardizare fit pe train
    def xy(split, rs):
        X = np.concatenate([r["X"] for r in rs])
        y = np.concatenate([np.array(s.labels) for s in sims[split]])
        m = y >= 0
        return X[m], y[m]

    Xtr, ytr = xy("train", runs["train"])
    std = Standardizer().fit(Xtr)
    sm = fit_softmax(std.transform(Xtr), ytr)
    tree = Tree(depth=4, min_leaf=60).fit(Xtr, ytr)
    results["tree_rules"] = tree.rules(FEATURE_NAMES)

    for split in ("test", "shift"):
        S, R = sims[split], runs[split]
        res = {}
        res["clasa majoritară (0)"] = frame_summary("majoritar", S, [np.zeros(len(s.frames), int) for s in S], rng)
        res["prag pe a"] = frame_summary("prag pe a", S, [a_threshold_predict(r["a"], cuts) for r in R], rng)
        res["softmax"] = frame_summary("softmax", S, [softmax_proba(std.transform(r["X"]), sm).argmax(1) for r in R], rng)
        res["arbore"] = frame_summary("arbore", S, [tree.predict(r["X"]) for r in R], rng)
        res["reguli, praguri vechi (legacy)"] = summarize("legacy", S, runs_legacy[split], rng)
        res["rules fără calibrare"] = summarize("rules fără calibrare", S, runs_nocal[split], rng)
        res["rules+calibrare"] = summarize("rules+calibrare", S, R, rng)
        res["rules+calibrare"]["_runs"] = R
        if args.llm and split == "test":
            from .llm_client import OllamaClient
            llm = OllamaClient()
            if llm.available():
                llm.warmup()
                rl = [run_system(s, llm=llm) for s in S]
                res["LLM+reguli+calibrare"] = summarize("LLM", S, rl, rng)
                res["LLM+reguli+calibrare"]["llm_calls"] = sum(r["llm_calls"] for r in rl)
        results[split] = res

    # ROC/PR pentru 1-a și calibrarea softmax, pe test
    y = np.concatenate([np.array(s.labels) for s in sims["test"]])
    a = np.concatenate([r["a"] for r in runs["test"]])
    m = y >= 0
    roc = roc_pr((y[m] >= 2).astype(int), 1 - a[m])
    Xte = np.concatenate([r["X"] for r in runs["test"]])[m]
    p2 = softmax_proba(std.transform(Xte), sm)[:, 2:].sum(1)
    yb = (y[m] >= 2).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p2, bins) - 1, 0, 9)
    rel = {"conf": [float(p2[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 20],
           "freq": [float(yb[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 20],
           "brier": float(np.mean((p2 - yb) ** 2))}
    results["roc_attention"] = {"auc": roc["auc"], "ap": roc["ap"]}
    results["softmax_calibration"] = {"brier": rel["brier"]}

    if not args.no_write:
        ex = next((i for i, s in enumerate(sims["test"]) if s.emergency_onset is not None), 0)
        results["figures"] = plot_all(sims["test"], results["test"], roc, rel, ex)
    for split in ("test", "shift"):
        results[split]["rules+calibrare"].pop("_runs", None)
    profile.use("cv")
    results["elapsed_s"] = round(time.time() - t_start, 1)
    if not args.no_write:
        write_report(results)
    print_summary(results)
    return results


def _fmt(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _row(name, r):
    s = r["scores"]
    lo, hi = r["macro_f1_ci"]
    r3 = s["recall"][3]
    e_lo, e_hi = r["recall3_ci"]
    fa = r.get("false_alarms_per_hour")
    c = r.get("calls_112")
    lat = r.get("latency", {})
    lat3 = lat.get(3, {}).get("median_s") if lat else None
    return (f"| {name} | {s['macro_f1']:.3f} [{lo:.2f}–{hi:.2f}] | {r3:.2f} [{e_lo:.2f}–{e_hi:.2f}] | "
            f"{s['cost_per_frame']:.3f} | {_fmt(fa, 1)} | {_fmt(lat3, 1)} | "
            f"{(str(c['tp']) + '/' + str(c['tp'] + c['fn']) + ' · fals ' + str(c['fp'])) if c else '–'} |")


def write_report(res: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "evaluation_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = ["# Rezultatele evaluării ML1 (generat automat)", "",
         f"Generat de `python -m ml1.evaluate` (seed {res['config']['seed']}, {res['elapsed_s']} s). "
         "Protocolul și limitele sunt în [EVALUATION.md](EVALUATION.md). **Datele sunt simulate**: "
         "cifrele compară variante între ele, nu prezic performanța pe șoferi reali.", "",
         f"Șoferi: {res['config']['drivers']} · cadre: {res['config']['frames']} · {res['config']['hz']} Hz", ""]
    head = ("| Variantă | macro-F1 [IC 95%] | recall severitate 3 [IC 95%] | cost / cadru ↓ | alarme false / oră ↓ | "
            "latență urgență (s, mediană) | 112 corecte · false |")
    for split, title in (("test", "Test (șoferi standard, nevăzuți)"), ("shift", "Shift (ochi mici, zgomot, fața pierdută des)")):
        L += [f"## {title}", "", head, "|---|---|---|---|---|---|---|"]
        for name, r in res[split].items():
            L.append(_row(name, r))
        L.append("")
        r = res[split]["rules+calibrare"]
        L += ["Recall pe clasă (rules+calibrare): " + ", ".join(f"{SEV_NAMES[k]}: {r['scores']['recall'][k]:.2f}" for k in range(K)),
              "", "Precizie pe clasă (rules+calibrare): " + ", ".join(f"{SEV_NAMES[k]}: {r['scores']['precision'][k]:.2f}" for k in range(K)), ""]
        lat = r["latency"]
        L.append("Latența de detecție (rules+calibrare): " + "; ".join(
            f"severitate {lv}: mediană {_fmt(lat[lv]['median_s'], 1)} s, p90 {_fmt(lat[lv]['p90_s'], 1)} s, ratate {lat[lv]['missed']}/{lat[lv]['n']}"
            for lv in (1, 2, 3)))
        L.append("")
    pv = res.get("profile_val", {})
    if pv:
        L += ["## Alegerea profilului de praguri (pe validare)", "",
              "| Profil | macro-F1 | cost / cadru | alarme false / oră | 112 corecte · ratate · false |", "|---|---|---|---|---|"]
        for name, v in pv.items():
            c = v["calls_112"]
            L.append(f"| {name}{' **(ales)**' if name == res['profile_chosen'] else ''} | {v['macro_f1']:.3f} | "
                     f"{v['cost']:.3f} | {_fmt(v['false_alarms_per_hour'], 1)} | {c['tp']} · {c['fn']} · {c['fp']} |")
        L.append("")
    L += [f"Praguri pe scorul a alese pe validare (cost minim): {res['a_cuts_from_val']} "
          "(sub primul = 1, sub al doilea = 2, sub al treilea = 3).", "",
          f"Scorul 1 − a ca detector pentru severitate ≥ 2 (test): ROC-AUC {res['roc_attention']['auc']:.3f}, "
          f"AP {res['roc_attention']['ap']:.3f}. Calibrarea softmax: Brier {res['softmax_calibration']['brier']:.3f}.", "",
          "## Arborele de decizie învățat (reguli citibile)", "", "```", *res["tree_rules"], "```", ""]
    for f in res.get("figures", []):
        L += [f"![{f}](figures/{f})", ""]
    (DOCS / "EVALUATION_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def print_summary(res: dict) -> None:
    for split in ("test", "shift"):
        print(f"\n== {split}")
        for name, r in res[split].items():
            s = r["scores"]
            extra = ""
            if "calls_112" in r:
                c = r["calls_112"]
                extra = f"  FA/h={_fmt(r['false_alarms_per_hour'], 1)}  112 tp/fn/fp={c['tp']}/{c['fn']}/{c['fp']}"
            print(f"  {name:<24} macroF1={s['macro_f1']:.3f}  recall3={s['recall'][3]:.2f}  cost={s['cost_per_frame']:.3f}{extra}")
    print("\nprofil (val):", {k: (round(v['macro_f1'], 3), round(v['cost'], 3)) for k, v in res['profile_val'].items()},
          "ales:", res['profile_chosen'])
    print(f"\npraguri a (val): {res['a_cuts_from_val']}   AUC(1-a)={res['roc_attention']['auc']:.3f}   {res['elapsed_s']} s")


if __name__ == "__main__":
    main()
