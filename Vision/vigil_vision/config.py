"""Încarcă config.yaml (praguri, fps, cameră, MQTT). [comun]

    cfg = load_config()            # caută Vision/config.yaml
    cfg["chest"]["zone_bottom"]

Valorile din DEFAULTS sunt folosite când lipsesc din fișier, deci fișierul
poate conține doar ce vrei să schimbi.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent          # .../Vision
DEFAULT_PATH = ROOT / "config.yaml"

DEFAULTS: dict[str, Any] = {
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 30, "mirror": True},
    "mqtt": {"enabled": True, "host": "localhost", "port": 1883,
             "topic": "vigil/vision", "print_fallback": True},
    "publish_hz": 10,
    "models": {"dir": "models", "pose": "pose_landmarker_lite.task",
               "hand": "hand_landmarker.task", "yolo": "yolov8n.pt"},
    "view": {"mode": "three_quarter", "camera_side": "right", "head_ratio": 1.4,
             "calib_sec": 5, "unit_tau": 1.0},
    "perception": {"pose_min_conf": 0.5, "hand_min_conf": 0.5, "num_hands": 2},
    "yolo": {"every_n_frames": 5, "imgsz": 320, "conf": 0.35, "max_age_sec": 1.0},
    "chest": {"view_forward": {"front": 0.25, "three_quarter": 0.45, "side": 0.7},
              "view_back": {"front": 0.25, "three_quarter": 0.20, "side": 0.25}, "zone_top": 0.15, "zone_bottom": 0.90,
              "min_visibility": 0.5,
              "width_tau": 1.0, "pos_tau": 0.25, "corner_tau": 0.2,
              "face_radius": 0.45, "chin_margin": 0.10, "min_inside": 0.6, "tips_above_shoulder": 0.05,
              "move_window_sec": 0.5, "move_thr": 0.25, "depth_max": 0.0,
              "max_hand_scale": 0.35, "pose_scale_factor": 1.8},
    "breathing": {"source": "auto", "window_sec": 10, "min_window_sec": 5, "calib_sec": 5,
                  "baseline_floor": 0.003, "amp_ratio_low": 1.3, "amp_ratio_high": 3.0,
                  "rate_low_bpm": 16, "rate_high_bpm": 30, "score_enter": 0.55,
                  "score_exit": 0.40, "enter_evals": 4, "exit_evals": 6, "jaw_open_thr": 0.35,
                  "flow_width": 320, "min_texture_frac": 0.08},
    "phone": {"hand_pad": 0.25, "min_score": 0.4},
    "slump": {"calib_sec": 5, "default_gap": 0.55, "drop_low": 0.25, "drop_high": 0.6,
              "score_enter": 0.50, "score_exit": 0.38, "still_thr": 0.015,
              "still_min_sec": 2.0, "hold_sec": 15,
              "leaning_deg": 12, "leaning_drop": 0.2,
              "lean": {"torso_deg": [10, 35], "head_forward": [0.2, 0.6],
                       "shoulder_drop": [0.1, 0.5], "closer": [1.1, 1.4]}},
    "smoothing": {"min_cutoff": 1.5, "beta": 0.5},
    "events": {"gap_sec": 1.0,
               "min_duration_sec": {"hand_on_chest": 2.0, "heavy_breathing": 3.0,
                                    "phone_use": 1.0,
                                    "head_on_wheel": 2.0, "default": 1.0}},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    data = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _merge(DEFAULTS, data)


def model_path(cfg: dict, key: str) -> Path:
    """Calea absolută a unui model din config['models']."""
    d = Path(cfg["models"]["dir"])
    if not d.is_absolute():
        d = ROOT / d
    return d / cfg["models"][key]
