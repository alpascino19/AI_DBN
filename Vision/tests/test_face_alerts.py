"""Teste sintetice pentru head_pose, calibration și alerts (fără cameră)."""
import math

import numpy as np

from vigil_vision.face.alerts import AlertManager
from vigil_vision.face.calibration import Calibrator
from vigil_vision.face.eyes import EyeAnalyzer, EyeState
from vigil_vision.face.head_pose import CENTER, DOWN, LEFT, RIGHT, HeadPoseAnalyzer, HeadState, head_angles
from vigil_vision.perception.face import FaceResult, Idx

FPS = 30


def fake_face(yaw_deg=0.0, pitch_deg=0.0):
    """Față sintetică: colțuri ochi/gură într-un plan, nasul ieșit în față (z negativ)."""
    lm = np.zeros((478, 3), np.float32)
    pts = {Idx.RIGHT_EYE[0]: (-0.1, -0.05, 0), Idx.LEFT_EYE[3]: (0.1, -0.05, 0),
           Idx.MOUTH_LEFT: (-0.06, 0.08, 0), Idx.MOUTH_RIGHT: (0.06, 0.08, 0)}
    for i, p in pts.items():
        lm[i] = p
    d = 0.06  # cât iese nasul
    y, pt = math.radians(yaw_deg), math.radians(pitch_deg)
    lm[Idx.NOSE_TIP] = (d * math.sin(y), 0.015 + d * math.sin(pt), -d * math.cos(y) * math.cos(pt))
    lm[:, :2] += 0.5
    return FaceResult(frame_id=0, timestamp=0.0, width=640, height=640, landmarks=lm)


def test_head_angle_signs():
    yaw, pitch, _ = head_angles(fake_face(yaw_deg=30))
    assert yaw > 20
    yaw, pitch, _ = head_angles(fake_face(yaw_deg=-30))
    assert yaw < -20
    _, p0, _ = head_angles(fake_face())
    _, p1, _ = head_angles(fake_face(pitch_deg=30))
    assert p1 - p0 > 20  # în jos = pitch crește


def run_head(head, yaw, pitch, sec, t0, iris=0.0):
    t = t0
    for _ in range(int(sec * FPS)):
        head.update_angles(yaw, pitch, 0.0, iris, t)
        t += 1 / FPS
    return t


def test_directions_and_off_road():
    h = HeadPoseAnalyzer()
    t = run_head(h, 0, 0, 1, 0.0)
    assert h.state.direction == CENTER and h.state.gaze_on_road
    t = run_head(h, 40, 0, 1, t)
    assert h.state.direction == RIGHT and h.state.off_road_sec > 0.5
    t = run_head(h, -40, 0, 1, t)
    assert h.state.direction == LEFT
    run_head(h, 0, 30, 1, t)
    assert h.state.direction == DOWN
    # iris singur (cap drept, ochii la dreapta)
    h2 = HeadPoseAnalyzer()
    run_head(h2, 0, 0, 1, 0.0, iris=0.3)
    assert h2.state.direction == RIGHT


def test_calibration_neutral():
    eyes, head = EyeAnalyzer(), HeadPoseAnalyzer()
    cal = Calibrator(duration_sec=2.0)
    t = 0.0
    while not cal.done:
        es = eyes.update_ear(0.28, t)
        hs = head.update_angles(5.0, 12.0, 0.0, 0.0, t)  # camera mai jos -> pitch 12 la neutru
        cal.update(es, hs, t)
        t += 1 / FPS
    cal.apply(eyes, head)
    assert abs(cal.result.pitch0 - 12.0) < 0.5
    hs = head.update_angles(5.0, 12.0, 0.0, 0.0, t)
    assert hs.direction == CENTER  # fără calibrare ar fi fost aproape de "down"


def test_alerts_eyes_closed_escalates():
    am = AlertManager()
    head = HeadState(face_visible=True, direction=CENTER, gaze_on_road=True)
    codes = []
    for i in range(int(4 * FPS)):
        t = i / FPS
        eyes = EyeState(face_visible=True, eyes_closed=True, closed_sec=t)
        codes += [a.code for a in am.update(eyes, head, {}, t)]
    assert "eyes_closed" in codes and "eyes_closed_critical" in codes
    assert am.level == 3


def test_alerts_look_left_after_2s_only():
    am = AlertManager()
    eyes = EyeState(face_visible=True)
    head = HeadState(face_visible=True, direction=LEFT, gaze_on_road=False)
    assert am.update(eyes, head, {}, 0.0) == []
    assert am.update(eyes, head, {}, 1.5) == []
    new = am.update(eyes, head, {}, 2.1)
    assert [a.code for a in new] == ["look_left"]
    # nu se repetă imediat
    assert am.update(eyes, head, {}, 2.5) == []


def test_alerts_no_face():
    am = AlertManager()
    eyes, head = EyeState(), HeadState(face_visible=False)
    am.update(eyes, head, {}, 0.0)
    assert [a.code for a in am.update(eyes, head, {}, 3.5)] == ["no_face"]


def test_eyes_only_looking_down_is_down():
    """Telefon ținut jos: capul aproape drept, doar ochii în jos."""
    h = HeadPoseAnalyzer()
    t = 0.0
    for _ in range(FPS):
        h.update_angles(0.0, 5.0, 0.0, 0.0, t, eye_down=0.6)
        t += 1 / FPS
    assert h.state.direction == DOWN and not h.state.gaze_on_road


def test_eye_down_baseline_from_calibration():
    """Unii șoferi au eyeLookDown ~0.3 privind drumul (camera sus): nu e telefon."""
    eyes, head = EyeAnalyzer(), HeadPoseAnalyzer()
    cal = Calibrator(duration_sec=2.0)
    t = 0.0
    while not cal.done:
        cal.update(eyes.update_ear(0.28, t), head.update_angles(0, 0, 0, 0, t, eye_down=0.3), t)
        t += 1 / FPS
    cal.apply(eyes, head)
    for _ in range(FPS):
        hs = head.update_angles(0, 0, 0, 0, t, eye_down=0.3)
        t += 1 / FPS
    assert hs.direction == CENTER
    for _ in range(FPS):
        hs = head.update_angles(0, 0, 0, 0, t, eye_down=0.8)
        t += 1 / FPS
    assert hs.direction == DOWN


def test_phone_gives_look_down_not_eyes_closed():
    am = AlertManager()
    eyes = EyeState(face_visible=True, eyes_closed=True)   # pleoapele coborâte par închise
    head = HeadState(face_visible=True, direction=DOWN, gaze_on_road=False, off_road_ratio=1.0)
    codes = []
    for i in range(int(2.5 * FPS)):
        t = i / FPS
        eyes.closed_sec = t
        codes += [a.code for a in am.update(eyes, head, {}, t)]
    assert "look_down" in codes and "distracted" in codes
    assert "eyes_closed" not in codes


# ---------- reguli pentru corp (detecții CV2) ----------
def _run_body(dets_fn, sec, head_dir=CENTER, face=True):
    am = AlertManager()
    eyes = EyeState(face_visible=face)
    head = HeadState(face_visible=face, direction=head_dir if face else None,
                     gaze_on_road=head_dir == CENTER if face else None)
    codes = set()
    for i in range(int(sec * FPS)):
        codes |= {a.code for a in am.update(eyes, head, dets_fn(), i / FPS)}
    return am, codes


def test_hand_on_chest_long_is_level_3_and_hides_level_2():
    am, codes = _run_body(lambda: {"hand_on_chest": (True, 0.8)}, 7)
    assert "chest_long" in codes and am.level == 3
    assert "hand_on_chest" not in {a.code for a in am.active}


def test_hand_on_chest_alone_is_level_2():
    am, codes = _run_body(lambda: {"hand_on_chest": (True, 0.8)}, 3)
    assert "hand_on_chest" in codes and am.level == 2


def test_head_on_wheel_hides_no_face():
    am, codes = _run_body(lambda: {"head_on_wheel": (True, 0.9)}, 5, face=False)
    active = {a.code for a in am.active}
    assert "head_on_wheel" in active and "no_face" not in active and am.level == 3


def test_phone_while_looking_away_is_level_2():
    am, codes = _run_body(lambda: {"phone_use": (True, 0.6)}, 2, head_dir=DOWN)
    active = {a.code for a in am.active}
    assert "phone_looking" in active and "phone_use" not in active and "look_down" not in active
    am2, _ = _run_body(lambda: {"phone_use": (True, 0.6)}, 2, head_dir=CENTER)
    assert {a.code for a in am2.active} == {"phone_use"} and am2.level == 1


def test_frequent_yawning_shown_at_most_30s():
    am, e, h = AlertManager(), EyeState(face_visible=True), HeadState(face_visible=True, direction=CENTER,
                                                                      gaze_on_road=True, off_road_ratio=0.0)
    dets = {"frequent_yawning": (True, 0.9)}
    assert any(w.code == "frequent_yawning" for w in am.update(e, h, dets, 0.0))
    am.update(e, h, dets, 20.0)
    assert any(w.code == "frequent_yawning" for w in am.active)
    am.update(e, h, dets, 31.0)
    assert not any(w.code == "frequent_yawning" for w in am.active)
    assert not am.update(e, h, dets, 130.0)          # nici sunetul de repetare nu mai vine


def test_posture_rules():
    am, _ = _run_body(lambda: {"slumped_side": (True, 0.8)}, 3, face=False)
    active = {a.code for a in am.active}
    assert "slumped_side" in active and "no_face" not in active and am.level == 3
    am, _ = _run_body(lambda: {"out_of_position": (True, 0.7)}, 4, face=False)
    assert {a.code for a in am.active} == {"out_of_position"} and am.level == 1
    am, _ = _run_body(lambda: {"out_of_position": (True, 0.7)}, 11, face=False)
    assert {a.code for a in am.active} == {"out_of_position_long"} and am.level == 2
    am, codes = _run_body(lambda: {"bad_posture": (True, 0.6)}, 9)
    assert "bad_posture" not in codes
    am, codes = _run_body(lambda: {"bad_posture": (True, 0.6)}, 11)
    assert "bad_posture" in codes and am.level == 1
