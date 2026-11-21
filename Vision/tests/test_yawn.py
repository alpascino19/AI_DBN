"""Teste sintetice pentru căscat (fără cameră)."""
from vigil_vision.face.alerts import AlertManager
from vigil_vision.face.eyes import EyeState
from vigil_vision.face.head_pose import CENTER, HeadState
from vigil_vision.face.yawn import YawnAnalyzer

FPS = 30
CLOSED, WIDE, TALK = (0.05, 0.05), (0.8, 0.8), (0.6, 0.6)


def run(y, val, sec, t0):
    t = t0
    for _ in range(int(sec * FPS)):
        y.update_values(val[0], val[1], t)
        t += 1 / FPS
    return t


def test_yawn_counted():
    y = YawnAnalyzer()
    t = run(y, CLOSED, 1, 0.0)
    t = run(y, WIDE, 2.5, t)
    assert y.state.yawning and y.detections()["yawning"][0]
    run(y, CLOSED, 1, t)
    assert y.state.yawn_count == 1 and not y.state.yawning
    assert 2.3 <= y.state.last_yawn_sec <= 2.7


def test_talking_is_not_a_yawn():
    y = YawnAnalyzer()
    t = 0.0
    for _ in range(10):  # deschideri scurte de 0.3 s, ca la vorbit
        t = run(y, TALK, 0.3, t)
        t = run(y, CLOSED, 0.6, t)
    assert y.state.yawn_count == 0


def test_very_long_open_is_not_a_yawn():
    y = YawnAnalyzer()
    t = run(y, WIDE, 10, 0.0)
    assert not y.state.yawning  # > max_sec
    run(y, CLOSED, 1, t)
    assert y.state.yawn_count == 0


def test_short_gap_does_not_split_yawn():
    y = YawnAnalyzer()
    t = run(y, WIDE, 1.0, 0.0)
    t = run(y, CLOSED, 0.2, t)   # sub gap_sec (0.4 s)
    t = run(y, WIDE, 1.0, t)
    run(y, CLOSED, 1, t)
    assert y.state.yawn_count == 1


def test_frequent_yawning_and_alerts():
    y, am = YawnAnalyzer(), AlertManager()
    eyes = EyeState(face_visible=True)
    head = HeadState(face_visible=True, direction=CENTER, gaze_on_road=True)
    codes, t = [], 0.0
    for _ in range(3):
        for val, sec in ((WIDE, 2.0), (CLOSED, 20.0)):
            for _ in range(int(sec * FPS)):
                y.update_values(val[0], val[1], t)
                codes += [a.code for a in am.update(eyes, head, y.detections(), t)]
                t += 1 / FPS
    assert y.state.yawns_recent == 3
    assert codes.count("yawn") == 3          # o avertizare pe căscat
    assert "frequent_yawning" in codes
