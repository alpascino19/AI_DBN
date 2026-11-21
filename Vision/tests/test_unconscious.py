"""Teste sintetice pentru pierderea cunoștinței (fără cameră)."""
from vigil_vision.face.alerts import AlertManager
from vigil_vision.face.eyes import EyeState
from vigil_vision.face.head_pose import CENTER, DOWN, HeadState
from vigil_vision.face.unconscious import OK, SUSPECT, UNCONSCIOUS, UnconsciousAnalyzer

FPS = 30


def frames(unc, sec, t0, pitch, roll=0.0, closed=False, closed_from=None, visible=True, pitch_to=None):
    """Rulează `sec` secunde; pitch poate trece liniar la pitch_to."""
    t, n = t0, int(sec * FPS)
    for i in range(n):
        p = pitch if pitch_to is None else pitch + (pitch_to - pitch) * min(1.0, i / max(1, n - 1))
        eyes = EyeState(face_visible=visible, eyes_closed=closed,
                        closed_sec=(t - closed_from) if closed and closed_from is not None else 0.0)
        head = HeadState(face_visible=visible, pitch=p if visible else None, roll=roll if visible else None,
                         direction=(DOWN if p > 18 else CENTER) if visible else None)
        unc.update(eyes, head, t)
        t += 1 / FPS
    return t


def test_faint_head_drop_eyes_closed_still():
    unc = UnconsciousAnalyzer()
    t = frames(unc, 3, 0.0, pitch=0)                       # normal
    t_drop = t
    t = frames(unc, 0.3, t, pitch=0, pitch_to=40, closed=True, closed_from=t_drop)  # cădere bruscă
    t = frames(unc, 4, t, pitch=40, closed=True, closed_from=t_drop)               # nemișcat
    s = unc.state
    assert s.sudden and s.head_drop
    assert s.state == UNCONSCIOUS and s.score >= 0.65
    assert unc.detections()["unconscious"][0] and unc.detections()["head_drop"][0]


def test_looking_down_at_phone_is_not_unconscious():
    unc = UnconsciousAnalyzer()
    t = frames(unc, 2, 0.0, pitch=0)
    t = frames(unc, 2, t, pitch=0, pitch_to=30)            # privire în jos lentă, ochi deschiși
    frames(unc, 5, t, pitch=30)
    s = unc.state
    assert s.state != UNCONSCIOUS and not s.head_drop
    assert s.score < 0.65


def test_drowsy_eyes_closed_head_upright_is_at_most_suspect():
    unc = UnconsciousAnalyzer()
    frames(unc, 6, 0.0, pitch=0, closed=True, closed_from=0.0)
    assert unc.state.state in (OK, SUSPECT)
    assert not unc.state.head_drop


def test_head_falls_sideways():
    unc = UnconsciousAnalyzer()
    t = frames(unc, 2, 0.0, pitch=0, roll=0)
    frames(unc, 4, t, pitch=5, roll=45, closed=True, closed_from=t)
    assert unc.state.state == UNCONSCIOUS


def test_state_held_when_face_disappears_then_released():
    unc = UnconsciousAnalyzer()
    t = frames(unc, 2, 0.0, pitch=0)
    t = frames(unc, 4, t, pitch=40, closed=True, closed_from=t)
    assert unc.state.state == UNCONSCIOUS
    t = frames(unc, 5, t, pitch=0, visible=False)          # a căzut pe volan: fața dispare
    assert unc.state.state == UNCONSCIOUS                  # păstrat (hold_sec = 10)
    frames(unc, 6, t, pitch=0, visible=False)
    assert unc.state.state == OK                           # după hold_sec


def test_unconscious_alert_level_3_unless_phone():
    am = AlertManager()
    eyes = EyeState(face_visible=True, eyes_closed=True, closed_sec=5)
    head = HeadState(face_visible=True, direction=DOWN, gaze_on_road=False)
    dets = {"unconscious": (True, 0.8), "head_drop": (True, 0.8)}
    am.update(eyes, head, dets, 0.0)
    active = {a.code for a in am.active}
    assert "unconscious" in active and am.level == 3
    assert not active & {"eyes_closed_critical", "eyes_closed", "head_drop", "look_down"}
    am2 = AlertManager()
    am2.update(eyes, head, {**dets, "phone_use": (True, 0.6)}, 0.0)
    assert "unconscious" not in {a.code for a in am2.active}
