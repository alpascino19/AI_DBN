"""Teste pe date sintetice (fără cameră): python -m pytest tests/test_eyes.py"""
import numpy as np

from vigil_vision.face.eyes import EyeAnalyzer, eye_aspect_ratio

FPS = 30
OPEN, CLOSED = 0.30, 0.10


def run(eyes, ears, t0=0.0):
    t = t0
    for e in ears:
        eyes.update_ear(e, t)
        t += 1 / FPS
    return t


def test_ear_formula():
    # ochi de lățime 4 și înălțime 1 -> EAR = (1 + 1) / (2 * 4) = 0.25
    pts = np.array([[0, 0], [1, 0.5], [3, 0.5], [4, 0], [3, -0.5], [1, -0.5]], float)
    assert abs(eye_aspect_ratio(pts) - 0.25) < 1e-6


def test_blinks_counted_and_rate():
    eyes = EyeAnalyzer()
    # 20 s: o clipire de 0.15 s la fiecare 3 s -> ~20 / min
    seq = []
    for _ in range(20 // 3):
        seq += [OPEN] * int(2.85 * FPS) + [CLOSED] * int(0.15 * FPS)
    seq += [OPEN] * FPS
    run(eyes, seq)
    s = eyes.state
    assert s.blink_count == 6
    assert 15 <= s.blink_rate <= 25
    assert s.perclos < 0.1


def test_long_closure_not_a_blink_and_detected():
    eyes = EyeAnalyzer()
    t = run(eyes, [OPEN] * 11 * FPS)
    run(eyes, [CLOSED] * 2 * FPS, t)
    assert eyes.state.eyes_closed
    assert eyes.state.closed_sec >= 1.9
    active, conf = eyes.detections()["eyes_closed_long"]
    assert active and conf >= 0.5
    assert eyes.state.blink_count == 0


def test_low_blink_rate_detected():
    eyes = EyeAnalyzer()
    run(eyes, [OPEN] * 20 * FPS)  # 20 s fără clipit
    assert eyes.detections()["low_blink_rate"][0]


def test_calibration_changes_threshold():
    eyes = EyeAnalyzer()
    before = eyes.threshold
    eyes.set_open_baseline(0.24)  # șofer cu ochi mai mici
    assert eyes.ear_open == 0.24
    assert eyes.threshold < before


def test_half_open_eyes_are_not_closed():
    eyes = EyeAnalyzer()
    run(eyes, [0.19] * 3 * FPS)          # ochi întredeschiși, fără blendshapes
    assert not eyes.state.eyes_closed
    eyes2 = EyeAnalyzer()
    for i in range(3 * FPS):             # cu blendshapes: eyeBlink moderat
        eyes2.update_ear(0.17, i / FPS, blink=0.35)
    assert not eyes2.state.eyes_closed


def test_smile_does_not_close_eyes():
    # EAR scade mult la zâmbet, eyeBlink crește puțin
    no_smile, smile = EyeAnalyzer(), EyeAnalyzer()
    for i in range(2 * FPS):
        no_smile.update_ear(0.13, i / FPS, blink=0.45, smile=0.0)
        smile.update_ear(0.13, i / FPS, blink=0.45, smile=0.6)
    assert no_smile.state.eyes_closed      # fără zâmbet chiar arată închis
    assert not smile.state.eyes_closed     # cu zâmbet: compensat


def test_really_closed_while_smiling():
    eyes = EyeAnalyzer()
    for i in range(2 * FPS):
        eyes.update_ear(0.10, i / FPS, blink=0.9, smile=0.6)
    assert eyes.state.eyes_closed and eyes.state.closed_sec > 1.5


def test_closed_reference_learned_from_blinks():
    eyes = EyeAnalyzer()
    seq = []
    for _ in range(5):                    # clipiri care coboară doar la 0.16 (ochi mari)
        seq += [0.30] * 60 + [0.10, 0.10, 0.10, 0.16] + [0.30] * 10
    run(eyes, seq)
    assert eyes.state.blink_count == 5
    assert 0.07 <= eyes.ear_closed <= 0.168


def test_fatigue_clears_after_15s_of_normal_eyes():
    """15 s ochi închiși -> oboseală; după ~15 s normale dispare (nu după ~50 s)."""
    from vigil_vision.face.alerts import AlertManager
    from vigil_vision.face.head_pose import CENTER, HeadState

    eyes, am = EyeAnalyzer(), AlertManager()
    head = HeadState(face_visible=True, direction=CENTER, gaze_on_road=True)
    normal = [OPEN] * int(2.85 * FPS) + [CLOSED] * int(0.15 * FPS)  # o clipire la 3 s
    t, log = 0.0, []  # (t, fatigue activ, perclos 60 s)

    def feed(seq):
        nonlocal t
        for e in seq:
            eyes.update_ear(e, t)
            am.update(eyes.state, head, eyes.detections(), t)
            log.append((t, any(a.code == "fatigue" for a in am.active), eyes.state.perclos))
            t += 1 / FPS

    feed(normal * 5)              # 15 s normal
    feed([CLOSED] * 15 * FPS)     # 15 s ochi închiși
    assert log[-1][1], "oboseala trebuia să fie activă"
    t_open = t
    feed(normal * 10)             # 30 s normal

    after = [(ts, on, pc) for ts, on, pc in log if ts >= t_open]
    assert any(on for ts, on, _ in after if ts < t_open + 5)  # nu dispare instant
    t_clear, _, perclos_at_clear = next(x for x in after if not x[1])
    assert 10.0 <= t_clear - t_open <= 18.0                    # ~15 s, nu ~50 s
    assert perclos_at_clear >= 0.15                            # fereastra de 60 s încă "obosită"
