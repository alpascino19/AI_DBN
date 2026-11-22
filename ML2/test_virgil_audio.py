"""Teste audio pentru Virgil, fără microfon: voce sintetizată -> Whisper -> rutarea întrebării.
Verifică ce aude Virgil și unde trimite întrebarea (TomTom, vreme, căutare...), nu răspunsul modelului.
Vocile: edge-tts Emil și Alina (naturale, cer internet; sărite fără rețea) și Piper (offline, mai robotică).
    python -m pytest -q test_virgil_audio.py        (~1-2 min: încarcă Whisper)
"""
import zlib

import numpy as np
import pytest

import actions
import info_assistant as ia
from test_voice import tts as piper_tts

# (fraza vorbită, sursa așteptată, ce trebuie să conțină destinația / plecarea după Whisper; None = nu se verifică)
CASES = [
    ("Hei Virgil, ce vreme e azi?", {"weather"}, None, None),
    ("Hei Virgil, cum e traficul?", {"traffic"}, None, None),
    ("Hei Virgil, cât fac până la gară?", {"eta"}, "gar", None),
    ("Hei Virgil, cum ajung la Iulius Mall?", {"eta"}, "iulius", None),
    ("Hei Virgil, cât fac de la Piața Unirii până la aeroport?", {"eta"}, "aeroport", "unirii"),
    ("Hei Virgil, care e distanța dintre Cluj și Bacău?", {"eta"}, "bac", "cluj"),
    ("Hei Virgil, unde mă aflu?", {"where"}, None, None),
    ("Hei Virgil, ce faci?", {"chat"}, None, None),
    ("Hei Virgil, cine e președintele României?", {"search"}, None, None),
]
# Piper e robotică („Undemoflu” pentru „unde mă aflu”): doar informativ, nu potrivim Virgil după greșelile ei
VOICES = ["ro-RO-EmilNeural", "ro-RO-AlinaNeural",
          pytest.param("piper", marks=pytest.mark.xfail(strict=False, reason="voce robotică, doar informativ"))]


def synth(text, voice, gain=0.5, rate="-5%"):
    """Fraza la 16 kHz, cu o secundă de liniște înainte și după, ca la microfon. `rate`: viteza vorbirii (edge-tts)."""
    if voice == "piper":
        return piper_tts(text, gain)
    old = actions.EDGE_VOICE, actions.EDGE_TIMEOUT_SEC, actions.EDGE_RATE
    actions.EDGE_VOICE, actions.EDGE_TIMEOUT_SEC, actions.EDGE_RATE = voice, 20, rate   # în teste contează vocea
    try:
        pcm, sr = actions._edge_tts(text)
    except Exception as e:
        pytest.skip(f"edge-tts indisponibil (internet?): {e!r}")
    finally:
        actions.EDGE_VOICE, actions.EDGE_TIMEOUT_SEC, actions.EDGE_RATE = old
    a = pcm.astype(np.float32) / 32768
    a = np.interp(np.arange(0, len(a), sr / actions.MIC_SR), np.arange(len(a)), a)
    a *= gain / (np.abs(a).max() or 1)
    pad = np.zeros(actions.MIC_SR, np.float32)
    return np.concatenate([pad, a, pad]).astype(np.float32)


def hear(text, voice):
    """Ce înțelege Virgil din fraza rostită aproape de microfon, cu zgomot de fond (același la fiecare rulare)."""
    audio = synth(text, voice)
    rng = np.random.default_rng(zlib.crc32(f"{voice}{text}".encode()))
    heard, _, _ = actions.transcribe(audio + (rng.standard_normal(len(audio)) * 0.003).astype(np.float32))
    return heard


@pytest.mark.parametrize("voice", VOICES)
@pytest.mark.parametrize("spoken,topics,dest,start", CASES, ids=[c[0][11:40] for c in CASES])
def test_spoken_question_routed(spoken, topics, dest, start, voice):
    heard = hear(spoken, voice)
    assert actions.heard_wake_word(heard), f"nu a auzit „Hei Virgil”: {heard!r}"
    question = ia.strip_wake(heard)
    assert ia.route(question) == topics, f"auzit {heard!r} -> {ia.route(question)}"
    if dest:
        assert dest in ia.plain(ia.destination(question) or ""), f"destinație greșită din {heard!r}"
    if start:
        assert start in ia.plain(ia.origin(question) or ""), f"plecare greșită din {heard!r}"


def test_silence_is_not_a_question():
    pad = np.zeros(actions.MIC_SR * 3, np.float32)
    rng = np.random.default_rng(0)
    heard, _, _ = actions.transcribe(pad + (rng.standard_normal(len(pad)) * 0.003).astype(np.float32))
    assert not heard, f"a auzit ceva în liniște: {heard!r}"


@pytest.mark.parametrize("voice", ["ro-RO-EmilNeural", "ro-RO-AlinaNeural"])
def test_thanks_ends_conversation(voice):
    assert ia.ends_conversation(hear("Mersi, e suficient.", voice))


# ---- om real în mașină: zgomot de motor și drum, ecou de habitaclu, vorbire rapidă / lentă, ezitări ----

def in_car(audio, rng, snr_db=12.0, echo_sec=0.25):
    """Vocea ca la microfonul laptopului în mers: reverberație scurtă + zgomot de joasă frecvență (motor, rulare)."""
    n = int(actions.MIC_SR * echo_sec)
    impulse = rng.standard_normal(n) * np.exp(-6 * np.arange(n) / n) * 0.3
    impulse[0] = 1.0
    size = len(audio) + n
    wet = np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(impulse, size), size)[:len(audio)]
    rumble = np.cumsum(rng.standard_normal(len(audio)))           # zgomot brun: energia la frecvențe joase
    rumble -= np.convolve(rumble, np.ones(800) / 800, "same")      # fără deriva lentă, rămâne huruitul
    speech = wet[np.abs(wet) > 0.01]
    p_speech = np.mean(speech ** 2) if len(speech) else 1e-4
    rumble *= np.sqrt(p_speech / 10 ** (snr_db / 10) / np.mean(rumble ** 2))
    return (wet + rumble).astype(np.float32)


HUMAN_CASES = [
    ("Hei Virgil, ăăă, cât fac până la gară?", {"eta"}),
    ("Hei Virgil, spune-mi te rog vremea în Alba Iulia.", {"weather"}),
    ("Hei Virgil, cum e traficul acum?", {"traffic"}),
    ("Hei Virgil, unde mă aflu?", {"where"}),
    ("Hei Virgil, cine e președintele României?", {"search"}),
]
CONDITIONS = {"în mers": dict(rate="-5%"), "rapid": dict(rate="+20%"), "lent": dict(rate="-25%")}


@pytest.mark.parametrize("condition", list(CONDITIONS))
@pytest.mark.parametrize("voice", ["ro-RO-EmilNeural", "ro-RO-AlinaNeural"])
@pytest.mark.parametrize("spoken,topics", HUMAN_CASES, ids=[c[0][11:36] for c in HUMAN_CASES])
def test_human_in_car(spoken, topics, voice, condition):
    rng = np.random.default_rng(zlib.crc32(f"{voice}{condition}{spoken}".encode()))
    heard, _, details = actions.transcribe(in_car(synth(spoken, voice, **CONDITIONS[condition]), rng))
    assert actions.heard_wake_word(heard), f"nu a auzit „Hei Virgil”: {heard!r} {details}"
    assert ia.route(ia.strip_wake(heard)) == topics, f"auzit {heard!r} -> {ia.route(ia.strip_wake(heard))}"
