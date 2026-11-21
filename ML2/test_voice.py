"""Testează recunoașterea pe voce sintetizată (Piper), fără microfon: python test_voice.py"""
import numpy as np

import actions

rng = np.random.default_rng(0)


def tts(text, gain):
    from piper import PiperVoice
    voice = PiperVoice.load(str(actions.VOICE))
    a = np.concatenate([c.audio_int16_array for c in voice.synthesize(text)]).astype(np.float32) / 32768
    a = np.interp(np.arange(0, len(a), 22050 / 16000), np.arange(len(a)), a)  # 22.05 kHz -> 16 kHz
    pad = np.zeros(actions.MIC_SR, np.float32)
    return np.concatenate([pad, a * gain, pad]).astype(np.float32)


def noise(n):
    return (rng.standard_normal(n) * 0.003).astype(np.float32)


def test_near_voice_accepted():
    audio = tts("Da, sunt bine, mulțumesc.", 0.5)
    text, latency, _ = actions.transcribe(audio + noise(len(audio)))
    assert "bine" in text.lower() and latency is not None, text


def test_far_voice_rejected():
    audio = tts("Ce mai faci, ai văzut meciul aseară?", 0.5 * 10 ** (-30 / 20))  # 30 dB mai încet
    text, _, details = actions.transcribe(audio + noise(len(audio)))
    assert text == "", (text, details)


def test_silence_and_noise_give_nothing():
    for audio in (np.zeros(8 * actions.MIC_SR, np.float32), noise(8 * actions.MIC_SR) * 5):
        text, latency, _ = actions.transcribe(audio)
        assert text == "" and latency is None, text


def silence(sec):
    return np.zeros(int(sec * actions.MIC_SR), np.float32)


def test_utterance_state():
    speech = tts("Nu mă simt bine, mă doare capul de dimineață.", 0.5)[actions.MIC_SR:-actions.MIC_SR]  # fără pad
    # nimeni nu începe să vorbească în termen -> tăcere
    assert actions.utterance_state(silence(actions.LISTEN_SEC + 0.3)) == "silence"
    assert actions.utterance_state(silence(1)) == "wait"
    # vorbește încă / pauză scurtă în mijlocul frazei -> mai ascultăm (nu tăiem „Nu mă simt... bine”)
    assert actions.utterance_state(np.concatenate([silence(0.5), speech])) == "wait"
    assert actions.utterance_state(np.concatenate([silence(0.5), speech, silence(0.4)])) == "wait"
    # a terminat de vorbit -> gata, chiar dacă a început după secunda 3 sau fraza e mai lungă de 4 s
    assert actions.utterance_state(np.concatenate([silence(0.5), speech, silence(1.6)])) == "done"
    assert actions.utterance_state(np.concatenate([silence(3.5), speech, speech, silence(1.6)])) == "done"
    # vocea din fundal nu contează ca răspuns
    assert actions.utterance_state(np.concatenate([speech * 0.01, silence(actions.LISTEN_SEC)])) == "silence"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
