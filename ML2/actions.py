"""Uneltele ML 2: voce (edge-tts, rezervă Piper/pyttsx3), alarmă, conversație cu șoferul
(faster-whisper + LLM local prin Ollama), 112 simulat."""
import difflib
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import threading
import time
import urllib.request

import numpy as np
import sounddevice as sd

HERE = pathlib.Path(__file__).parent
VOICE = HERE / "voices" / "ro_RO-mihai-medium.onnx"
LISTEN_SEC = 4          # termenul în care șoferul trebuie să ÎNCEAPĂ să răspundă; altfel = tăcere (alarmă)
END_SILENCE_SEC = 1.0   # pauza după care a terminat; VAD-ul mai adaugă ~0.4 s, deci nu taie „Nu mă simt... bine”
MAX_UTTERANCE_SEC = 15  # o frază nu poate ține microfonul la nesfârșit
MIC_SR = 16000

# --- vocea: edge-tts (neural, cere internet; pleacă doar textul rostit), altfel Piper local ---
EDGE_VOICE = "ro-RO-EmilNeural"  # sau "ro-RO-AlinaNeural"
EDGE_RATE, EDGE_PITCH = "-5%", "-2Hz"  # puțin mai calm și mai grav, stil Jarvis
EDGE_TIMEOUT_SEC = 5

# --- conversația: LLM local prin Ollama ---
OLLAMA_URL = "http://localhost:11434/api/chat"
# același model și aceeași variabilă ca ML 1: pe un singur calculator, un singur model rămâne în placa video
LLM_MODEL = os.environ.get("VIGIL_MODEL", "qwen2.5:1.5b")
LLM_TIMEOUT_SEC = 15
MAX_TURNS = 3  # câte răspunsuri ale șoferului ascultăm într-o conversație
WAKE_WORDS = ("virgil", "vigil")  # „Hei Virgil” trezește asistentul; Whisper scrie numele în mai multe feluri
# Fără ghilimele românești în prompt: modelul le copia și închidea replica cu ” în loc de ", stricând JSON-ul.
PERSONA = """Ești Virgil, asistentul de bord al mașinii, calm și politicos ca JARVIS din Iron Man. \
Verifici cum se simte șoferul și îl ții atent la drum. Vorbești română naturală, la forma de politețe, \
fără vocative (nu spui domnule sau doamnă).

REGULI DE FORMĂ (replica e citită cu voce tare):
- O singură propoziție scurtă, maxim 12 cuvinte, fără introduceri de tipul înțeleg perfect sau pare că.
- Cel mult o întrebare, punctuală și deschisă, nu de tip da/nu: un da scurt se aude greu.
- Fără emoji, liste, simboluri sau ghilimele. Nu spune că ai sunat la 112 sau la medic. \
Nu promite ce mașina nu face (cafea, climatizare, oprire automată): doar sugerezi.

CE FACI, după ce spune șoferul (reply = replica ta, end = încheiem conversația):
- E obosit: pui o întrebare scurtă, de exemplu De când conduceți fără pauză? Aici end este false.
- După ce afli cât de obosit e: sugerezi pauza, de exemplu Vă propun o pauză la următoarea parcare. end true.
- Durere în piept, amețeală, nu poate respira, vorbește incoerent: Am alertat sistemul de siguranță, \
opriți în siguranță. end true.
- E bine sau încheie (mulțumesc, gata, atât): Perfect, drum bun. end true.
- Altă întrebare sau subiect: răspunzi scurt, fără să inventezi date, de exemplu Nu am date meteo, \
dar drumul pare liber. end true.
end false doar când chiar aștepți un răspuns: dacă șoferul tace, sună alarma."""
REPLY_SCHEMA = {"type": "object", "required": ["reply", "end"],
                "properties": {"reply": {"type": "string"}, "end": {"type": "boolean"}}}

# --- reglaje recunoaștere vocală ---
# "large-v3-turbo" înțelege mult mai bine frazele libere (ex. întrebările pentru Virgil), dar ~6 s/frază pe CPU
WHISPER_MODEL = os.environ.get("VIGIL_WHISPER", "small")
BEAM_SIZE = 5
# contextul ghidează Whisper spre română corectă. Fără răspunsuri-exemplu: pe audio scurt sau nesigur Whisper
# copiază promptul, iar un „Da, sunt bine” inventat ar putea ascunde o urgență.
# Numele e aici ca Whisper să scrie „Virgil”, nu „vidile”; pe liniște VAD-ul nu lasă nimic să ajungă la Whisper.
# doar substantive: o propoziție întreagă în prompt („Da, sunt bine”) ajungea copiată în transcrieri
PROMPT = ("Conversație în limba română între șofer și Virgil, asistentul de bord al mașinii. Hei Virgil. "
          "Cluj-Napoca, Iulius Mall, Piața Unirii, gară, aeroport, trafic, vreme, distanța dintre.")
# Silero VAD mai strict: ignoră murmurul și vorbirea scurtă din fundal
VAD = {"threshold": 0.6, "min_speech_duration_ms": 300, "min_silence_duration_ms": 500}
# vocea mai slabă de atât e considerată din fundal (alți oameni din sală).
# Măsurat pe microfonul laptopului: șoferul -20.7..-24 dBFS, voci/zgomote din jur -26..-39 dBFS.
# Alt microfon sau altă distanță: recalibrează cu voice_test.py ("speech_dbfs"), ~5 dB sub vocea normală.
NEAR_DBFS = -28
ALARM_SR = 22050

log = logging.getLogger("ml2")
_piper = None
_whisper = None
_edge_down_until = 0.0  # după o eroare de rețea nu mai încercăm edge-tts un minut


def _play(audio, sr, interrupt):
    """Redă audio; se oprește dacă vine o urgență. True = redat complet."""
    sd.play(audio, sr)
    if interrupt.wait(len(audio) / sr):
        sd.stop()
        return False
    sd.wait()
    return True


def _edge_tts(text):
    """Voce neurală Microsoft -> (int16, sample rate). mp3-ul e decodat cu ffmpeg."""
    import asyncio

    import edge_tts

    async def fetch():
        mp3 = b""
        async for chunk in edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH).stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        return mp3

    mp3 = asyncio.run(asyncio.wait_for(fetch(), EDGE_TIMEOUT_SEC))
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                          "-ar", "24000", "pipe:1"], input=mp3, capture_output=True, check=True).stdout
    if not pcm:
        raise RuntimeError("edge-tts a întors audio gol")
    return np.frombuffer(pcm, np.int16), 24000


def _synth(text):
    global _piper, _edge_down_until
    if time.monotonic() >= _edge_down_until:
        try:
            return _edge_tts(text)
        except Exception as e:
            log.warning("edge-tts indisponibil (%r), folosesc Piper pentru 60 s", e)
            _edge_down_until = time.monotonic() + 60
    if _piper is None:
        from piper import PiperVoice
        _piper = PiperVoice.load(str(VOICE))
    chunks = list(_piper.synthesize(text))
    return np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate


def _synth_interruptible(text, interrupt):
    """Sinteza (edge-tts poate dura secunde) pe un fir separat, ca o urgență să nu aștepte după ea.
    None = întrerupt; firul abandonat termină singur în fundal."""
    out = []

    def work():
        try:
            out.append(_synth(text))
        except Exception as e:
            out.append(e)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        if interrupt.wait(0.05):
            return None
    if isinstance(out[0], Exception):
        raise out[0]
    return out[0]


def speak(text, interrupt):
    log.info("[VOCE] %s", text)
    try:
        audio = _synth_interruptible(text, interrupt)
        return audio is not None and _play(*audio, interrupt)
    except Exception:
        log.exception("TTS a eșuat, folosesc pyttsx3")
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return not interrupt.is_set()


def play_alarm(level, interrupt):
    t = np.arange(int(ALARM_SR * 0.25)) / ALARM_SR
    beep = np.sin(2 * np.pi * 880 * t)
    gap = np.zeros(int(ALARM_SR * 0.15))
    if level == 1:
        audio = beep * 0.3
    elif level == 2:
        audio = np.concatenate([beep, gap] * 3) * 0.6
    else:  # sirenă două tonuri, 4 s
        n = np.arange(ALARM_SR * 4) / ALARM_SR
        freq = np.where(np.sin(2 * np.pi * 2 * n) > 0, 1000, 700)
        audio = np.sin(2 * np.pi * np.cumsum(freq) / ALARM_SR) * 0.9
    log.info("[ALARMĂ] nivel %d", level)
    _play(audio.astype(np.float32), ALARM_SR, interrupt)


def speech_dbfs(audio):
    """Volumul vorbirii: percentila 90 a RMS-ului pe ferestre de 30 ms (pauzele nu-l trag în jos)."""
    frames = audio[: len(audio) // 480 * 480].reshape(-1, 480)
    if not len(frames):
        return -120.0
    return float(20 * np.log10(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), 90) + 1e-9))


# Fraze din subtitrări video pe care Whisper le inventează pe audio scurt sau zgomot (văzute la noi:
# „Vă mulțumim pentru vizionare!” în loc de „Hei Virgil”). Un fals „a răspuns” ar ascunde o urgență.
HALLUCINATIONS = re.compile(r"pentru vizionare|abona[tț]i|amara\.org|subtitr|nu uita[tț]i s[aă] da[tț]i like")


def is_hallucination(text):
    return bool(HALLUCINATIONS.search(text.lower()))


def transcribe(audio):
    """audio float32 mono 16 kHz -> (text, secunde până la primul cuvânt, detalii pentru calibrare)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(audio, language="ro", beam_size=BEAM_SIZE, initial_prompt=PROMPT,
                                      vad_filter=True, vad_parameters=VAD, condition_on_previous_text=False)
    kept, rejected = [], []
    for s in segments:
        level = round(speech_dbfs(audio[int(s.start * MIC_SR):int(s.end * MIC_SR)]), 1)
        # Whisper inventează text pe zgomot; un fals „a răspuns” ar ascunde un șofer inconștient
        if s.no_speech_prob >= 0.5 or s.avg_logprob <= -1.0 or s.compression_ratio >= 2.4:
            rejected.append({"text": s.text.strip(), "dbfs": level, "why": "nesigur"})
        elif is_hallucination(s.text):
            rejected.append({"text": s.text.strip(), "dbfs": level, "why": "halucinație"})
        elif level < NEAR_DBFS:
            rejected.append({"text": s.text.strip(), "dbfs": level, "why": "departe"})
        else:
            kept.append((s, level))
    text = " ".join(s.text.strip() for s, _ in kept).strip()
    details = {"speech_dbfs": max((lvl for _, lvl in kept), default=None), "rejected": rejected}
    return text, (round(kept[0][0].start, 2) if text else None), details


def utterance_state(audio):
    """Decide din ce s-a înregistrat până acum: 'wait' = mai ascultăm, 'silence' = n-a început să vorbească
    în LISTEN_SEC, 'done' = a terminat fraza (END_SILENCE_SEC de liniște după vorbire)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    options = VadOptions(threshold=VAD["threshold"], min_speech_duration_ms=VAD["min_speech_duration_ms"],
                         min_silence_duration_ms=VAD["min_silence_duration_ms"])
    speech = [s for s in get_speech_timestamps(audio, options)
              if speech_dbfs(audio[s["start"]:s["end"]]) >= NEAR_DBFS]  # vocile din fundal nu contează
    elapsed = len(audio) / MIC_SR
    if not speech:
        return "silence" if elapsed >= LISTEN_SEC else "wait"
    if elapsed - speech[-1]["end"] / MIC_SR >= END_SILENCE_SEC or elapsed >= MAX_UTTERANCE_SEC:
        return "done"
    return "wait"


def record(interrupt):
    """Ascultă de la microfon până când șoferul termină fraza (sau tace LISTEN_SEC). None = întrerupt."""
    block = MIC_SR // 4  # verificăm de 4 ori pe secundă
    chunks = []
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", blocksize=block) as stream:
        while not interrupt.is_set():
            chunks.append(stream.read(block)[0][:, 0].copy())
            audio = np.concatenate(chunks)
            if utterance_state(audio) != "wait":
                return audio
    return None


def listen(interrupt):
    """Ascultă un răspuns -> (răspunsul pentru ML 1, audio, detalii), sau None dacă e întrerupt."""
    audio = record(interrupt)
    if audio is None:
        return None
    text, latency, details = transcribe(audio)
    # microfon saturat = voce distorsionată = transcriere greșită; se repară din volumul microfonului în Windows
    details["clipped_pct"] = round(float(np.mean(np.abs(audio) > 0.98)) * 100, 2)
    if details["clipped_pct"] > 0.1:
        log.warning("microfon saturat (%.2f%% din eșantioane): scădeți volumul microfonului", details["clipped_pct"])
    if details["rejected"]:
        log.info("[RESPINS] %s", details["rejected"])
    return {"answered": bool(text), "latency_sec": latency, "text": text or None}, audio, details


def chat(history, timeout=LLM_TIMEOUT_SEC):
    """Următoarea replică de la LLM-ul local -> (text, end), sau None dacă Ollama nu răspunde."""
    body = json.dumps({"model": LLM_MODEL, "messages": history, "stream": False, "format": REPLY_SCHEMA,
                       # context mic + replici scurte = mai mult din model pe GPU, răspuns mai rapid
                       "keep_alive": "30m", "options": {"temperature": 0.7, "num_ctx": 2048, "num_predict": 120}}).encode()  # 80 tăia JSON-ul
    request = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            out = json.loads(json.loads(r.read())["message"]["content"])
        reply = out["reply"].strip() if isinstance(out.get("reply"), str) else ""
        # fără întrebare nu așteptăm răspuns: altfel tăcerea după „Perfect, drum bun” ar porni alarma
        return (reply, out.get("end") is True or "?" not in reply) if reply else None
    except Exception as e:
        log.warning("LLM indisponibil (%r), închei conversația", e)
        return None


def warmup():
    """Încarcă Piper, Whisper și LLM-ul la pornire; altfel primul ask_driver întârzie ~15 s."""
    global _piper
    from piper import PiperVoice
    _piper = PiperVoice.load(str(VOICE))
    transcribe(np.zeros(MIC_SR, np.float32))
    chat([{"role": "user", "content": "Salut"}], timeout=180)  # prima încărcare de pe disc durează ~50 s
    log.info("modelele de voce sunt încărcate")


def converse(interrupt, publish, opening=None, max_turns=MAX_TURNS, heard=None, on_silence=None):
    """Conversație condusă de ce spune șoferul; replicile le generează LLM-ul local.
    opening: prima replică a asistentului. heard: ce a spus deja șoferul (ex. „Hei Virgil, ...”).
    max_turns=None: fără limită. on_silence: chemat când șoferul tace, apoi ascultăm din nou;
    fără el, tăcerea încheie (o escaladează ML 1). Se încheie și când LLM-ul spune end=true.
    Fiecare răspuns (sau tăcere) pleacă imediat pe vigil/driver_response, ca ML 1 să nu aștepte finalul."""
    history = [{"role": "system", "content": PERSONA}]
    say, turn = opening, 0
    while True:
        if heard is None:
            if say:
                history.append({"role": "assistant", "content": say})
                if not speak(say, interrupt):  # microfonul pornește doar după ce TTS-ul termină
                    return
            result = listen(interrupt)
            if result is None:
                return
            response = result[0]
            publish("vigil/driver_response", json.dumps(response, ensure_ascii=False))
            if not response["answered"]:
                if on_silence is None:
                    return
                on_silence()
                say = None
                continue
            heard = response["text"]
        log.info("[ȘOFER] %s", heard)
        turn += 1
        history.append({"role": "user", "content": heard})
        heard = None
        last = max_turns is not None and turn >= max_turns
        if last:
            history.append({"role": "system", "content": "Ultima replică: încheie politicos conversația."})
        nxt = chat(history)
        if nxt is None:
            return
        say, end = nxt
        if last or end:
            speak(say, interrupt)
            return


def heard_wake_word(text):
    """„Hei Virgil”, „hei vigil”, „Virgilă”... — potrivire aproximativă, Whisper nu scrie numele mereu la fel."""
    return any(difflib.SequenceMatcher(None, word, name).ratio() >= 0.75
               for word in re.findall(r"\w+", (text or "").lower()) for name in WAKE_WORDS)


def assistant(interrupt, publish, on_silence):
    """Asistentul vocal: în idle ascultă doar „Hei Virgil”; apoi conversație până când șoferul spune
    că e bine (LLM end=true), după care revine în idle. Tăcerea în conversație -> on_silence()."""
    while True:
        result = listen(interrupt)
        if result is None:
            return
        text = result[0]["text"]
        if text and heard_wake_word(text):
            log.info("[TREZIT] %s", text)
            converse(interrupt, publish, max_turns=None, heard=text, on_silence=on_silence)
            log.info("[IDLE] aștept „Hei Virgil”")


def ask_driver(question, interrupt, publish):
    converse(interrupt, publish, opening=question)


def call_112(report, interrupt):
    # doar simulat: nimic nu pleacă de pe laptop
    line = "=" * 60
    log.critical("\n%s\nAPEL 112 SIMULAT\n%s\n%s\nAVARII PORNITE (simulat)", line, report, line)
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    (out / f"112_{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text(report, encoding="utf-8")
    speak("Am pornit avariile și am trimis alerta de urgență, simulat. Rămâneți calm, sunt aici cu dumneavoastră.",
          interrupt)


def suggest_break(interrupt):
    speak(random.choice([
        "Văd că oboseala își spune cuvântul. Ce-ați zice de o pauză scurtă la următoarea parcare?",
        "Cred că v-ar prinde bine o cafea și cinci minute de mers pe jos. Următoarea parcare e o idee bună.",
        "Drumul nu fuge nicăieri. Vă propun o pauză scurtă, ca să ajungem amândoi întregi.",
    ]), interrupt)


ASSISTANT_TOPIC_REPLY = "vigil/assistant/reply"
# întrebările libere fără OmniRoute: o replică de la LLM-ul local, doar cu datele reale din info.facts()
ASSISTANT_FACTS = ("Răspunzi la o întrebare pusă de șofer din butonul de pe ecran. Folosește DOAR datele de mai jos; "
                   "dacă nu ai informația, spune scurt că nu o ai. end este true.\n")
# butoanele rapide, ca întrebări pentru asistentul de informare (info_assistant.py)
ASSISTANT_QUESTIONS = {"weather": "Cum e vremea acum?", "traffic": "Cum e traficul acum în zonă?"}
LOOKUP_TIMEOUT_SEC = 20   # OmniRoute + Tavily + Waze pot dura; peste atât răspundem din sursele offline


def interruptible(fn, interrupt, timeout):
    """Rulează fn() pe un fir separat și așteaptă rezultatul, dar renunță imediat la o alertă ML 1 (interrupt)
    sau după timeout -> None. Workerul executorului nu rămâne blocat într-o cerere HTTP cât sună o urgență."""
    box = {}

    def work():
        try:
            box["value"] = fn()
        except Exception as e:                  # noqa: BLE001
            log.warning("căutarea asistentului a eșuat: %r", e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while t.is_alive() and time.monotonic() < deadline:
        if interrupt.wait(0.05):
            return None
    return box.get("value")


def assistant_lookup(kind, question):
    """Răspunsul la o întrebare: kind = "time" | "weather" | "traffic" | None (întrebare liberă).
    Cu OmniRoute (OMNIROUTE_API_KEY): asistentul de informare al echipei (info_assistant.py: Waze, Tavily,
    știri, model mai bun), aceleași răspunsuri ca la „Hei Virgil”. Fără el sau dacă nu răspunde: info.py
    (ceas, Open-Meteo, estimare de trafic) și LLM-ul local pentru restul."""
    import info
    if kind == "time":
        return info.answer("time")
    if os.environ.get("OMNIROUTE_API_KEY"):
        import info_assistant
        reply = info_assistant.answer(question)
        if reply:
            return {"text": reply, "source": "asistentul de informare (OmniRoute)"}
    if kind in info.REQUESTS:
        return info.answer(kind)
    reply = chat([{"role": "system", "content": PERSONA},
                  {"role": "system", "content": ASSISTANT_FACTS + info.facts()},
                  {"role": "user", "content": question}])
    return {"text": reply[0], "source": "LLM local"} if reply else None


def assistant_request(request, interrupt, publish):
    """Cerere din dashboard (vigil/assistant): „listen” = Virgil ascultă o întrebare; „time” / „weather” /
    „traffic” = răspunde direct. Nu publică pe vigil/driver_response: ML 1 nu trebuie să ia o întrebare
    despre vreme drept răspunsul la „Sunteți bine?”. Tăcerea aici nu pornește nicio alarmă."""
    import info

    def status(state, **extra):
        publish(ASSISTANT_TOPIC_REPLY, json.dumps({"request": request, "status": state, "ts": time.time(), **extra},
                                                  ensure_ascii=False))

    heard = None
    if request == "listen":
        status("listening")
        if not speak("Vă ascult.", interrupt):
            return status("interrupted")
        result = listen(interrupt)
        if result is None:                   # întrerupt de o alertă ML 1
            return status("interrupted")
        heard = result[0]["text"]
        if not heard:
            text = "Nu v-am auzit. Apăsați din nou când doriți."
            status("speaking", text=text, source="")
            speak(text, interrupt)
            return status("done", text=text, source="")
        log.info("[ÎNTREBARE] %s", heard)
        kind, question = info.intent(heard), heard
    else:
        kind, question = request, ASSISTANT_QUESTIONS.get(request, "")

    status("working", heard=heard)
    out = interruptible(lambda: assistant_lookup(kind, question), interrupt, LOOKUP_TIMEOUT_SEC)
    if interrupt.is_set():
        return status("interrupted", heard=heard)
    if out is None and kind in info.REQUESTS:  # sursele online lente / căzute: răspunsul offline e instant
        out = info.answer(kind)
    text, source = (out["text"], out["source"]) if out else ("Nu am putut afla informația acum.", "indisponibil")
    status("speaking", heard=heard, text=text, source=source)
    if not speak(text, interrupt):
        return status("interrupted", heard=heard, text=text, source=source)
    status("done", heard=heard, text=text, source=source)


def run(tool, args, interrupt, publish):
    if tool == "assistant":
        assistant_request(args["request"], interrupt, publish)
    elif tool == "speak":
        speak(args["text"], interrupt)
    elif tool == "play_alarm":
        play_alarm(args["level"], interrupt)
    elif tool == "ask_driver":
        ask_driver(args["question"], interrupt, publish)
    elif tool == "call_112":
        call_112(args["report"], interrupt)
    elif tool == "suggest_break":
        suggest_break(interrupt)
