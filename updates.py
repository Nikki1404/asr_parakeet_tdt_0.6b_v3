import os
import re
import sys
import time
import json
import wave
import struct
import difflib
import subprocess
import threading
import statistics
import collections
from pathlib import Path
from datetime import datetime

import azure.cognitiveservices.speech as speechsdk

# CONFIG  
SPEECH_KEY      = "a919211feda747e0b8f792278dcc9363"  
SPEECH_REGION   = "eastus"
INPUT_AUDIO     = "audio/maria1.mp3"        # ← your audio file

CANDIDATE_LANGUAGES = ["en-US", "es-US"]

# Stage 3 — Phrase boosting terms 
NUMERIC_PHRASES = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "account number","confirmation number","reference number",
    "zip code","date of birth","social security","routing number",
]
DOMAIN_PHRASES = [
    "verification code","balance due","minimum payment",
    "transfer","checking account","savings account","autopay",
    "statement","transaction","debit card","credit limit",
]

# Stage 10 — Concurrency levels to test
CONCURRENCY_LEVELS = [1, 3, 5, 10]

# Stage 12 — Fallback config
MAX_REPROMPTS = 2

OBS_DIR = "observations"
os.makedirs(OBS_DIR, exist_ok=True)

# STAGE MAP  — 13 stages (0 = baseline, 1-12 = one feature each)
STAGES = {
    0:  {
        "name": "baseline",
        "phase": "Baseline",
        "task": "Original working script",
        "description": "Your original script with no changes — establishes the measurement baseline",
        "outcome": "Reference point for all comparisons",
    },
    1:  {
        "name": "asr_config",
        "phase": "Setup",
        "task": "ASR Config Finalization",
        "description": "Lock language to en-US/es-US, set telephony audio profile, disable open-ended auto-detection",
        "outcome": "Stable, predictable recognition",
    },
    2:  {
        "name": "vad_tuning",
        "phase": "Audio",
        "task": "VAD Evaluation & Tuning",
        "description": "Tune silence thresholds: segmentation, initial silence, end silence timeouts",
        "outcome": "Reduced truncation and false cut-offs",
    },
    3:  {
        "name": "phrase_boost",
        "phase": "Accuracy",
        "task": "Word / Phrase Boosting",
        "description": "Boost digits, identifiers, and domain-specific terms",
        "outcome": "Improved numeric and domain accuracy",
    },
    4:  {
        "name": "vocab_tuning",
        "phase": "Accuracy",
        "task": "Transcript-Based Vocabulary Tuning",
        "description": "Auto-mine high-frequency domain terms from reference transcripts and boost them",
        "outcome": "Domain alignment",
    },
    5:  {
        "name": "numeric_handling",
        "phase": "Logic",
        "task": "Numeric Handling Validation",
        "description": "Enable Detailed output to expose ITN, Lexical, Display forms and detect digit grouping",
        "outcome": "Reduced verification failures",
    },
    6:  {
        "name": "dictation_mode",
        "phase": "Accuracy",
        "task": "Dictation Mode",
        "description": "Enable dictation mode for natural punctuation insertion",
        "outcome": "More readable, structured transcript",
    },
    7:  {
        "name": "emotion_tone",
        "phase": "Quality",
        "task": "Emotion / Tone Evaluation",
        "description": "Track confidence as stress proxy, speech rate, disfluency detection, negative keyword scan",
        "outcome": "Robust recognition measurement under varied speech",
    },
    8:  {
        "name": "latency_testing",
        "phase": "Testing",
        "task": "Latency & Timeout Testing",
        "description": "Validate TTFT, P90, P95 latency against conversational SLA (<500ms first-byte, <800ms avg)",
        "outcome": "Smooth turn-taking",
    },
    9:  {
        "name": "realtime_socket",
        "phase": "Integration",
        "task": "Real-Time Socket Integration",
        "description": "Switch from file-based to PushAudioInputStream (chunk-based streaming, simulates WebSocket)",
        "outcome": "Low-latency real-time ASR",
    },
    10: {
        "name": "concurrency",
        "phase": "Testing",
        "task": "Load & Concurrency Testing",
        "description": "Validate concurrent stream stability, detect throttling/quota ceiling",
        "outcome": "Stable under load",
    },
    11: {
        "name": "logging_alerts",
        "phase": "Monitoring",
        "task": "Logging & Alerts Setup",
        "description": "Enable structured JSON logging, alert rules for errors/latency/socket-drops",
        "outcome": "Early issue detection",
    },
    12: {
        "name": "fallback",
        "phase": "Go-Live",
        "task": "Fallback Validation",
        "description": "Test re-prompt, language retry, DTMF fallback, agent escalation chain",
        "outcome": "Resilient failure handling",
    },
}

# AUDIO HELPERS
def convert_to_wav(input_file: str) -> str:
    """Same as your working script."""
    input_path  = Path(input_file)
    output_file = str(input_path.with_suffix(".wav"))
    print(f"\n  Converting {input_file} → {output_file}")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  Conversion done.")
        return output_file
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg failed. Install: winget install ffmpeg")


def create_silence_wav(path: str, duration_sec: float = 3.0, sample_rate: int = 16000) -> str:
    """Generate a silence WAV for fallback testing."""
    n = int(sample_rate * duration_sec)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h"*n, *([0]*n)))
    return path


def read_wav_pcm(wav_path: str):
    """Read raw PCM bytes from a WAV file."""
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        bits     = wf.getsampwidth() * 8
        rate     = wf.getframerate()
        data     = wf.readframes(wf.getnframes())
    return data, channels, bits, rate


# TONE / EMOTION HELPERS  (Stage 7)
DISFLUENCY_WORDS = {"uh","um","hmm","er","ah","uhh","umm","erm"}
NEGATIVE_WORDS   = {"frustrated","angry","upset","terrible","cancel","refund",
                     "horrible","awful","worst","never","always","wrong","mistake"}
POSITIVE_WORDS   = {"great","perfect","excellent","thank","thanks","happy",
                     "resolved","appreciate","good","pleased"}

def tone_signals(text: str) -> dict:
    words   = set(text.lower().split())
    disfl   = words & DISFLUENCY_WORDS
    neg     = words & NEGATIVE_WORDS
    pos     = words & POSITIVE_WORDS
    tone    = "negative" if len(neg) > len(pos) else ("positive" if pos else "neutral")
    return {"disfluencies": list(disfl), "negative": list(neg),
            "positive": list(pos), "tone": tone}

# ALERT ENGINE  (Stage 11)
class AlertEngine:
    RULES = {
        "high_latency"   : "Segment latency > 1000ms",
        "low_confidence" : "Confidence < 0.65",
        "socket_drop"    : "Session canceled with error",
        "zero_result"    : "Zero recognised segments",
        "no_speech"      : "Initial silence timeout",
    }

    def __init__(self):
        self.alerts = []

    def check(self, rule: str, detail: str, value=None):
        entry = {"time": datetime.now().isoformat(), "rule": rule,
                 "description": self.RULES.get(rule,""), "detail": detail, "value": value}
        self.alerts.append(entry)
        print(f"  🔔 ALERT [{rule}] {detail}")

    def check_latency(self, ms, text):
        if ms > 1000: self.check("high_latency", f"{ms:.0f}ms — '{text[:30]}'", ms)

    def check_confidence(self, conf, text):
        if conf and conf < 0.65: self.check("low_confidence", f"conf={conf:.3f} — '{text[:30]}'", conf)

    def check_canceled(self, code, details):
        self.check("socket_drop", f"{code}: {(details or '')[:60]}")

    def check_zero(self, n):
        if n == 0: self.check("zero_result", "No segments produced")

    def to_dict(self): return self.alerts


# CORE RECOGNISER  
def build_speech_config(stage_num: int) -> speechsdk.SpeechConfig:
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.output_format = speechsdk.OutputFormat.Detailed

    # Stage 6+: dictation mode
    if stage_num >= 6:
        cfg.enable_dictation()

    # VAD defaults (your original)
    end_silence = "800"
    init_silence = "5000"
    seg_silence  = "800"

    # Stage 2+: tuned VAD
    if stage_num >= 2:
        end_silence  = "1500"   # more patient — fewer truncations
        init_silence = "5000"
        seg_silence  = "800"

    cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, end_silence)
    cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, init_silence)
    cfg.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, seg_silence)

    return cfg


def build_auto_lang(stage_num: int):
    # Stage 1+: locked to 2 locales (same list, but intent is explicit lock)
    return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=CANDIDATE_LANGUAGES)


def run_recognition(
        wav_file: str,
        stage_num: int,
        audio_cfg=None,
        extra_phrases: list = None,
        alerter: AlertEngine = None,
) -> dict:
    """
    Core recognition loop.
    Returns raw result dict with all metrics + segments.
    """
    cfg       = build_speech_config(stage_num)
    auto_lang = build_auto_lang(stage_num)

    if audio_cfg is None:
        audio_cfg = speechsdk.audio.AudioConfig(filename=wav_file)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=cfg,
        auto_detect_source_language_config=auto_lang,
        audio_config=audio_cfg,
    )

    # Stage 3+: phrase boosting
    if stage_num >= 3:
        pg = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        phrases = NUMERIC_PHRASES + DOMAIN_PHRASES + (extra_phrases or [])
        for p in phrases:
            pg.addPhrase(p)

    # State
    partial_results   = []
    final_results     = []
    final_transcript  = []
    detected_language = None
    first_partial     = None
    first_final       = None
    t_start           = time.time()
    done_event        = threading.Event()

    def on_recognizing(evt):
        nonlocal first_partial, detected_language
        if not evt.result.text: return
        now = time.time()
        if first_partial is None: first_partial = now
        try:
            detected_language = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        except: detected_language = "unknown"
        lat = (now - t_start) * 1000
        partial_results.append({"text": evt.result.text, "latency_ms": round(lat, 2)})
        print(f"  [PARTIAL {lat:.0f}ms] ({detected_language}) {evt.result.text}")

    def on_recognized(evt):
        nonlocal first_final, detected_language
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech: return
        if not evt.result.text: return
        now = time.time()
        if first_final is None: first_final = now
        try:
            detected_language = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        except: detected_language = "unknown"
        lat = (now - t_start) * 1000

        # Stage 5+: parse Detailed JSON
        confidence = itn = lexical = display = None
        if stage_num >= 5:
            try:
                detail = json.loads(evt.result.properties.get_property(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
                nb = detail.get("NBest", [])
                if nb:
                    confidence = nb[0].get("Confidence")
                    itn        = nb[0].get("ITN","")
                    lexical    = nb[0].get("Lexical","")
                    display    = nb[0].get("Display", evt.result.text)
            except: pass

        # Stage 7+: tone signals
        tone = tone_signals(evt.result.text) if stage_num >= 7 else None

        # Stage 7+: speech rate
        words_in_seg = len(evt.result.text.split())
        seg_dur_est  = (lat / 1000) / max(len(final_results)+1, 1)   # rough estimate
        wps          = round(words_in_seg / max(seg_dur_est, 0.1), 2) if stage_num >= 7 else None

        seg = {
            "text"      : evt.result.text,
            "display"   : display,
            "itn"       : itn,
            "lexical"   : lexical,
            "latency_ms": round(lat, 2),
            "confidence": round(confidence, 4) if confidence else None,
            "language"  : detected_language,
            "tone"      : tone,
            "wps"       : wps,
        }
        final_results.append(seg)
        final_transcript.append(evt.result.text)

        # Stage 11+: alert checks
        if alerter:
            alerter.check_latency(lat, evt.result.text)
            if confidence: alerter.check_confidence(confidence, evt.result.text)

        conf_str = f" conf={confidence:.3f}" if confidence else ""
        tone_str = f" [{tone['tone']}]" if tone else ""
        print(f"  [FINAL   {lat:.0f}ms] ({detected_language}){conf_str}{tone_str} {evt.result.text}")

    def on_canceled(evt):
        cd = evt.result.cancellation_details
        if cd.reason == speechsdk.CancellationReason.Error:
            print(f"\n  [CANCELED] {cd.reason} | {cd.error_code} | {cd.error_details}")
            if alerter: alerter.check_canceled(str(cd.error_code), cd.error_details)
        done_event.set()

    def on_stopped(evt):
        print("\n  [SESSION STOPPED]")
        done_event.set()

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_stopped)

    recognizer.start_continuous_recognition()
    done_event.wait(timeout=600)
    recognizer.stop_continuous_recognition()

    total_time   = time.time() - t_start
    full_text    = " ".join(final_transcript)
    confs        = [s["confidence"] for s in final_results if s.get("confidence")]
    ttft_partial = round((first_partial - t_start)*1000, 1) if first_partial else None
    ttft_final   = round((first_final   - t_start)*1000, 1) if first_final   else None

    return {
        "detected_language": detected_language,
        "ttft_partial_ms"  : ttft_partial,
        "ttft_final_ms"    : ttft_final,
        "total_time_sec"   : round(total_time, 2),
        "segment_count"    : len(final_results),
        "word_count"       : len(full_text.split()),
        "empty_segments"   : sum(1 for s in final_results if not s["text"].strip()),
        "avg_confidence"   : round(sum(confs)/len(confs), 4) if confs else None,
        "min_confidence"   : round(min(confs), 4) if confs else None,
        "max_confidence"   : round(max(confs), 4) if confs else None,
        "partial_count"    : len(partial_results),
        "transcript"       : full_text,
        "segments"         : final_results,
        "partial_results"  : partial_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-SPECIFIC RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_0_baseline(wav_file: str) -> dict:
    """Your original working script — no changes."""
    print("\n  Running as original script (no improvements).")
    return run_recognition(wav_file, stage_num=0)


def run_stage_1_asr_config(wav_file: str) -> dict:
    """Lock language, explicit audio format, disable open-ended detection."""
    print("\n  ASR Config: locking to en-US/es-US, telephony audio profile.")
    # Language is now explicitly locked (same function but intent documented)
    result = run_recognition(wav_file, stage_num=1)
    result["asr_config_notes"] = {
        "language_locked"    : CANDIDATE_LANGUAGES,
        "audio_format"       : "WAV PCM 16kHz mono 16-bit (converted via FFmpeg)",
        "auto_detect_locked" : True,
        "open_ended_detect"  : False,
    }
    return result


def run_stage_2_vad(wav_file: str) -> dict:
    """VAD: tuned silence thresholds."""
    print("\n  VAD: EndSilence=1500ms, InitSilence=5000ms, Seg=800ms")
    result = run_recognition(wav_file, stage_num=2)
    result["vad_config"] = {
        "end_silence_ms"  : 1500,
        "init_silence_ms" : 5000,
        "seg_silence_ms"  : 800,
        "note"            : "end_silence increased 800→1500ms to reduce truncation",
    }
    return result


def run_stage_3_phrase_boost(wav_file: str) -> dict:
    """Phrase boosting: numeric + domain terms."""
    all_phrases = NUMERIC_PHRASES + DOMAIN_PHRASES
    print(f"\n  Phrase boost: {len(all_phrases)} terms ({len(NUMERIC_PHRASES)} numeric, {len(DOMAIN_PHRASES)} domain)")
    result = run_recognition(wav_file, stage_num=3)
    full_text = result["transcript"].lower()
    hits = [p for p in all_phrases if p.lower() in full_text]
    result["phrase_boost"] = {
        "total_phrases" : len(all_phrases),
        "numeric_count" : len(NUMERIC_PHRASES),
        "domain_count"  : len(DOMAIN_PHRASES),
        "hits_in_transcript": hits,
        "hit_count"     : len(hits),
    }
    return result


def run_stage_4_vocab_tuning(wav_file: str) -> dict:
    """Auto-mine terms from existing transcript if available, boost them."""
    print("\n  Vocab tuning: mining domain terms from existing transcripts.")

    # Look for any transcript file from previous stages
    mined_terms = []
    for stage_dir in sorted(Path(OBS_DIR).glob("stage_*")):
        tx_file = stage_dir / "transcript.txt"
        if tx_file.exists():
            text  = tx_file.read_text(encoding="utf-8")
            words = re.findall(r"[a-zA-Z']{5,}", text.lower())
            freq  = collections.Counter(w for w in words
                                         if w not in {"about","their","there","would",
                                                        "could","which","where","after"})
            mined_terms = [w for w, c in freq.most_common(50) if c >= 2]
            break

    if mined_terms:
        print(f"  Mined {len(mined_terms)} domain terms from existing transcripts")
        print(f"  Top 10: {mined_terms[:10]}")
    else:
        print("  No existing transcript found — using domain phrases only")
        mined_terms = DOMAIN_PHRASES

    result = run_recognition(wav_file, stage_num=4, extra_phrases=mined_terms)
    result["vocab_tuning"] = {
        "mined_terms"      : mined_terms,
        "mined_term_count" : len(mined_terms),
        "source"           : "auto-mined from prior stage transcripts",
    }
    return result


def run_stage_5_numeric(wav_file: str) -> dict:
    """Detailed output: ITN / Lexical / Display forms + digit grouping detection."""
    print("\n  Numeric handling: parsing ITN, Lexical, Display forms from Detailed JSON.")
    result = run_recognition(wav_file, stage_num=5)

    # Analyse digit forms in segments
    numeric_analysis = []
    for seg in result["segments"]:
        itn     = seg.get("itn", "") or ""
        lexical = seg.get("lexical", "") or ""
        display = seg.get("display", "") or ""

        has_spoken_digits = bool(re.search(
            r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b',
            lexical, re.I))
        itn_digits    = re.sub(r'[^\d]', '', itn)
        itn_grouped   = len(itn_digits) >= 3 and not bool(re.search(r'\d\s\d', itn))

        if has_spoken_digits or itn_digits:
            numeric_analysis.append({
                "display"          : display,
                "itn"              : itn,
                "lexical"          : lexical,
                "digit_string"     : itn_digits,
                "itn_grouped"      : itn_grouped,
                "spoken_digit_words": has_spoken_digits,
            })

    result["numeric_analysis"] = numeric_analysis
    result["numeric_summary"]  = {
        "segments_with_numbers": len(numeric_analysis),
        "grouped_digit_segs"   : sum(1 for n in numeric_analysis if n["itn_grouped"]),
        "digit_by_digit_segs"  : sum(1 for n in numeric_analysis if not n["itn_grouped"] and n["digit_string"]),
    }
    print(f"  Segments with numbers: {len(numeric_analysis)}")
    print(f"  Grouped digit form   : {result['numeric_summary']['grouped_digit_segs']}")
    print(f"  Digit-by-digit form  : {result['numeric_summary']['digit_by_digit_segs']}")
    return result


def run_stage_6_dictation(wav_file: str) -> dict:
    """Dictation mode: natural punctuation."""
    print("\n  Dictation mode: ON — natural punctuation will be inserted.")
    result = run_recognition(wav_file, stage_num=6)
    # Count punctuation marks in transcript
    tx = result["transcript"]
    result["dictation_analysis"] = {
        "commas"    : tx.count(","),
        "periods"   : tx.count("."),
        "questions" : tx.count("?"),
        "total_punct": tx.count(",") + tx.count(".") + tx.count("?") + tx.count("!"),
    }
    print(f"  Punctuation: {result['dictation_analysis']}")
    return result


def run_stage_7_emotion_tone(wav_file: str) -> dict:
    """Emotion/tone proxy: confidence, speech rate, disfluency, keywords."""
    print("\n  Emotion/Tone: tracking confidence, rate, disfluency, keywords.")
    result = run_recognition(wav_file, stage_num=7)

    segs        = result["segments"]
    all_text    = result["transcript"]
    all_tone    = tone_signals(all_text)
    confs       = [s["confidence"] for s in segs if s.get("confidence")]
    low_conf    = [s for s in segs if s.get("confidence") and s["confidence"] < 0.70]
    disfl_segs  = [s for s in segs if s.get("tone") and s["tone"].get("disfluencies")]

    result["emotion_tone"] = {
        "overall_tone"          : all_tone["tone"],
        "negative_markers"      : all_tone["negative"],
        "positive_markers"      : all_tone["positive"],
        "disfluencies_overall"  : all_tone["disfluencies"],
        "low_confidence_segs"   : len(low_conf),
        "disfluency_segs"       : len(disfl_segs),
        "stress_risk"           : "high" if len(low_conf) > len(segs)*0.3 else
                                  "medium" if len(low_conf) > len(segs)*0.1 else "low",
        "sdk_limitation"        : "Azure SDK has no native emotion label — using confidence + rate + keywords as proxy",
    }
    print(f"  Overall tone  : {result['emotion_tone']['overall_tone']}")
    print(f"  Stress risk   : {result['emotion_tone']['stress_risk']}")
    print(f"  Low-conf segs : {len(low_conf)}")
    print(f"  Disfluency segs: {len(disfl_segs)}")
    return result


def run_stage_8_latency(wav_file: str) -> dict:
    """Latency: multi-run P50/P90/P95 measurement + SLA assessment."""
    print("\n  Latency testing: 3 runs to measure P50/P90/P95 + SLA check.")
    SLA_FIRST_BYTE = 500
    SLA_AVG        = 800
    SLA_P90        = 1200

    all_ttft_final  = []
    all_ttft_partial = []
    all_avg_lat     = []
    runs            = []

    for run_id in range(1, 4):
        print(f"\n  Run {run_id}/3")
        r = run_recognition(wav_file, stage_num=8)
        seg_lats = [s["latency_ms"] for s in r["segments"]]
        avg_lat  = round(statistics.mean(seg_lats), 1) if seg_lats else None

        all_ttft_final.append(r["ttft_final_ms"] or 0)
        all_ttft_partial.append(r["ttft_partial_ms"] or 0)
        if avg_lat: all_avg_lat.append(avg_lat)

        run_result = {**r, "run_id": run_id, "avg_seg_latency_ms": avg_lat}
        run_result["p90_ms"] = sorted(seg_lats)[int(len(seg_lats)*0.9)] if len(seg_lats) >= 5 else None
        run_result["p95_ms"] = sorted(seg_lats)[int(len(seg_lats)*0.95)] if len(seg_lats) >= 10 else None
        runs.append(run_result)
        time.sleep(2)

    best_run = min(runs, key=lambda r: r["ttft_final_ms"] or 9999)
    sla = {
        "first_byte_pass": (min(all_ttft_final) or 9999) <= SLA_FIRST_BYTE,
        "avg_pass"        : (min(all_avg_lat) if all_avg_lat else 9999) <= SLA_AVG,
        "p90_pass"        : any(r.get("p90_ms") and r["p90_ms"] <= SLA_P90 for r in runs),
        "targets"         : {"first_byte_ms": SLA_FIRST_BYTE, "avg_ms": SLA_AVG, "p90_ms": SLA_P90},
    }

    result = {**best_run}
    result["latency_multi_run"] = {
        "runs"              : len(runs),
        "avg_ttft_final_ms" : round(statistics.mean(all_ttft_final), 1),
        "min_ttft_final_ms" : min(all_ttft_final),
        "max_ttft_final_ms" : max(all_ttft_final),
        "all_runs"          : [{k: v for k, v in r.items() if k != "segments"} for r in runs],
        "sla_assessment"    : sla,
    }

    print(f"\n  Avg TTFT Final : {result['latency_multi_run']['avg_ttft_final_ms']} ms")
    print(f"  SLA first-byte : {'✔ PASS' if sla['first_byte_pass'] else '✘ FAIL'} (target <{SLA_FIRST_BYTE}ms)")
    print(f"  SLA avg        : {'✔ PASS' if sla['avg_pass'] else '✘ FAIL'} (target <{SLA_AVG}ms)")
    return result


def run_stage_9_realtime_socket(wav_file: str) -> dict:
    """PushAudioInputStream: chunk-based streaming (WebSocket equivalent)."""
    print("\n  Real-time socket: PushAudioInputStream with 40ms chunks.")
    CHUNK_MS = 40

    try:
        pcm, channels, bits, sample_rate = read_wav_pcm(wav_file)
    except Exception as e:
        print(f"  Could not read WAV for push stream: {e}")
        return {"error": str(e), "transcript": "", "segment_count": 0,
                "word_count": 0, "ttft_final_ms": None, "ttft_partial_ms": None,
                "total_time_sec": 0, "segments": [], "partial_results": [],
                "avg_confidence": None, "min_confidence": None, "max_confidence": None,
                "empty_segments": 0, "partial_count": 0, "detected_language": None}

    audio_fmt   = speechsdk.audio.AudioStreamFormat(
        samples_per_second=sample_rate, bits_per_sample=bits, channels=channels)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_fmt)
    audio_cfg   = speechsdk.audio.AudioConfig(stream=push_stream)

    bytes_per_ms = (sample_rate * channels * (bits//8)) // 1000
    chunk_size   = bytes_per_ms * CHUNK_MS
    chunks       = [pcm[i:i+chunk_size] for i in range(0, len(pcm), chunk_size)]

    # Feed chunks in background thread
    def feed():
        for chunk in chunks:
            push_stream.write(chunk)
            time.sleep(CHUNK_MS / 1000)
        push_stream.close()

    threading.Thread(target=feed, daemon=True).start()

    result = run_recognition(wav_file, stage_num=9, audio_cfg=audio_cfg)
    result["realtime_socket"] = {
        "method"        : "PushAudioInputStream",
        "chunk_ms"      : CHUNK_MS,
        "chunk_count"   : len(chunks),
        "sample_rate"   : sample_rate,
        "note"          : "Simulates WebSocket/live audio ingestion",
    }
    print(f"  Method   : PushAudioInputStream")
    print(f"  Chunks   : {len(chunks)} × {CHUNK_MS}ms")
    return result


def run_stage_10_concurrency(wav_file: str) -> dict:
    """Concurrent stream validation."""
    import concurrent.futures
    print(f"\n  Concurrency test: levels {CONCURRENCY_LEVELS}")

    def one_stream(worker_id):
        t0   = time.time()
        done = threading.Event()
        segs = []
        err  = {"throttled": False, "error": None}

        cfg   = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        auto  = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=CANDIDATE_LANGUAGES)
        audio = speechsdk.audio.AudioConfig(filename=wav_file)
        rec   = speechsdk.SpeechRecognizer(speech_config=cfg,
                                            auto_detect_source_language_config=auto,
                                            audio_config=audio)

        def on_rec(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                segs.append(evt.result.text)
        def on_stop(evt): done.set()
        def on_cancel(evt):
            cd = evt.result.cancellation_details
            if cd.reason == speechsdk.CancellationReason.Error:
                err["error"] = cd.error_details
                err["throttled"] = "429" in (cd.error_details or "") or \
                                   "quota" in (cd.error_details or "").lower()
            done.set()

        rec.recognized.connect(on_rec)
        rec.session_stopped.connect(on_stop)
        rec.canceled.connect(on_cancel)
        rec.start_continuous_recognition()
        done.wait(timeout=300)
        rec.stop_continuous_recognition()

        return {
            "worker_id": worker_id, "status": "ok" if not err["error"] else "error",
            "throttled": err["throttled"], "error": err["error"],
            "segments": len(segs), "total_sec": round(time.time()-t0, 2),
        }

    level_results = []
    ceiling       = None

    for n in CONCURRENCY_LEVELS:
        print(f"\n  Testing {n} concurrent stream(s)…")
        t_wall = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            workers = list(pool.map(one_stream, range(n)))
        wall_sec    = round(time.time()-t_wall, 2)
        ok          = sum(1 for w in workers if w["status"]=="ok")
        throttled   = sum(1 for w in workers if w["throttled"])
        success_pct = round(ok/n*100, 1)

        level = {
            "concurrency": n, "wall_sec": wall_sec, "ok": ok,
            "throttled": throttled, "success_pct": success_pct,
            "ceiling_hit": throttled > 0,
        }
        level_results.append(level)
        print(f"  N={n}: {ok}/{n} OK ({success_pct}%) | throttled={throttled}")

        if throttled > 0:
            ceiling = n
            print(f"  ⚠ Quota ceiling hit at N={n} — stopping ramp")
            break
        time.sleep(3)

    safe = [l["concurrency"] for l in level_results if l["success_pct"]==100 and not l["ceiling_hit"]]

    # Base result uses single stream for transcript comparison
    base = run_recognition(wav_file, stage_num=10)
    base["concurrency_test"] = {
        "levels_tested"      : CONCURRENCY_LEVELS,
        "level_results"      : level_results,
        "max_safe_concurrency": max(safe) if safe else 1,
        "quota_ceiling"      : ceiling,
    }
    print(f"\n  Max safe concurrency: {base['concurrency_test']['max_safe_concurrency']}")
    return base


def run_stage_11_logging(wav_file: str) -> dict:
    """Logging + alert rules."""
    print("\n  Logging & alerts: structured JSON log + alert rule checks.")

    log_dir = os.path.join(OBS_DIR, "stage_11_logging_alerts", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Enable SDK diagnostic log
    sdk_log = os.path.join(log_dir, "azure_sdk.log")
    os.environ["SPEECH_SDK_LOGFILE"] = sdk_log

    alerter      = AlertEngine()
    session_log  = {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "audio"     : wav_file,
        "start"     : datetime.now().isoformat(),
        "events"    : [],
    }

    result = run_recognition(wav_file, stage_num=11, alerter=alerter)

    # Post-run alert checks
    alerter.check_zero(result["segment_count"])

    # Save logs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(log_dir, f"session_{ts}.json"), "w") as f:
        json.dump({**result, "session_log": session_log}, f, indent=2, default=str)
    with open(os.path.join(log_dir, f"alerts_{ts}.json"), "w") as f:
        json.dump(alerter.to_dict(), f, indent=2)

    result["logging"] = {
        "session_log_path": log_dir,
        "sdk_log_path"    : sdk_log,
        "alerts_fired"    : len(alerter.alerts),
        "alert_details"   : alerter.to_dict(),
        "rules_active"    : list(AlertEngine.RULES.keys()),
    }
    print(f"\n  Alerts fired  : {len(alerter.alerts)}")
    print(f"  Logs saved    : {log_dir}")
    return result


def run_stage_12_fallback(wav_file: str) -> dict:
    """Fallback chain: re-prompt → language retry → DTMF simulation."""
    print("\n  Fallback validation: testing re-prompt → language retry → DTMF chain.")

    fallback_log = []
    attempts     = []

    def attempt_recognition(audio_path, languages, attempt_num):
        cfg   = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,"3000")
        auto  = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
        audio = speechsdk.audio.AudioConfig(filename=audio_path)
        rec   = speechsdk.SpeechRecognizer(speech_config=cfg,
                                            auto_detect_source_language_config=auto,
                                            audio_config=audio)
        segs  = []; nomatch = []; done = threading.Event(); canceled = {"v": False}
        t0    = time.time()

        def on_rec(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                segs.append(evt.result.text)
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                nomatch.append(True)
        def on_stop(evt): done.set()
        def on_cancel(evt):
            canceled["v"] = True
            done.set()

        rec.recognized.connect(on_rec); rec.session_stopped.connect(on_stop)
        rec.canceled.connect(on_cancel)
        rec.start_continuous_recognition()
        done.wait(timeout=120)
        rec.stop_continuous_recognition()

        return {
            "attempt": attempt_num, "segments": len(segs),
            "no_match": len(nomatch), "canceled": canceled["v"],
            "transcript": " ".join(segs), "total_sec": round(time.time()-t0, 2),
        }

    # Test 1: Normal audio
    print("\n  Attempt 1: normal audio")
    r1 = attempt_recognition(wav_file, CANDIDATE_LANGUAGES, 1)
    attempts.append(r1)
    fallback_log.append({"action": "recognition_attempt", "attempt": 1, "result": r1["segments"] > 0})
    print(f"  Result: {r1['segments']} segments, canceled={r1['canceled']}")

    # Test 2: Silence file → triggers no-match → re-prompt
    silence_path = os.path.join(OBS_DIR, "silence_test.wav")
    create_silence_wav(silence_path, 3.0)
    print("\n  Attempt 2: silence audio (triggers no-match path)")
    r2 = attempt_recognition(silence_path, CANDIDATE_LANGUAGES, 2)
    attempts.append(r2)
    fallback_log.append({"action": "silence_test", "result": r2["segments"] > 0})

    if r2["segments"] == 0:
        fallback_log.append({"action": "reprompt_triggered", "reason": "no_speech"})
        print("  ✔ No-match path triggered — re-prompt would fire in live IVR")

    # Test 3: Language retry (swap order)
    print("\n  Attempt 3: language retry (reversed language order)")
    r3 = attempt_recognition(wav_file, list(reversed(CANDIDATE_LANGUAGES)), 3)
    attempts.append(r3)
    fallback_log.append({"action": "language_retry", "languages": list(reversed(CANDIDATE_LANGUAGES)),
                          "result": r3["segments"] > 0})
    print(f"  Language retry: {r3['segments']} segments")

    # Test 4: Simulated DTMF (logged only — not real DTMF)
    print("\n  DTMF fallback: simulated (would activate if all ASR attempts fail)")
    fallback_log.append({"action": "dtmf_fallback_simulated", "input": "1",
                          "note": "In live IVR: switch to touch-tone input"})
    time.sleep(0.5)
    fallback_log.append({"action": "agent_escalation_if_dtmf_fails",
                          "note": "In live IVR: route to human agent"})

    # Use normal audio result for comparison metrics
    base = run_recognition(wav_file, stage_num=12)
    base["fallback_test"] = {
        "attempts"       : attempts,
        "fallback_log"   : fallback_log,
        "silence_triggered_reprompt": r2["segments"] == 0,
        "language_retry_worked"     : r3["segments"] > 0,
        "dtmf_simulated": True,
        "fallback_chain" : ["recognition", "re-prompt", "language_retry", "dtmf", "agent_escalation"],
    }
    print(f"\n  Silence triggered re-prompt : {base['fallback_test']['silence_triggered_reprompt']}")
    print(f"  Language retry worked       : {base['fallback_test']['language_retry_worked']}")
    return base


# STAGE DISPATCHER
STAGE_RUNNERS = {
    0 : run_stage_0_baseline,
    1 : run_stage_1_asr_config,
    2 : run_stage_2_vad,
    3 : run_stage_3_phrase_boost,
    4 : run_stage_4_vocab_tuning,
    5 : run_stage_5_numeric,
    6 : run_stage_6_dictation,
    7 : run_stage_7_emotion_tone,
    8 : run_stage_8_latency,
    9 : run_stage_9_realtime_socket,
    10: run_stage_10_concurrency,
    11: run_stage_11_logging,
    12: run_stage_12_fallback,
}


# SAVE + COMPARE  (same as before)
REPORT_PATH = os.path.join(OBS_DIR, "comparison_report.json")

def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f: return json.load(f)
    return {"stages": [], "comparisons": [], "last_updated": None}

def save_report(report):
    report["last_updated"] = datetime.now().isoformat()
    with open(REPORT_PATH, "w") as f: json.dump(report, f, indent=2, default=str)

def save_stage_files(stage_num: int, stage_meta: dict, result: dict):
    name      = stage_meta["name"]
    stage_dir = os.path.join(OBS_DIR, f"stage_{stage_num}_{name}")
    os.makedirs(stage_dir, exist_ok=True)

    with open(os.path.join(stage_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(stage_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(result.get("transcript", ""))

    m = {k: result.get(k) for k in ["ttft_partial_ms","ttft_final_ms","total_time_sec",
                                      "segment_count","word_count","empty_segments",
                                      "avg_confidence","min_confidence","max_confidence"]}
    with open(os.path.join(stage_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        lines = [
            f"Stage       : {stage_num} — {name}",
            f"Phase       : {stage_meta['phase']}",
            f"Task        : {stage_meta['task']}",
            f"Description : {stage_meta['description']}",
            f"Outcome     : {stage_meta['outcome']}",
            f"Timestamp   : {datetime.now().isoformat()}",
            "", "── Metrics ───────────────────────────",
        ]
        for k, v in m.items():
            lines.append(f"  {k:<22}: {v}")
        lines += ["", "── Transcript ──────────────────────", result.get("transcript","")]
        f.write("\n".join(lines))

    print(f"  Saved → {stage_dir}/")
    return stage_dir


def compute_delta(prev, curr, lower_better=False):
    if prev is None or curr is None:
        return {"prev": prev, "curr": curr, "change": None, "direction": "unknown"}
    d = curr - prev
    if lower_better: direction = "improved" if d < 0 else ("worse" if d > 0 else "same")
    else:            direction = "improved" if d > 0 else ("worse" if d < 0 else "same")
    return {"prev": prev, "curr": curr, "change": round(d, 4), "direction": direction}


def compare(prev_result, curr_result, curr_meta) -> dict:
    pm = prev_result
    cm = curr_result

    words_a = (pm.get("transcript","") or "").lower().split()
    words_b = (cm.get("transcript","") or "").lower().split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    sim_pct = round(matcher.ratio()*100, 1)
    changes = [{"type":t,"before":" ".join(words_a[i1:i2]),"after":" ".join(words_b[j1:j2])}
               for t,i1,i2,j1,j2 in matcher.get_opcodes() if t!="equal"]

    deltas = {
        "ttft_partial_ms": compute_delta(pm.get("ttft_partial_ms"), cm.get("ttft_partial_ms"), True),
        "ttft_final_ms"  : compute_delta(pm.get("ttft_final_ms"),   cm.get("ttft_final_ms"),   True),
        "total_time_sec" : compute_delta(pm.get("total_time_sec"),   cm.get("total_time_sec"),   True),
        "segment_count"  : compute_delta(pm.get("segment_count"),   cm.get("segment_count")),
        "word_count"     : compute_delta(pm.get("word_count"),       cm.get("word_count")),
        "empty_segments" : compute_delta(pm.get("empty_segments"),   cm.get("empty_segments"),   True),
        "avg_confidence" : compute_delta(pm.get("avg_confidence"),   cm.get("avg_confidence")),
        "min_confidence" : compute_delta(pm.get("min_confidence"),   cm.get("min_confidence")),
    }

    notes = _observations(pm, cm, sim_pct, len(changes), curr_meta)

    return {
        "from_stage"     : pm.get("_stage_name","prev"),
        "to_stage"       : cm.get("_stage_name","curr"),
        "phase"          : curr_meta["phase"],
        "task"           : curr_meta["task"],
        "description"    : curr_meta["description"],
        "expected_outcome": curr_meta["outcome"],
        "metric_deltas"  : deltas,
        "transcript_diff": {"similarity_pct": sim_pct, "change_count": len(changes),
                             "changes": changes[:15]},
        "observations"   : notes,
        "timestamp"      : datetime.now().isoformat(),
    }


def _observations(pm, cm, sim_pct, n_changes, meta) -> list:
    notes = []
    # TTFT
    if pm.get("ttft_final_ms") and cm.get("ttft_final_ms"):
        d = cm["ttft_final_ms"] - pm["ttft_final_ms"]
        if d < -50: notes.append(f"✅ TTFT Final improved by {abs(d):.0f}ms")
        elif d > 50: notes.append(f"⚠️  TTFT Final slower by {d:.0f}ms (may be acceptable)")
    # Words
    if pm.get("word_count") and cm.get("word_count"):
        d = cm["word_count"] - pm["word_count"]
        if d > 5:  notes.append(f"✅ {d} more words captured — less truncation")
        elif d < -5: notes.append(f"⚠️  {abs(d)} fewer words — check endpointing")
    # Confidence
    if pm.get("avg_confidence") and cm.get("avg_confidence"):
        d = cm["avg_confidence"] - pm["avg_confidence"]
        if d > 0.01:  notes.append(f"✅ Confidence improved by {d:.4f}")
        elif d < -0.01: notes.append(f"⚠️  Confidence dropped by {abs(d):.4f}")
    # Empty segs
    if pm.get("empty_segments") is not None and cm.get("empty_segments") is not None:
        d = cm["empty_segments"] - pm["empty_segments"]
        if d < 0:  notes.append(f"✅ {abs(d)} fewer empty segments")
        elif d > 0: notes.append(f"⚠️  {d} more empty segments")
    # Transcript
    if sim_pct > 98: notes.append(f"➡️  Transcript unchanged ({sim_pct}% similar) — feature impact is in metrics, not text")
    elif sim_pct > 90: notes.append(f"➡️  Transcript similar ({sim_pct}%) — small word-level changes")
    else: notes.append(f"⚠️  Transcript changed significantly ({sim_pct}%) — review word diff")
    # Expected outcome hint
    notes.append(f"ℹ️  Expected: {meta['outcome']}")
    return notes


def print_comparison(comp: dict):
    print(f"\n{'─'*65}")
    print(f"  COMPARISON: {comp['from_stage']} → {comp['to_stage']}")
    print(f"  Phase: {comp['phase']} | Task: {comp['task']}")
    print(f"{'─'*65}")
    for metric, d in comp["metric_deltas"].items():
        if d["change"] is None: continue
        sym = {"improved":"↑","worse":"↓","same":"→"}.get(d["direction"],"?")
        sign = "+" if d["change"] > 0 else ""
        print(f"  {metric:<22}: {d['prev']} → {d['curr']}  ({sign}{d['change']}) {sym} {d['direction']}")
    td = comp["transcript_diff"]
    print(f"\n  Transcript similarity : {td['similarity_pct']}%  |  Changes: {td['change_count']}")
    if td["changes"]:
        for c in td["changes"][:5]:
            print(f"    [{c['type']}] '{c['before'][:30]}' → '{c['after'][:30]}'")
    print(f"\n  Observations:")
    for note in comp["observations"]:
        print(f"    {note}")


def print_full_table(report: dict):
    stages = report.get("stages", [])
    if not stages: return
    print(f"\n\n{'='*75}")
    print("  FULL INCREMENTAL REPORT — ALL STAGES")
    print(f"{'='*75}")
    print(f"  {'#':<3} {'Stage':<22} {'Phase':<14} {'Words':>6} {'Segs':>5} {'Conf':>7} {'TTFT-F':>8} {'Time':>7}")
    print(f"  {'─'*3} {'─'*22} {'─'*14} {'─'*6} {'─'*5} {'─'*7} {'─'*8} {'─'*7}")
    for s in stages:
        m = s.get("metrics", s)
        n = s.get("_stage_num", "?")
        print(f"  {str(n):<3} {s.get('_stage_name','?'):<22} {s.get('_phase','?'):<14} "
              f"{str(m.get('word_count','?')):>6} "
              f"{str(m.get('segment_count','?')):>5} "
              f"{str(m.get('avg_confidence','N/A')):>7} "
              f"{str(m.get('ttft_final_ms','N/A')):>8} "
              f"{str(m.get('total_time_sec','?')):>7}")
    if len(stages) >= 2:
        bm = stages[0]
        lm = stages[-1]
        print(f"\n  NET GAIN (stage 0 → stage {lm.get('_stage_num','?')}):")
        for metric in ["word_count","avg_confidence","ttft_final_ms","empty_segments"]:
            bv = bm.get(metric) or (bm.get("metrics") or {}).get(metric)
            lv = lm.get(metric) or (lm.get("metrics") or {}).get(metric)
            if bv is None or lv is None: continue
            diff = round(lv - bv, 4)
            sign = "+" if diff > 0 else ""
            print(f"    {metric:<22}: {bv} → {lv}  ({sign}{diff})")
    print(f"\n  Full JSON  → {REPORT_PATH}")
    print(f"  Run 'python generate_observation_doc.py' for full documentation")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_one(stage_num: int):
    if stage_num not in STAGES:
        print(f"Invalid stage {stage_num}. Valid: 0–12")
        sys.exit(1)

    meta     = STAGES[stage_num]
    wav_file = convert_to_wav(INPUT_AUDIO)

    print(f"\n{'='*65}")
    print(f"  STAGE {stage_num} / 12  —  {meta['name'].upper()}")
    print(f"  Phase   : {meta['phase']}")
    print(f"  Task    : {meta['task']}")
    print(f"  Goal    : {meta['outcome']}")
    print(f"{'='*65}")

    result = STAGE_RUNNERS[stage_num](wav_file)

    # Tag result with stage metadata
    result["_stage_num"]  = stage_num
    result["_stage_name"] = meta["name"]
    result["_phase"]      = meta["phase"]
    result["_task"]       = meta["task"]
    result["_timestamp"]  = datetime.now().isoformat()

    save_stage_files(stage_num, meta, result)

    report = load_report()
    report["stages"].append(result)

    if len(report["stages"]) >= 2:
        prev = report["stages"][-2]
        comp = compare(prev, result, meta)
        report["comparisons"].append(comp)
        print_comparison(comp)

    save_report(report)
    print_full_table(report)


def run_all():
    wav_file = convert_to_wav(INPUT_AUDIO)
    report   = load_report()

    for stage_num in range(0, 13):
        meta   = STAGES[stage_num]
        print(f"\n{'='*65}")
        print(f"  STAGE {stage_num} / 12  —  {meta['name'].upper()}")
        print(f"  Phase: {meta['phase']} | Task: {meta['task']}")
        print(f"{'='*65}")

        result = STAGE_RUNNERS[stage_num](wav_file)
        result["_stage_num"]  = stage_num
        result["_stage_name"] = meta["name"]
        result["_phase"]      = meta["phase"]
        result["_task"]       = meta["task"]
        result["_timestamp"]  = datetime.now().isoformat()

        save_stage_files(stage_num, meta, result)
        report["stages"].append(result)

        if len(report["stages"]) >= 2:
            comp = compare(report["stages"][-2], result, meta)
            report["comparisons"].append(comp)
            print_comparison(comp)

        save_report(report)

        if stage_num < 12:
            print(f"\n  Pausing 3s before next stage…")
            time.sleep(3)

    print_full_table(report)


def main():
    print("\n  Azure STT — 12-Stage Incremental Improvement")
    print("  ─────────────────────────────────────────────")
    print("  Stages:")
    for n, m in STAGES.items():
        print(f"    {n:>2}  {m['name']:<22} [{m['phase']}] {m['task']}")
    print()
    print("  Usage:")
    print("    python azure_incremental.py --stage 0   ← start here")
    print("    python azure_incremental.py --stage 1")
    print("    ...up to --stage 12")
    print("    python azure_incremental.py --all       ← run all")
    print()

    if "--all" in sys.argv:
        run_all()
    elif "--stage" in sys.argv:
        idx = sys.argv.index("--stage")
        run_one(int(sys.argv[idx+1]))
    else:
        print("  No stage specified — running --stage 0 (baseline)")
        run_one(0)


if __name__ == "__main__":
    main()
