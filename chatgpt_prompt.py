#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        AZURE STT TRANSCRIPTION QUALITY LAB  —  ALL 12 STAGES       ║
╚══════════════════════════════════════════════════════════════════════╝

Maps exactly to the requirements table:

  Stage 0  : Baseline                    — Original unmodified script
  Stage 1  : ASR Config Finalization     — Lock lang/locale, audio format, disable auto-detect
  Stage 2  : Concurrency & Quota         — Validate rate-limits, concurrent sessions
  Stage 3  : Real-Time Socket            — PushAudioStream (WebSocket-style streaming)
  Stage 4  : VAD Evaluation & Tuning     — Three sub-stages (default / conservative / aggressive)
  Stage 5  : Word / Phrase Boosting      — PhraseListGrammar for digits, IDs, domain terms
  Stage 6  : Transcript Vocabulary       — Boost words extracted from baseline transcript
  Stage 7  : Numeric Handling            — digit-by-digit vs grouped; context-aware PP
  Stage 8  : Emotion / Tone Evaluation   — Confidence scores, NBest, stressed-speech resilience
  Stage 9  : Latency & Timeout Testing   — TTFT, P50/P95 latency, SLA thresholds
  Stage 10 : Load & Concurrency          — Peak concurrent streams, success rate under load
  Stage 11 : Logging & Alerts            — Structured JSON logs, alert thresholds
  Stage 12 : Fallback Validation         — Empty/silence handling, low-confidence re-prompt
  Stage C1 : Combined Best               — PRODUCTION RECOMMENDATION
  Stage C2 : Combined All                — Everything together

Outputs:
  transcription_report.md      Full per-stage metrics comparison table
  transcription_doc_guide.md   Human-readable documentation guide (how-to, param refs, observations)
  results.json                 Machine-readable metrics
  transcription_audit.log      Structured JSON log (Stage 11+)
  Console                      Live partials, finals, per-stage summaries

Usage:
  python transcription_stages_lab.py                        # all stages
  python transcription_stages_lab.py --stage stage_5        # single stage
  python transcription_stages_lab.py --file my_audio.mp3    # custom audio
  python transcription_stages_lab.py --concurrency 5        # override concurrency count
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

# ══════════════════════════════════════════════════════════════════════
# ██  CONFIG  —  edit these before running
# ══════════════════════════════════════════════════════════════════════

SPEECH_KEY    = "YOUR_AZURE_SPEECH_KEY"
SPEECH_REGION = "eastus"

CANDIDATE_LANGUAGES = ["en-US", "es-ES"]
INPUT_AUDIO_FILE    = "audio/maria1.mp3"

# SLA threshold for Latency stage (ms) — conversations should be < this
LATENCY_SLA_MS = 800

# Concurrency test: how many simultaneous sessions to test
CONCURRENCY_LEVELS = [1, 3, 5]

# Confidence threshold below which a segment is flagged for re-prompt (Stage 12)
CONFIDENCE_REPROMPT_THRESHOLD = 0.6

# Domain phrases for Stage 5 — customize for your call domain
DOMAIN_PHRASES = [
    # Digit sequences
    "zero one two three four five six seven eight nine",
    # Identifiers
    "account number", "reference number", "customer ID", "transaction ID",
    "verification code", "PIN number", "ZIP code", "date of birth",
    "social security number", "credit card number", "extension number",
    # Short / easily-dropped words
    "ID", "OK", "yes", "no", "yeah", "nope",
    # IVR menu patterns
    "press one", "press two", "press three", "press zero",
    "option one", "option two", "option three",
    # Timing / numeric phrases
    "one moment", "hold on", "just a second",
]

# ══════════════════════════════════════════════════════════════════════
# ██  NUMERIC POST-PROCESSOR
# ══════════════════════════════════════════════════════════════════════

_NEVER_CONVERT = frozenset({
    # Prepositions / articles — NEVER convert (user's explicit requirement)
    "to", "too",        # "to" must NEVER become "2"
    "for", "fore",      # "for" must NEVER become "4"
    "a",  "an",
    "be", "bee",
    "by", "buy", "bye",
    "no", "know",
    "or", "ore",
    "so", "sew",
    "in", "inn",
    "ate",              # past tense of eat (sounds like 8)
    "won",              # past tense of win (sounds like one)
})

_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0",
    "one":  "1",
    # "two" handled conservatively (homophones: to/too)
    "two":  "2",        # only converted in strong numeric context
    "three":"3", "four":"4", "five":"5",
    "six":  "6", "seven":"7", "eight":"8", "nine":"9",
    "ten":  "10", "eleven":"11", "twelve":"12",
    "thirteen":"13", "fourteen":"14", "fifteen":"15",
    "sixteen":"16", "seventeen":"17", "eighteen":"18", "nineteen":"19",
    "twenty":"20", "thirty":"30", "forty":"40", "fifty":"50",
    "sixty":"60", "seventy":"70", "eighty":"80", "ninety":"90",
}

_CONTEXT_BEFORE = frozenset({
    "number","#","digit","code","id","account","reference",
    "extension","ext","pin","zip","chapter","version",
    "track","room","floor","gate","seat","row","step",
    "item","option","press","dial","minus","plus","times",
})
_CONTEXT_AFTER = frozenset({
    "people","person","items","things","times","calls",
    "days","hours","minutes","seconds","weeks","months","years",
    "dollars","cents","percent","%","meters","feet","pounds",
    "tickets","orders","attempts",
})


def numeric_postprocess(text: str, language: str = "en-US") -> str:
    """
    Context-aware word→digit conversion.
    Safety rules:
      1. Spanish → pass-through (no conversion)
      2. _NEVER_CONVERT words → always kept as-is
      3. Conversion only when adjacent context is clearly numeric
    """
    if language and language.lower().startswith("es"):
        return text  # Spanish: untouched

    tokens = text.split()
    result = []
    for i, tok in enumerate(tokens):
        alpha = re.sub(r"[^a-zA-Z]", "", tok).lower()
        suffix = re.sub(r"[a-zA-Z]", "", tok)

        if alpha in _NEVER_CONVERT:
            result.append(tok)
            continue
        if alpha not in _WORD_TO_DIGIT:
            result.append(tok)
            continue

        prev = re.sub(r"[^a-zA-Z]", "", tokens[i - 1]).lower() if i > 0 else ""
        nxt  = re.sub(r"[^a-zA-Z]", "", tokens[i + 1]).lower() if i < len(tokens)-1 else ""

        prev_num = prev in _WORD_TO_DIGIT and prev not in _NEVER_CONVERT
        next_num = nxt  in _WORD_TO_DIGIT and nxt  not in _NEVER_CONVERT

        convert = (
            prev in _CONTEXT_BEFORE
            or nxt  in _CONTEXT_AFTER
            or (prev_num and prev != "")
            or (next_num and nxt  != "")
        )
        result.append((_WORD_TO_DIGIT[alpha] + suffix) if convert else tok)
    return " ".join(result)


# ══════════════════════════════════════════════════════════════════════
# ██  AUDIO CONVERSION
# ══════════════════════════════════════════════════════════════════════

def _ffmpeg_convert(src: str, dest: str, rate: int) -> str:
    cmd = ["ffmpeg", "-y", "-i", src,
           "-ar", str(rate), "-ac", "1", "-sample_fmt", "s16", dest]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FFmpeg error:\n{r.stderr[-400:]}")
        raise RuntimeError("FFmpeg conversion failed. Install: winget install ffmpeg")
    return dest


def convert_to_wav_16k(src: str) -> str:
    """16 kHz PCM WAV — standard broadband for Azure STT."""
    out = str(Path(src).with_suffix(".16k.wav"))
    print(f"  Converting → 16kHz WAV: {out}")
    return _ffmpeg_convert(src, out, 16000)


def convert_to_wav_8k(src: str) -> str:
    """8 kHz PCM WAV — telephony quality (phone call audio)."""
    out = str(Path(src).with_suffix(".8k.wav"))
    print(f"  Converting → 8kHz WAV (telephony): {out}")
    return _ffmpeg_convert(src, out, 8000)


# ══════════════════════════════════════════════════════════════════════
# ██  STRUCTURED LOGGER  (Stage 11)
# ══════════════════════════════════════════════════════════════════════

_audit_log: list[dict] = []          # in-memory audit records
_alert_log: list[dict] = []          # triggered alerts


def _log_session(stage_id: str, session_id: str, event: str, data: dict):
    record = {
        "ts":         datetime.utcnow().isoformat() + "Z",
        "stage":      stage_id,
        "session_id": session_id,
        "event":      event,
        **data,
    }
    _audit_log.append(record)


def _check_alerts(stage_id: str, session_id: str, metrics: dict):
    """Evaluate alert thresholds and record triggered alerts."""
    if metrics.get("ttft_final_ms") and metrics["ttft_final_ms"] > LATENCY_SLA_MS:
        _alert_log.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "alert": "HIGH_LATENCY",
            "stage": stage_id,
            "session_id": session_id,
            "ttft_final_ms": metrics["ttft_final_ms"],
            "threshold_ms": LATENCY_SLA_MS,
        })
    if metrics.get("segment_count") == 0:
        _alert_log.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "alert": "EMPTY_TRANSCRIPT",
            "stage": stage_id,
            "session_id": session_id,
        })
    if metrics.get("error_code"):
        _alert_log.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "alert": "RECOGNITION_ERROR",
            "stage": stage_id,
            "session_id": session_id,
            "error_code": metrics["error_code"],
        })


def flush_audit_log(path: str = "transcription_audit.log"):
    with open(path, "w", encoding="utf-8") as f:
        for rec in _audit_log:
            f.write(json.dumps(rec) + "\n")
    print(f"  ✓ Audit log → {path}  ({len(_audit_log)} records, {len(_alert_log)} alerts)")


# ══════════════════════════════════════════════════════════════════════
# ██  CORE TRANSCRIPTION ENGINE  (file-based, parameterized)
# ══════════════════════════════════════════════════════════════════════

def _build_speech_config(cfg: dict) -> speechsdk.SpeechConfig:
    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

    # Language
    if cfg.get("locked_language"):
        sc.speech_recognition_language = cfg["locked_language"]
    # Recognition mode
    if cfg.get("recognition_mode") == "dictation":
        sc.enable_dictation()
    else:
        sc.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_RecognitionMode,
            "CONVERSATION",
        )
    # Profanity
    pmap = {
        "masked":  speechsdk.ProfanityOption.Masked,
        "raw":     speechsdk.ProfanityOption.Raw,
        "removed": speechsdk.ProfanityOption.Removed,
    }
    sc.set_profanity(pmap.get(cfg.get("profanity", "masked"), speechsdk.ProfanityOption.Masked))
    # Output format
    sc.output_format = (
        speechsdk.OutputFormat.Detailed
        if cfg.get("output_format", "detailed") == "detailed"
        else speechsdk.OutputFormat.Simple
    )
    # VAD timeouts
    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
                    str(cfg.get("end_silence_ms", 800)))
    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
                    str(cfg.get("initial_silence_ms", 5000)))
    sc.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                    str(cfg.get("seg_silence_ms", 600)))
    return sc


def run_file_transcription(
    wav_file: str,
    cfg: dict,
    stage_id: str = "unknown",
    baseline_transcript: str = "",
    session_id: str = "",
) -> dict:
    """
    File-based continuous recognition. Returns standardised result dict.
    Used by: Stage 0, 1, 4, 5, 6, 7, 8, 9, C1, C2.
    """
    if not session_id:
        session_id = f"{stage_id}_{int(time.time()*1000)}"

    sc = _build_speech_config(cfg)
    audio_cfg = speechsdk.audio.AudioConfig(filename=wav_file)

    locked = cfg.get("locked_language")
    auto_detect_cfg = None
    if not locked:
        auto_detect_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=cfg.get("candidate_languages", CANDIDATE_LANGUAGES)
        )

    rec_kwargs = dict(speech_config=sc, audio_config=audio_cfg)
    if auto_detect_cfg:
        rec_kwargs["auto_detect_source_language_config"] = auto_detect_cfg

    recognizer = speechsdk.SpeechRecognizer(**rec_kwargs)

    # Phrase boosting
    phrases = cfg.get("phrase_list", [])
    if phrases:
        pg = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for p in phrases:
            pg.addPhrase(p)

    # ── state ────────────────────────────────────────────────────
    final_segments:  list[dict] = []
    partial_results: list[dict] = []
    detected_lang = locked or "Unknown"
    first_partial_t = first_final_t = None
    start_t = time.time()
    done = False
    error_info = {}

    def _get_lang(evt):
        if locked:
            return locked
        try:
            return speechsdk.AutoDetectSourceLanguageResult(evt.result).language or "Unknown"
        except Exception:
            return "Unknown"

    def on_recognizing(evt):
        nonlocal first_partial_t, detected_lang
        if not evt.result.text:
            return
        now = time.time()
        if first_partial_t is None:
            first_partial_t = now
        detected_lang = _get_lang(evt)
        latency = (now - start_t) * 1000
        partial_results.append({"text": evt.result.text, "latency_ms": round(latency, 1)})
        print(f"    [PARTIAL {latency:6.0f}ms] ({detected_lang}) {evt.result.text}")

    def on_recognized(evt):
        nonlocal first_final_t, detected_lang
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        if not evt.result.text:
            return
        now = time.time()
        if first_final_t is None:
            first_final_t = now
        detected_lang = _get_lang(evt)
        latency = (now - start_t) * 1000

        # Parse confidence from detailed JSON
        confidence = None
        nbest = []
        try:
            detail = json.loads(evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult, "{}"))
            nbest = detail.get("NBest", [])
            if nbest:
                confidence = nbest[0].get("Confidence")
        except Exception:
            pass

        seg = {
            "text":       evt.result.text,
            "latency_ms": round(latency, 1),
            "language":   detected_lang,
            "confidence": confidence,
            "nbest_count": len(nbest),
        }
        final_segments.append(seg)
        conf_str = f" conf={confidence:.2f}" if confidence else ""
        print(f"    [FINAL   {latency:6.0f}ms]{conf_str} ({detected_lang}) {evt.result.text}")

    def on_canceled(evt):
        nonlocal done
        try:
            d = evt.result.cancellation_details
            error_info["reason"]      = str(d.reason)
            error_info["error_code"]  = str(d.error_code)
            error_info["error_detail"]= d.error_details
            print(f"\n    ⚠ Canceled — {d.error_code}: {d.error_details}")
        except Exception:
            pass
        done = True

    def on_stopped(evt):
        nonlocal done
        done = True

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_stopped)

    _log_session(stage_id, session_id, "START", {"wav": wav_file})
    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()
    total_time = time.time() - start_t

    raw_transcript = " ".join(s["text"] for s in final_segments)
    if cfg.get("apply_numeric_pp", False):
        processed = numeric_postprocess(raw_transcript, detected_lang)
    else:
        processed = raw_transcript

    words = processed.split()
    digit_count = sum(1 for w in words if re.fullmatch(r"\d[\d,.]*", w))
    short_count = sum(1 for w in words if 1 <= len(re.sub(r"[^a-zA-Z]", "", w)) <= 3)

    ttft_p = round((first_partial_t - start_t)*1000, 1) if first_partial_t else None
    ttft_f = round((first_final_t   - start_t)*1000, 1) if first_final_t   else None

    sim = None
    if baseline_transcript:
        sim = round(SequenceMatcher(None, baseline_transcript.lower(),
                                     processed.lower()).ratio() * 100, 1)

    # Confidence stats
    confs = [s["confidence"] for s in final_segments if s.get("confidence") is not None]
    conf_avg = round(sum(confs)/len(confs), 3) if confs else None
    conf_min = round(min(confs), 3) if confs else None
    low_conf_segs = sum(1 for c in confs if c < CONFIDENCE_REPROMPT_THRESHOLD) if confs else 0

    metrics = {
        "raw_transcript":        raw_transcript,
        "processed_transcript":  processed,
        "detected_language":     detected_lang,
        "segment_count":         len(final_segments),
        "word_count":            len(words),
        "digit_token_count":     digit_count,
        "short_word_count":      short_count,
        "ttft_partial_ms":       ttft_p,
        "ttft_final_ms":         ttft_f,
        "total_time_sec":        round(total_time, 2),
        "partial_count":         len(partial_results),
        "similarity_pct":        sim,
        "confidence_avg":        conf_avg,
        "confidence_min":        conf_min,
        "low_conf_segments":     low_conf_segs,
        "numeric_pp_applied":    cfg.get("apply_numeric_pp", False),
        "error_info":            error_info,
    }

    _log_session(stage_id, session_id, "COMPLETE", {
        "segment_count": metrics["segment_count"],
        "ttft_final_ms": ttft_f,
        "total_time_sec": round(total_time, 2),
        "error_code": error_info.get("error_code"),
    })
    _check_alerts(stage_id, session_id, metrics)

    return metrics


# ══════════════════════════════════════════════════════════════════════
# ██  STREAMING ENGINE  (Stage 3 — PushAudioInputStream)
# ══════════════════════════════════════════════════════════════════════

def run_streaming_transcription(
    wav_file: str,
    cfg: dict,
    stage_id: str = "stage_3",
    baseline_transcript: str = "",
    chunk_ms: int = 100,
) -> dict:
    """
    Streams audio in real-time chunks via PushAudioInputStream,
    simulating a WebSocket / live microphone feed.
    """
    session_id = f"{stage_id}_{int(time.time()*1000)}"

    sc = _build_speech_config(cfg)

    # Detect sample rate from WAV header
    with open(wav_file, "rb") as f:
        header = f.read(44)
    sample_rate = int.from_bytes(header[24:28], "little") if len(header) >= 28 else 16000
    bytes_per_ms = (sample_rate // 1000) * 2  # 16-bit mono
    chunk_bytes  = bytes_per_ms * chunk_ms

    stream = speechsdk.audio.PushAudioInputStream(
        speechsdk.audio.AudioStreamFormat.get_wave_format_pcm(
            samples_per_second=sample_rate,
            bits_per_sample=16,
            channels=1,
        )
    )
    audio_cfg = speechsdk.audio.AudioConfig(stream=stream)

    locked = cfg.get("locked_language")
    auto_detect_cfg = None
    if not locked:
        auto_detect_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=cfg.get("candidate_languages", CANDIDATE_LANGUAGES)
        )

    rec_kwargs = dict(speech_config=sc, audio_config=audio_cfg)
    if auto_detect_cfg:
        rec_kwargs["auto_detect_source_language_config"] = auto_detect_cfg

    recognizer = speechsdk.SpeechRecognizer(**rec_kwargs)

    phrases = cfg.get("phrase_list", [])
    if phrases:
        pg = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for p in phrases:
            pg.addPhrase(p)

    final_segments:  list[dict] = []
    partial_results: list[dict] = []
    detected_lang = locked or "Unknown"
    first_partial_t = first_final_t = None
    start_t = time.time()
    done = False
    push_done = threading.Event()

    def _get_lang(evt):
        if locked:
            return locked
        try:
            return speechsdk.AutoDetectSourceLanguageResult(evt.result).language or "Unknown"
        except Exception:
            return "Unknown"

    def on_recognizing(evt):
        nonlocal first_partial_t, detected_lang
        if not evt.result.text:
            return
        now = time.time()
        if first_partial_t is None:
            first_partial_t = now
        detected_lang = _get_lang(evt)
        latency = (now - start_t) * 1000
        partial_results.append({"text": evt.result.text, "latency_ms": round(latency, 1)})
        print(f"    [STREAM-P {latency:6.0f}ms] ({detected_lang}) {evt.result.text}")

    def on_recognized(evt):
        nonlocal first_final_t, detected_lang
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        if not evt.result.text:
            return
        now = time.time()
        if first_final_t is None:
            first_final_t = now
        detected_lang = _get_lang(evt)
        latency = (now - start_t) * 1000
        final_segments.append({
            "text": evt.result.text, "latency_ms": round(latency, 1),
            "language": detected_lang, "confidence": None,
        })
        print(f"    [STREAM-F {latency:6.0f}ms] ({detected_lang}) {evt.result.text}")

    def on_stopped(evt):
        nonlocal done
        if push_done.is_set():
            done = True

    def on_canceled(evt):
        nonlocal done
        done = True

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stopped)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition()

    print(f"    Pushing audio in {chunk_ms}ms chunks ({chunk_bytes} bytes each)…")

    def push_audio():
        try:
            with open(wav_file, "rb") as f:
                f.read(44)  # skip WAV header
                chunks_pushed = 0
                while True:
                    chunk = f.read(chunk_bytes)
                    if not chunk:
                        break
                    stream.write(chunk)
                    chunks_pushed += 1
                    time.sleep(chunk_ms / 1000.0)
            print(f"    All audio pushed ({chunks_pushed} chunks × {chunk_ms}ms)")
        finally:
            stream.close()
            push_done.set()

    push_thread = threading.Thread(target=push_audio, daemon=True)
    push_thread.start()
    push_thread.join()

    timeout_at = time.time() + 30
    while not done and time.time() < timeout_at:
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()
    total_time = time.time() - start_t

    raw_transcript = " ".join(s["text"] for s in final_segments)
    if cfg.get("apply_numeric_pp", False):
        processed = numeric_postprocess(raw_transcript, detected_lang)
    else:
        processed = raw_transcript

    words = processed.split()
    sim = None
    if baseline_transcript:
        sim = round(SequenceMatcher(None, baseline_transcript.lower(),
                                     processed.lower()).ratio() * 100, 1)

    return {
        "raw_transcript":        raw_transcript,
        "processed_transcript":  processed,
        "detected_language":     detected_lang,
        "segment_count":         len(final_segments),
        "word_count":            len(words),
        "digit_token_count":     sum(1 for w in words if re.fullmatch(r"\d[\d,.]*", w)),
        "short_word_count":      sum(1 for w in words if 1 <= len(re.sub(r"[^a-zA-Z]", "", w)) <= 3),
        "ttft_partial_ms":       round((first_partial_t - start_t)*1000, 1) if first_partial_t else None,
        "ttft_final_ms":         round((first_final_t   - start_t)*1000, 1) if first_final_t else None,
        "total_time_sec":        round(total_time, 2),
        "chunk_ms":              chunk_ms,
        "sample_rate_hz":        sample_rate,
        "similarity_pct":        sim,
        "numeric_pp_applied":    cfg.get("apply_numeric_pp", False),
        "error_info":            {},
    }


# ══════════════════════════════════════════════════════════════════════
# ██  CONCURRENCY RUNNER  (Stages 2 & 10)
# ══════════════════════════════════════════════════════════════════════

def run_concurrent_sessions(
    wav_file: str,
    cfg: dict,
    n_sessions: int,
    stage_id: str,
) -> dict:
    """Runs N recognition sessions simultaneously."""
    print(f"    Launching {n_sessions} concurrent sessions…")

    results_list: list[dict] = []
    lock = threading.Lock()

    def one_session(idx: int):
        sid = f"{stage_id}_s{idx}_{int(time.time()*1000)}"
        try:
            res = run_file_transcription(wav_file, cfg, stage_id=stage_id, session_id=sid)
            with lock:
                results_list.append({
                    "idx":          idx,
                    "success":      True,
                    "ttft_f":       res.get("ttft_final_ms"),
                    "total_time":   res.get("total_time_sec"),
                    "word_count":   res.get("word_count"),
                    "error":        None,
                })
        except Exception as exc:
            with lock:
                results_list.append({
                    "idx":    idx,
                    "success": False,
                    "ttft_f": None,
                    "error":  str(exc),
                })

    start_t = time.time()
    with ThreadPoolExecutor(max_workers=n_sessions) as pool:
        futures = [pool.submit(one_session, i) for i in range(n_sessions)]
        for f in as_completed(futures):
            pass
    wall_time = time.time() - start_t

    successes    = [r for r in results_list if r["success"]]
    failures     = [r for r in results_list if not r["success"]]
    ttft_values  = [r["ttft_f"] for r in successes if r["ttft_f"] is not None]
    ttft_sorted  = sorted(ttft_values)

    def _percentile(lst, pct):
        if not lst:
            return None
        idx = int(len(lst) * pct / 100)
        return lst[min(idx, len(lst)-1)]

    agg = {
        "n_sessions":     n_sessions,
        "n_success":      len(successes),
        "n_failed":       len(failures),
        "success_rate_pct": round(len(successes)/n_sessions*100, 1),
        "ttft_avg_ms":    round(sum(ttft_values)/len(ttft_values), 1) if ttft_values else None,
        "ttft_p50_ms":    _percentile(ttft_sorted, 50),
        "ttft_p95_ms":    _percentile(ttft_sorted, 95),
        "ttft_max_ms":    max(ttft_values) if ttft_values else None,
        "wall_time_sec":  round(wall_time, 2),
        "throttle_errors": sum(1 for r in failures if "429" in (r.get("error","") or "")),
        "failure_reasons": [r.get("error") for r in failures],
    }

    print(f"      n={n_sessions}  success={agg['success_rate_pct']}%  "
          f"ttft_avg={agg['ttft_avg_ms']}ms  p95={agg['ttft_p95_ms']}ms  "
          f"throttled={agg['throttle_errors']}")
    return agg


# ══════════════════════════════════════════════════════════════════════
# ██  HELPERS
# ══════════════════════════════════════════════════════════════════════

def extract_baseline_phrases(transcript: str, min_len=4, min_freq=2) -> list[str]:
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "is","it","was","be","are","am","have","has","had","not","this",
        "that","with","from","by","as","we","i","you","he","she","they",
        "do","did","will","would","could","should","just","so","up","if",
    }
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    freq  = Counter(words)
    return sorted(
        w for w, c in freq.items()
        if c >= min_freq and len(w) >= min_len and w not in stopwords
    )


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(int(len(s)*pct/100), len(s)-1)]


# ══════════════════════════════════════════════════════════════════════
# ██  STAGE CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_all_stages(detected_language: str, baseline_phrases: list[str]) -> dict:
    """Returns a dict of all stage configs keyed by stage_id."""

    return {

        "stage_0": {
            "_meta": {
                "id":          "stage_0",
                "name":        "Stage 0 — Baseline",
                "phase":       "Reference",
                "task":        "Original script — no modifications",
                "description": "Exact copy of the working script. Auto-detect en-US/es-ES. "
                               "All default Azure settings. This is the reference for all comparisons.",
                "parameters_changed": "None — reference state",
                "parameters": {
                    "locked_language":    "None (auto-detect en-US + es-ES)",
                    "recognition_mode":   "conversation (default)",
                    "profanity":          "masked (default)",
                    "end_silence_ms":     800,
                    "initial_silence_ms": 5000,
                    "seg_silence_ms":     600,
                    "output_format":      "detailed",
                    "phrase_list":        "none",
                    "numeric_pp":         False,
                    "audio_format":       "16kHz PCM WAV",
                },
                "expected_outcome": "Reference transcript. All stages compared against this.",
                "what_to_observe":  "Segment count, word count, digit tokens, short words, TTFT.",
            },
            "locked_language":    None,
            "candidate_languages": CANDIDATE_LANGUAGES,
            "recognition_mode":   "conversation",
            "profanity":          "masked",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_1": {
            "_meta": {
                "id":          "stage_1",
                "name":        "Stage 1 — ASR Config Finalization",
                "phase":       "Setup",
                "task":        "Lock language/locale, audio format (telephony/broadband), disable auto-detect",
                "description": "Locks language to detected. Sets profanity to raw. Eliminates auto-detect overhead.",
                "parameters_changed":
                    "locked_language (auto→detected), profanity (masked→raw), audio format test",
                "parameters": {
                    "locked_language":    f"{detected_language}  ← LOCKED (was: auto-detect)",
                    "recognition_mode":   "conversation",
                    "profanity":          "raw  ← CHANGED (was: masked)",
                    "end_silence_ms":     800,
                    "initial_silence_ms": 5000,
                    "seg_silence_ms":     600,
                    "output_format":      "detailed",
                    "phrase_list":        "none",
                    "numeric_pp":         False,
                    "audio_format":       "16kHz PCM WAV (see stage_1b for 8kHz test)",
                },
                "expected_outcome": "Faster TTFT (no auto-detect latency). No masked words.",
                "what_to_observe":
                    "TTFT-Partial vs Stage 0. Any words unmasked? "
                    "Compare 16kHz vs 8kHz transcripts for accuracy difference.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_1b": {
            "_meta": {
                "id":          "stage_1b",
                "name":        "Stage 1b — ASR Config: Telephony 8kHz Format",
                "phase":       "Setup",
                "task":        "Test telephony audio format (8kHz) vs broadband (16kHz)",
                "description": "Runs same locked-language config on 8kHz downsampled audio.",
                "parameters_changed": "audio_format: 16kHz → 8kHz (telephony)",
                "parameters": {
                    "locked_language":    f"{detected_language}",
                    "recognition_mode":   "conversation",
                    "profanity":          "raw",
                    "end_silence_ms":     800,
                    "initial_silence_ms": 5000,
                    "seg_silence_ms":     600,
                    "audio_format":       "8kHz PCM WAV  ← CHANGED (was: 16kHz)",
                    "numeric_pp":         False,
                },
                "expected_outcome": "If source is phone call audio → 8kHz may score equally or better.",
                "what_to_observe":
                    "Word count and digit accuracy vs Stage 1 (16kHz). "
                    "If 8kHz ≥ 16kHz in word count → use 8kHz in production.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
            "_audio_format":      "8k",
        },

        "stage_2": {
            "_meta": {
                "id":          "stage_2",
                "name":        "Stage 2 — Concurrency & Quota Validation",
                "phase":       "Setup",
                "task":        "Validate concurrency limits, rate limits, and quotas",
                "description": f"Runs simultaneous sessions at levels: {CONCURRENCY_LEVELS}. "
                               "Measures success rate, throttle errors, avg/P50/P95 TTFT.",
                "parameters_changed": "N/A — tests infrastructure, not ASR config",
                "parameters": {
                    "concurrency_levels": str(CONCURRENCY_LEVELS),
                    "base_config":        "Stage 1 (locked language, profanity raw)",
                    "metric_collected":   "success_rate, ttft_avg, ttft_p95, throttle_errors",
                },
                "expected_outcome": "No throttling errors. P95 TTFT < SLA threshold.",
                "what_to_observe":
                    f"At which concurrency level do errors appear? "
                    f"Is P95 TTFT within {LATENCY_SLA_MS}ms SLA?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "simple",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_3": {
            "_meta": {
                "id":          "stage_3",
                "name":        "Stage 3 — Real-Time Socket Integration",
                "phase":       "Integration",
                "task":        "Implement and validate WebSocket/streaming ingestion",
                "description": "Uses PushAudioInputStream to push audio in 100ms real-time chunks.",
                "parameters_changed": "audio_input: file → PushAudioInputStream (streaming chunks)",
                "parameters": {
                    "locked_language":    f"{detected_language}",
                    "recognition_mode":   "conversation",
                    "profanity":          "raw",
                    "end_silence_ms":     800,
                    "initial_silence_ms": 5000,
                    "seg_silence_ms":     600,
                    "chunk_ms":           100,
                    "audio_input":        "PushAudioInputStream  ← CHANGED (was: file-based)",
                    "numeric_pp":         False,
                },
                "expected_outcome": "Low-latency real-time ASR. Partials appear within 300ms of speech.",
                "what_to_observe":
                    "TTFT-Partial vs Stage 0 (file-based). "
                    "Transcript quality should match Stage 1.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_4a": {
            "_meta": {
                "id":          "stage_4a",
                "name":        "Stage 4a — VAD: Default (800ms)",
                "phase":       "Audio",
                "task":        "VAD baseline — built-in default settings",
                "description": "Same as Stage 1. Isolates VAD behaviour at default 800ms.",
                "parameters_changed": "None from Stage 1 — VAD baseline",
                "parameters": {
                    "end_silence_ms":     "800  (default)",
                    "initial_silence_ms": "5000 (default)",
                    "seg_silence_ms":     "600  (default)",
                },
                "expected_outcome": "Same as Stage 1. Segment count reference for VAD comparison.",
                "what_to_observe":  "Segment count. Are any sentences truncated mid-speech?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },
        "stage_4b": {
            "_meta": {
                "id":          "stage_4b",
                "name":        "Stage 4b — VAD: Conservative (1200ms)",
                "phase":       "Audio",
                "task":        "VAD conservative — reduce truncation and false cut-offs",
                "description": "Increases end-silence to 1200ms (+50%). Best for speakers who pause mid-sentence.",
                "parameters_changed": "end_silence_ms: 800→1200, seg_silence_ms: 600→1000, initial_silence_ms: 5000→8000",
                "parameters": {
                    "end_silence_ms":     "1200  ← INCREASED (was: 800)",
                    "initial_silence_ms": "8000  ← INCREASED (was: 5000)",
                    "seg_silence_ms":     "1000  ← INCREASED (was: 600)",
                },
                "expected_outcome": "Fewer mid-sentence truncations.",
                "what_to_observe":
                    "Segment count vs 4a (should decrease). "
                    "Word count vs 4a (should increase or equal).",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },
        "stage_4c": {
            "_meta": {
                "id":          "stage_4c",
                "name":        "Stage 4c — VAD: Aggressive (2000ms)",
                "phase":       "Audio",
                "task":        "VAD aggressive — maximum pause tolerance",
                "description": "End-silence 2000ms. Risk: may merge two utterances if speaker pauses < 2s.",
                "parameters_changed": "end_silence_ms: 1200→2000, seg_silence_ms: 1000→1500, initial_silence_ms: 8000→15000",
                "parameters": {
                    "end_silence_ms":     "2000  ← INCREASED (was: 1200)",
                    "initial_silence_ms": "15000 ← INCREASED (was: 8000)",
                    "seg_silence_ms":     "1500  ← INCREASED (was: 1000)",
                },
                "expected_outcome": "Maximum pause tolerance. Best for slow/hesitant speakers.",
                "what_to_observe":
                    "If segment count drops drastically → utterances are merging (avoid 4c).",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     2000,
            "initial_silence_ms": 15000,
            "seg_silence_ms":     1500,
            "output_format":      "detailed",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_5": {
            "_meta": {
                "id":          "stage_5",
                "name":        "Stage 5 — Word / Phrase Boosting",
                "phase":       "Accuracy",
                "task":        "Boost digits, identifiers, domain terms via PhraseListGrammar",
                "description": f"Adds {len(DOMAIN_PHRASES)} domain-specific phrases to PhraseListGrammar.",
                "parameters_changed": f"phrase_list: none → {len(DOMAIN_PHRASES)} entries",
                "parameters": {
                    "locked_language":    f"{detected_language}",
                    "recognition_mode":   "conversation",
                    "profanity":          "raw",
                    "end_silence_ms":     1200,
                    "initial_silence_ms": 8000,
                    "seg_silence_ms":     1000,
                    "phrase_list":        f"{len(DOMAIN_PHRASES)} entries  ← ADDED",
                    "phrase_categories":  "digit-sequences, identifiers, short-words, IVR-menus",
                    "numeric_pp":         False,
                },
                "expected_outcome": "Improved numeric accuracy. Short words (ID, OK) less dropped.",
                "what_to_observe":  "digit_token_count vs Stage 0. short_word_count vs Stage 0.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   False,
        },

        "stage_6": {
            "_meta": {
                "id":          "stage_6",
                "name":        "Stage 6 — Transcript-Based Vocabulary Tuning",
                "phase":       "Accuracy",
                "task":        "Use sample transcripts to refine vocabulary/style boosting",
                "description": f"Extracts words ≥2 appearances from baseline ({len(baseline_phrases)} found) "
                               "and adds to phrase list on top of Stage 5.",
                "parameters_changed":
                    f"phrase_list: Stage5 list + {len(baseline_phrases)} baseline-extracted phrases",
                "parameters": {
                    "locked_language":    f"{detected_language}",
                    "recognition_mode":   "conversation",
                    "profanity":          "raw",
                    "end_silence_ms":     1200,
                    "initial_silence_ms": 8000,
                    "seg_silence_ms":     1000,
                    "phrase_list":        f"Stage5({len(DOMAIN_PHRASES)}) + baseline({len(baseline_phrases)})  ← ADDED",
                    "numeric_pp":         False,
                },
                "expected_outcome": "Domain-specific words from your audio get a recognition boost.",
                "what_to_observe":
                    "Check if any word mis-recognised in Stage 0 is now correct. "
                    "Compare similarity_pct to Stage 5.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES + baseline_phrases,
            "apply_numeric_pp":   False,
        },

        "stage_7a": {
            "_meta": {
                "id":          "stage_7a",
                "name":        "Stage 7a — Numeric: Conversation Mode (Azure native)",
                "phase":       "Logic",
                "task":        "Validate digit-by-digit vs grouped digit behavior — baseline",
                "description": "Measures how Azure natively outputs numbers in conversation mode without post-processing.",
                "parameters_changed": "None from Stage 5 — numeric baseline",
                "parameters": {
                    "recognition_mode": "conversation",
                    "numeric_pp":       False,
                    "phrase_list":      f"{len(DOMAIN_PHRASES)} entries",
                },
                "expected_outcome": "Mixed output — some words, some digits.",
                "what_to_observe":  "digit_token_count. How many numbers appear as words vs digits?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   False,
        },
        "stage_7b": {
            "_meta": {
                "id":          "stage_7b",
                "name":        "Stage 7b — Numeric: Dictation Mode",
                "phase":       "Logic",
                "task":        "Test dictation mode for improved digit-by-digit output",
                "description": "Switches to Azure dictation mode, optimised to output spoken numbers as digit tokens.",
                "parameters_changed": "recognition_mode: conversation → dictation  ← CHANGED",
                "parameters": {
                    "recognition_mode": "dictation  ← CHANGED (was: conversation)",
                    "numeric_pp":       False,
                    "phrase_list":      f"{len(DOMAIN_PHRASES)} entries",
                },
                "expected_outcome": "More digit tokens in transcript.",
                "what_to_observe":
                    "digit_token_count vs 7a. "
                    "Check: does 'I need to go' still say 'to' (not '2')?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   False,
        },
        "stage_7c": {
            "_meta": {
                "id":          "stage_7c",
                "name":        "Stage 7c — Numeric: Dictation + Context-Aware Post-Processor",
                "phase":       "Logic",
                "task":        "Full numeric handling: dictation + context-aware word-to-digit PP",
                "description": "Dictation mode + post-processor. SAFETY: 'to/for/a/an/won/ate' NEVER converted. Spanish: pass-through.",
                "parameters_changed": "numeric_pp: False → True  ← ADDED (on top of Stage 7b)",
                "parameters": {
                    "recognition_mode": "dictation",
                    "numeric_pp":       "True  ← ADDED",
                    "never_convert":    "'to','for','a','an','won','ate' + full list",
                    "spanish_handling": "pass-through (no conversion)",
                    "phrase_list":      f"{len(DOMAIN_PHRASES)} entries",
                },
                "expected_outcome": "Maximum digit output. 'to' never becomes '2'.",
                "what_to_observe":
                    "digit_token_count vs 7b (should be ≥). "
                    "Verify 'to'/'for'/'a' stayed as words.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   True,
        },

        "stage_8": {
            "_meta": {
                "id":          "stage_8",
                "name":        "Stage 8 — Emotion / Tone Evaluation",
                "phase":       "Quality",
                "task":        "Assess ASR behavior under neutral vs stressed speech",
                "description": "Parses per-segment confidence scores and NBest alternatives. "
                               f"Flags segments below {CONFIDENCE_REPROMPT_THRESHOLD} for re-prompt.",
                "parameters_changed": "output_format: detailed (confidence + NBest parsing enabled)",
                "parameters": {
                    "recognition_mode":   "conversation",
                    "output_format":      "detailed  (parses Confidence + NBest)",
                    "confidence_threshold": CONFIDENCE_REPROMPT_THRESHOLD,
                    "profanity":          "raw",
                    "end_silence_ms":     1200,
                    "phrase_list":        f"{len(DOMAIN_PHRASES)} entries",
                    "numeric_pp":         False,
                },
                "expected_outcome": "Robust recognition. Confidence > threshold for most segments.",
                "what_to_observe":
                    "confidence_avg and confidence_min. "
                    f"low_conf_segments count (segments below {CONFIDENCE_REPROMPT_THRESHOLD}).",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   False,
        },

        "stage_9": {
            "_meta": {
                "id":          "stage_9",
                "name":        "Stage 9 — Latency & Timeout Testing",
                "phase":       "Testing",
                "task":        "Validate response times within conversational SLA",
                "description": f"Runs 3 times, collects P50/P95 TTFT. SLA: TTFT-Final < {LATENCY_SLA_MS}ms. "
                               "Also tests tight timeout (500ms).",
                "parameters_changed": "3 runs for statistical stability; tight-timeout sub-test",
                "parameters": {
                    "sla_threshold_ms":   LATENCY_SLA_MS,
                    "runs_for_stats":     3,
                    "tight_end_silence":  500,
                    "normal_end_silence": 1200,
                    "phrase_list":        f"{len(DOMAIN_PHRASES)} entries",
                    "numeric_pp":         False,
                },
                "expected_outcome": f"TTFT-Final P95 < {LATENCY_SLA_MS}ms.",
                "what_to_observe":
                    "P50 and P95 TTFT across runs. "
                    "Does tight timeout (500ms) cause truncation vs normal (1200ms)?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   False,
        },

        "stage_10": {
            "_meta": {
                "id":          "stage_10",
                "name":        "Stage 10 — Load & Concurrency Testing",
                "phase":       "Testing",
                "task":        "Validate peak concurrent real-time streams",
                "description": f"Runs {CONCURRENCY_LEVELS} concurrent sessions via ThreadPoolExecutor.",
                "parameters_changed": "concurrent sessions: 1 → multiple (ThreadPoolExecutor)",
                "parameters": {
                    "concurrency_levels": str(CONCURRENCY_LEVELS),
                    "sla_threshold_ms":   LATENCY_SLA_MS,
                    "output_format":      "simple (reduces payload under load)",
                    "phrase_list":        "none (reduces setup time under load)",
                },
                "expected_outcome": "Stable under load. No throttle errors at expected peak concurrency.",
                "what_to_observe":
                    "At which concurrency level do throttle errors appear? "
                    "P95 TTFT degradation as concurrency increases.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "conversation",
            "profanity":          "raw",
            "end_silence_ms":     800,
            "initial_silence_ms": 5000,
            "seg_silence_ms":     600,
            "output_format":      "simple",
            "phrase_list":        [],
            "apply_numeric_pp":   False,
        },

        "stage_11": {
            "_meta": {
                "id":          "stage_11",
                "name":        "Stage 11 — Logging & Alerts Setup",
                "phase":       "Monitoring",
                "task":        "Enable error, latency, socket-drop monitoring",
                "description": "Runs Combined Best config while JSON logging is active. "
                               f"Alerts: TTFT > {LATENCY_SLA_MS}ms → HIGH_LATENCY; "
                               "empty → EMPTY_TRANSCRIPT; error → RECOGNITION_ERROR.",
                "parameters_changed": "Logging + alerts layer enabled. No ASR config changes.",
                "parameters": {
                    "log_file":           "transcription_audit.log",
                    "log_format":         "JSON (one record per line)",
                    "alert_latency_ms":   LATENCY_SLA_MS,
                    "alert_empty":        "True",
                    "alert_error":        "True",
                    "base_config":        "Combined Best (Stage C1)",
                },
                "expected_outcome": "Audit log populated. Alerts file shows triggered thresholds.",
                "what_to_observe":  "transcription_audit.log record count. Any alerts triggered?",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   True,
        },

        "stage_12": {
            "_meta": {
                "id":          "stage_12",
                "name":        "Stage 12 — Fallback Validation",
                "phase":       "Go-Live",
                "task":        "Test re-prompt / DTMF / alternate flow",
                "description": f"Validates fallback: low-conf segments flagged for re-prompt "
                               f"(threshold: {CONFIDENCE_REPROMPT_THRESHOLD}); empty → DTMF fallback.",
                "parameters_changed": "Fallback logic layer (post-processing). No ASR config changes.",
                "parameters": {
                    "reprompt_threshold":   CONFIDENCE_REPROMPT_THRESHOLD,
                    "dtmf_fallback":        "Triggered when transcript is empty or all-silence",
                    "re-prompt_trigger":    f"Any segment with confidence < {CONFIDENCE_REPROMPT_THRESHOLD}",
                    "base_config":          "Combined Best",
                },
                "expected_outcome": "Resilient failure handling. No silent failures.",
                "what_to_observe":
                    "low_conf_segments count. fallback_report.reprompt_flagged. "
                    "fallback_report.dtmf_fallback.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   True,
        },

        "stage_c1": {
            "_meta": {
                "id":          "stage_c1",
                "name":        "Stage C1 — Combined Best  ✅ PRODUCTION RECOMMENDATION",
                "phase":       "Production",
                "task":        "All effective stages combined",
                "description": "Stage1 (locked lang + raw profanity) + Stage 4b (conservative VAD) + "
                               "Stage 5 (phrase boosting) + Stage 7c (dictation + numeric PP).",
                "parameters_changed": "All effective stages applied together",
                "parameters": {
                    "locked_language":    f"{detected_language}  (Stage1)",
                    "recognition_mode":   "dictation  (Stage7)",
                    "profanity":          "raw  (Stage1)",
                    "end_silence_ms":     "1200  (Stage4b)",
                    "initial_silence_ms": "8000  (Stage4b)",
                    "seg_silence_ms":     "1000  (Stage4b)",
                    "phrase_list":        f"{len(DOMAIN_PHRASES)} entries  (Stage5)",
                    "numeric_pp":         "True  (Stage7c)",
                },
                "expected_outcome": "Best overall accuracy. All individual improvements compounded.",
                "what_to_observe":
                    "similarity_pct vs Stage 0. digit_token_count (expect highest). "
                    "TTFT (expect ≤ Stage 0). word_count (expect ≥ Stage 0).",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     1200,
            "initial_silence_ms": 8000,
            "seg_silence_ms":     1000,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES,
            "apply_numeric_pp":   True,
        },

        "stage_c2": {
            "_meta": {
                "id":          "stage_c2",
                "name":        "Stage C2 — Combined All Stages",
                "phase":       "Production",
                "task":        "Every stage combined for maximum coverage",
                "description": "C1 + Stage 4c (aggressive VAD) + Stage 6 (extended vocab).",
                "parameters_changed": "Stage C1 + aggressive VAD + extended vocab",
                "parameters": {
                    "locked_language":    f"{detected_language}",
                    "recognition_mode":   "dictation",
                    "profanity":          "raw",
                    "end_silence_ms":     "2000  (Stage4c)",
                    "initial_silence_ms": "15000 (Stage4c)",
                    "seg_silence_ms":     "1500  (Stage4c)",
                    "phrase_list":        f"Stage5({len(DOMAIN_PHRASES)}) + Stage6({len(baseline_phrases)})",
                    "numeric_pp":         "True",
                },
                "expected_outcome": "Maximum coverage. Compare to C1 to check if aggressive VAD helps.",
                "what_to_observe":
                    "If segment_count same as C1 → no benefit from aggressive VAD. "
                    "If word_count higher than C1 → C2 recovered words → use C2.",
            },
            "locked_language":    detected_language,
            "recognition_mode":   "dictation",
            "profanity":          "raw",
            "end_silence_ms":     2000,
            "initial_silence_ms": 15000,
            "seg_silence_ms":     1500,
            "output_format":      "detailed",
            "phrase_list":        DOMAIN_PHRASES + baseline_phrases,
            "apply_numeric_pp":   True,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# ██  REPORT GENERATOR  (transcription_report.md — metrics-focused)
# ══════════════════════════════════════════════════════════════════════

def generate_report(all_results: dict, audio_file: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baseline_transcript = all_results.get("stage_0", {}).get("processed_transcript", "")
    lines = []

    lines.append("# Azure STT Transcription Quality Lab — Full Report")
    lines.append(f"\n**Audio:** `{audio_file}`  **Generated:** {now}  **Region:** {SPEECH_REGION}")
    lines.append(f"\n**Candidate Languages:** {CANDIDATE_LANGUAGES}  "
                 f"**SLA Threshold:** {LATENCY_SLA_MS}ms  "
                 f"**Re-prompt Threshold:** confidence < {CONFIDENCE_REPROMPT_THRESHOLD}\n")
    lines.append("---\n")

    lines.append("## Stage-by-Stage Analysis\n")

    ordered = [
        "stage_0","stage_1","stage_1b","stage_2","stage_3",
        "stage_4a","stage_4b","stage_4c",
        "stage_5","stage_6",
        "stage_7a","stage_7b","stage_7c",
        "stage_8","stage_9","stage_10","stage_11","stage_12",
        "stage_c1","stage_c2",
    ]

    for sid in ordered:
        if sid not in all_results:
            continue
        res  = all_results[sid]
        meta = res.get("_meta", {})

        lines.append(f"### {meta.get('name', sid)}")
        lines.append(f"\n**Phase:** {meta.get('phase','')}  |  **Task:** {meta.get('task','')}\n")
        lines.append(f"**Description:** {meta.get('description','')}\n")
        lines.append(f"**Parameters Changed:** `{meta.get('parameters_changed','')}`\n")

        lines.append("#### Parameters Used\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in meta.get("parameters", {}).items():
            lines.append(f"| `{k}` | `{v}` |")

        lines.append("\n#### Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        for key, label in [
            ("detected_language",    "Detected Language"),
            ("segment_count",        "Segments"),
            ("word_count",           "Word Count"),
            ("digit_token_count",    "Digit Tokens"),
            ("short_word_count",     "Short Words (1–3 chars)"),
            ("ttft_partial_ms",      "TTFT Partial (ms)"),
            ("ttft_final_ms",        "TTFT Final (ms)"),
            ("total_time_sec",       "Total Time (sec)"),
            ("similarity_pct",       "Similarity to Baseline"),
            ("confidence_avg",       "Confidence Avg"),
            ("confidence_min",       "Confidence Min"),
            ("low_conf_segments",    "Low-Conf Segments"),
            ("numeric_pp_applied",   "Numeric PP Applied"),
        ]:
            val = res.get(key)
            if val is None:
                continue
            if key == "similarity_pct":
                val = f"{val}%"
            lines.append(f"| {label} | `{val}` |")

        if "concurrency_results" in res:
            lines.append("\n#### Concurrency Results\n")
            lines.append("| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | P Max |")
            lines.append("|----------|----------|-----------|----------|-----|-----|-------|")
            for cr in res["concurrency_results"]:
                lines.append(
                    f"| {cr.get('n_sessions')} "
                    f"| {cr.get('success_rate_pct')}% "
                    f"| {cr.get('throttle_errors')} "
                    f"| {cr.get('ttft_avg_ms')} ms "
                    f"| {cr.get('ttft_p50_ms')} ms "
                    f"| {cr.get('ttft_p95_ms')} ms "
                    f"| {cr.get('ttft_max_ms')} ms |"
                )

        if "latency_runs" in res:
            lines.append("\n#### Latency Runs\n")
            lines.append("| Run | TTFT-P (ms) | TTFT-F (ms) | Total (s) | SLA Pass? |")
            lines.append("|-----|------------|------------|-----------|----------|")
            for lr in res["latency_runs"]:
                sla = "✅" if (lr.get("ttft_final_ms") or 9999) < LATENCY_SLA_MS else "❌"
                lines.append(
                    f"| {lr.get('run')} "
                    f"| {lr.get('ttft_partial_ms')} "
                    f"| {lr.get('ttft_final_ms')} "
                    f"| {lr.get('total_time_sec')} "
                    f"| {sla} |"
                )

        if "fallback_report" in res:
            fb = res["fallback_report"]
            lines.append("\n#### Fallback Report\n")
            lines.append(f"- Low-confidence segments: **{fb.get('low_conf_count',0)}**")
            lines.append(f"- Re-prompt flagged: **{fb.get('reprompt_flagged',False)}**")
            lines.append(f"- DTMF fallback triggered: **{fb.get('dtmf_fallback',False)}**")
            if fb.get("flagged_segments"):
                lines.append("\n**Flagged Segments:**")
                for fs in fb["flagged_segments"][:5]:
                    lines.append(f"  - `{fs.get('text')}` (confidence: {fs.get('confidence')})")

        lines.append(f"\n**Expected Outcome:** {meta.get('expected_outcome','')}\n")
        lines.append(f"**What to Observe:** {meta.get('what_to_observe','')}\n")

        if sid != "stage_0" and baseline_transcript:
            sim = res.get("similarity_pct")
            if sim == 100:
                lines.append("> ℹ️  Transcript **identical** to baseline — no change for this audio.\n")
            elif sim is not None:
                diff_pct = round(100 - sim, 1)
                lines.append(f"> 🔄  Transcript **{diff_pct}% different** from baseline "
                              f"(similarity: {sim}%).\n")

        lines.append("---\n")

    lines.append("## 📊 Full Comparison Table\n")
    lines.append(
        "| Stage | Phase | Seg | Words | Digits | Short | TTFT-P | TTFT-F | "
        "Time(s) | Conf-Avg | vs BL |"
    )
    lines.append(
        "|-------|-------|-----|-------|--------|-------|--------|--------|"
        "---------|----------|-------|"
    )
    for sid in ordered:
        if sid not in all_results:
            continue
        r    = all_results[sid]
        meta = r.get("_meta", {})
        lines.append(
            f"| {sid} | {meta.get('phase','')[:8]} "
            f"| {r.get('segment_count','?')} "
            f"| {r.get('word_count','?')} "
            f"| {r.get('digit_token_count','?')} "
            f"| {r.get('short_word_count','?')} "
            f"| {r.get('ttft_partial_ms','?')} "
            f"| {r.get('ttft_final_ms','?')} "
            f"| {r.get('total_time_sec','?')} "
            f"| {r.get('confidence_avg','—')} "
            f"| {str(r.get('similarity_pct','—'))+'%' if r.get('similarity_pct') is not None else '—'} |"
        )

    if _alert_log:
        lines.append("\n## ⚠️ Alerts Triggered\n")
        lines.append("| Alert | Stage | Session | Detail |")
        lines.append("|-------|-------|---------|--------|")
        for a in _alert_log:
            detail = " | ".join(f"{k}={v}" for k,v in a.items()
                                if k not in ("ts","alert","stage","session_id"))
            lines.append(f"| {a['alert']} | {a.get('stage')} | {a.get('session_id','')} | {detail} |")
    else:
        lines.append("\n## ✅ No Alerts Triggered\n")

    lines.append("## ✅ Production Recommendation\n")
    lines.append("""
**Use Stage C1 (Combined Best)** for production.

| Component | Config | Reason |
|-----------|--------|--------|
| Language | Locked (Stage 1) | Eliminates auto-detect latency |
| Profanity | Raw (Stage 1) | Full transcript fidelity |
| VAD | EndSilence=1200ms (Stage 4b) | Natural pause tolerance |
| Phrase Boost | 30+ domain phrases (Stage 5) | Domain accuracy |
| Numeric | Dictation + PP (Stage 7c) | Digit output, 'to' never '2' |
| Spanish | Pass-through | Spanish words preserved |

**Stage C2** if audio has very long pauses (>1.5s within a sentence).
---
""")

    lines.append("## 📝 Final Transcripts\n")
    lines.append("> Shown once here. 'processed' = after numeric PP (where applied).\n")
    for sid in ordered:
        if sid not in all_results:
            continue
        r    = all_results[sid]
        meta = r.get("_meta", {})
        lines.append(f"### {meta.get('name', sid)}\n")
        lines.append(f"```\n{r.get('raw_transcript','(empty)')}\n```\n")
        if r.get("numeric_pp_applied") and r.get("raw_transcript") != r.get("processed_transcript"):
            lines.append("**After numeric post-processor:**")
            lines.append(f"```\n{r.get('processed_transcript','')}\n```\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ██  DOCUMENTATION GUIDE GENERATOR  (transcription_doc_guide.md)
# ══════════════════════════════════════════════════════════════════════

def generate_doc_guide(all_results: dict, audio_file: str) -> str:
    """
    Generates a comprehensive human-readable documentation guide after all stages complete.
    Includes: quick-reference table, setup, per-stage how-to, actual results filled in,
    parameter reference, improvement identification guide, and production config.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    # ── Header ───────────────────────────────────────────────────
    lines.append("# Azure STT Transcription Quality Lab — Documentation Guide")
    lines.append(f"\n> **Audio:** `{audio_file}`  |  **Generated:** {now}  |  **Region:** `{SPEECH_REGION}`")
    lines.append(f"> **Languages Tested:** {CANDIDATE_LANGUAGES}  |  "
                 f"**SLA:** {LATENCY_SLA_MS}ms  |  "
                 f"**Re-prompt threshold:** confidence < {CONFIDENCE_REPROMPT_THRESHOLD}\n")
    lines.append("---\n")

    # ── Quick Reference Table ─────────────────────────────────────
    lines.append("## Quick Reference: Stage → Requirements Table Mapping\n")
    lines.append("| Script Stage | Phase | Task | Priority |")
    lines.append("|---|---|---|---|")
    stage_ref = [
        ("Stage 0",          "Reference",   "Baseline (original script)",                                                  "—"),
        ("Stage 1 / 1b",     "Setup",       "**ASR Config Finalization** — lock lang/locale, audio format, disable auto-detect", "High"),
        ("Stage 2",          "Setup",       "**Concurrency & Quota Validation** — validate limits, rates, quotas",         "High"),
        ("Stage 3",          "Integration", "**Real-Time Socket Integration** — WebSocket/streaming ingestion",            "High"),
        ("Stage 4a / 4b / 4c","Audio",      "**VAD Evaluation & Tuning** — silence thresholds, endpointing",              "—"),
        ("Stage 5",          "Accuracy",    "**Word / Phrase Boosting** — boost digits, identifiers, domain terms",        "High"),
        ("Stage 6",          "Accuracy",    "**Transcript-Based Vocabulary Tuning** — sample transcript vocab",            "High"),
        ("Stage 7a / 7b / 7c","Logic",      "**Numeric Handling Validation** — digit-by-digit vs grouped",                "High"),
        ("Stage 8",          "Quality",     "**Emotion / Tone Evaluation** — neutral vs stressed speech",                  "High"),
        ("Stage 9",          "Testing",     "**Latency & Timeout Testing** — response times, SLA",                         "High"),
        ("Stage 10",         "Testing",     "**Load & Concurrency Testing** — peak concurrent streams",                    "High"),
        ("Stage 11",         "Monitoring",  "**Logging & Alerts Setup** — error, latency, socket-drop",                    "High"),
        ("Stage 12",         "Go-Live",     "**Fallback Validation** — re-prompt, DTMF, alternate flow",                   "High"),
        ("Stage C1",         "Production",  "Combined Best ← **USE THIS IN PRODUCTION**",                                  "—"),
        ("Stage C2",         "Production",  "Combined All",                                                                 "—"),
    ]
    for s, ph, task, pri in stage_ref:
        lines.append(f"| {s} | {ph} | {task} | {pri} |")

    lines.append("")

    # ── Setup ─────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 1. Setup\n")
    lines.append("```bash")
    lines.append("pip install azure-cognitiveservices-speech")
    lines.append("")
    lines.append("# FFmpeg (for audio conversion)")
    lines.append("winget install ffmpeg          # Windows")
    lines.append("brew install ffmpeg            # macOS")
    lines.append("sudo apt install ffmpeg        # Ubuntu/Linux")
    lines.append("")
    lines.append("# Edit config at top of script")
    lines.append('SPEECH_KEY    = "YOUR_AZURE_SPEECH_KEY"')
    lines.append('SPEECH_REGION = "eastus"')
    lines.append('INPUT_AUDIO_FILE = "audio/your_file.mp3"')
    lines.append("```")
    lines.append("\nCustomize `DOMAIN_PHRASES` with terms specific to your domain (product names, IVR menus, identifiers).\n")

    # ── How to Run ────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 2. How to Run\n")
    lines.append("```bash")
    lines.append("# All 12 stages (recommended for full documentation)")
    lines.append("python transcription_stages_lab.py")
    lines.append("")
    lines.append("# Single stage")
    lines.append("python transcription_stages_lab.py --stage stage_5")
    lines.append("")
    lines.append("# Custom audio file")
    lines.append("python transcription_stages_lab.py --file audio/call2.mp3")
    lines.append("")
    lines.append("# Override max concurrency for Stages 2 & 10")
    lines.append("python transcription_stages_lab.py --concurrency 10")
    lines.append("")
    lines.append("# Skip audio conversion (if already 16kHz PCM WAV)")
    lines.append("python transcription_stages_lab.py --skip-wav")
    lines.append("```\n")

    lines.append("**Output files generated:**\n")
    lines.append("| File | Contents |")
    lines.append("|------|----------|")
    lines.append("| `transcription_report.md` | Full per-stage documentation, comparison table, transcripts |")
    lines.append("| `transcription_doc_guide.md` | This documentation guide with actual results filled in |")
    lines.append("| `results.json` | Machine-readable metrics for all stages |")
    lines.append("| `transcription_audit.log` | Structured JSON log records (one per line) |")
    lines.append("| Console | Live partials + finals + stage summaries |\n")

    # ── Per-Stage Detail ──────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 3. Stage-by-Stage: What It Does, Parameters, Actual Results\n")

    ordered = [
        "stage_0","stage_1","stage_1b","stage_2","stage_3",
        "stage_4a","stage_4b","stage_4c",
        "stage_5","stage_6",
        "stage_7a","stage_7b","stage_7c",
        "stage_8","stage_9","stage_10","stage_11","stage_12",
        "stage_c1","stage_c2",
    ]

    stage_doc = {
        "stage_0": {
            "what_it_does": "Your original script, unchanged. Auto-detects language between en-US and es-ES. All Azure default settings. This is the reference every other stage compares against.",
            "extra": "",
        },
        "stage_1": {
            "what_it_does": "Locks language to whatever Stage 0 detected. Sets profanity to `raw` so no words are masked. Eliminates auto-detect overhead per utterance.",
            "extra": "**Stage 1b** runs the same config on 8kHz (telephony) audio to compare formats.",
        },
        "stage_1b": {
            "what_it_does": "Same locked-language config as Stage 1, but audio is downsampled to 8kHz. Tests whether your source audio is better matched to telephony or broadband format.",
            "extra": "If Stage 1b word count ≥ Stage 1 → source is phone call audio → use 8kHz in production.",
        },
        "stage_2": {
            "what_it_does": f"Runs {CONCURRENCY_LEVELS} simultaneous recognition sessions. Tests whether your Azure subscription tier handles concurrent streams without throttling (HTTP 429).",
            "extra": "This stage validates infrastructure limits, not transcription quality.",
        },
        "stage_3": {
            "what_it_does": "Uses `PushAudioInputStream` to push audio in 100ms real-time chunks, simulating a live WebSocket or microphone feed. Validates streaming latency vs file-based.",
            "extra": "Watch for chunk-boundary artifacts — words split across 100ms boundaries.",
        },
        "stage_4a": {
            "what_it_does": "Same as Stage 1. Isolates VAD behaviour at the default 800ms end-silence. Reference point for VAD comparison.",
            "extra": "",
        },
        "stage_4b": {
            "what_it_does": "Increases end-silence to 1200ms (+50%). Best for speakers who pause mid-sentence or read numbers slowly. Reduces false cut-offs.",
            "extra": "**Decision rule:** If 4b segment count < 4a AND word count ≥ 4a → use 4b.",
        },
        "stage_4c": {
            "what_it_does": "End-silence 2000ms. Maximum pause tolerance for very slow/deliberate speakers.",
            "extra": "**Risk:** If speaker pauses > 2s between sentences, 4c may merge them. If segment count drops drastically → stick with 4b.",
        },
        "stage_5": {
            "what_it_does": f"Adds `PhraseListGrammar` with {len(DOMAIN_PHRASES)} domain-specific entries. Soft vocabulary injection — increases recognizer's prior for these phrases when acoustic evidence is ambiguous.",
            "extra": "Phrase categories: digit sequences, identifiers (account/PIN/ZIP), short words (ID/OK), IVR patterns (press one/two).",
        },
        "stage_6": {
            "what_it_does": "Extracts words that appeared ≥2 times in the baseline transcript and adds them to the phrase list on top of Stage 5. Self-bootstraps vocabulary from your domain audio.",
            "extra": "Best used when domain has unusual proper nouns or technical terms.",
        },
        "stage_7a": {
            "what_it_does": "Measures how Azure natively outputs numbers in conversation mode without any post-processing. Reference for numeric comparison.",
            "extra": "",
        },
        "stage_7b": {
            "what_it_does": "Switches to Azure dictation mode, which is trained to output spoken numbers as digit tokens natively. No post-processing added yet.",
            "extra": "Check: does `\"I need to go\"` still say `\"to\"` (not `\"2\"`)?",
        },
        "stage_7c": {
            "what_it_does": "Dictation mode + context-aware post-processor that converts remaining word-numbers to digits only in clear numeric context.",
            "extra": (
                "**Safety rules:**\n\n"
                "| Word | Converted? | Rule |\n"
                "|------|-----------|------|\n"
                "| `\"to\"` | ❌ NEVER | Preposition |\n"
                "| `\"for\"` | ❌ NEVER | Preposition |\n"
                "| `\"a\"` / `\"an\"` | ❌ NEVER | Article |\n"
                "| `\"won\"` | ❌ NEVER | Past tense |\n"
                "| `\"ate\"` | ❌ NEVER | Past tense |\n"
                "| `\"one\"` → `\"1\"` | ✅ numeric context | After `number`, `press`, etc. |\n"
                "| Spanish words | ❌ NEVER | Pass-through |\n\n"
                "**Examples:**\n"
                "```\n"
                "\"I need to go\"                  → \"I need to go\"        (to: NEVER)\n"
                "\"press one for English\"          → \"press 1 for English\" (after 'press')\n"
                "\"account number one two three\"   → \"account number 1 2 3\"\n"
                "\"there are two people\"           → \"there are 2 people\"  ('people' = quantity)\n"
                "Spanish: \"presione dos\"          → \"presione dos\"        (pass-through)\n"
                "```"
            ),
        },
        "stage_8": {
            "what_it_does": "Parses per-segment confidence scores and NBest alternatives from Azure's Detailed output. Low confidence indicates uncertainty from noise, stressed speech, accent variation.",
            "extra": f"Azure confidence ranges 0.0–1.0. Clean call-centre audio: 0.85–0.97. Values < {CONFIDENCE_REPROMPT_THRESHOLD} → human review.",
        },
        "stage_9": {
            "what_it_does": f"Runs 3 recognition passes and collects P50/P95 TTFT statistics. SLA target: TTFT-Final P95 < {LATENCY_SLA_MS}ms. Also tests with tight timeout (500ms) to check truncation risk.",
            "extra": "Alert fires in `transcription_audit.log` if any run exceeds SLA.",
        },
        "stage_10": {
            "what_it_does": f"Runs multiple concurrent sessions at increasing levels via `ThreadPoolExecutor`. Tests system stability under peak load. Azure Free tier: 1 concurrent; S0 tier: 20 concurrent.",
            "extra": "Use `wall_time_sec` to calculate throughput (sessions/second).",
        },
        "stage_11": {
            "what_it_does": "Runs Combined Best config while structured JSON logging is active. Every event (start, segment, complete, error) is logged to `transcription_audit.log`.",
            "extra": (
                "**Alert thresholds:**\n\n"
                "| Alert | Trigger |\n"
                "|-------|---------|\n"
                f"| `HIGH_LATENCY` | `ttft_final_ms > {LATENCY_SLA_MS}ms` |\n"
                "| `EMPTY_TRANSCRIPT` | `segment_count == 0` |\n"
                "| `RECOGNITION_ERROR` | `error_code` in cancellation details |\n\n"
                "**Log record format:**\n"
                "```json\n"
                '{"ts": "2025-01-15T10:23:45.123Z", "stage": "stage_11",\n'
                ' "event": "COMPLETE", "segment_count": 7,\n'
                ' "ttft_final_ms": 412.5, "total_time_sec": 18.3}\n'
                "```"
            ),
        },
        "stage_12": {
            "what_it_does": f"Validates fallback handling. Segments with confidence < {CONFIDENCE_REPROMPT_THRESHOLD} are flagged for re-prompt. Empty transcripts trigger DTMF fallback.",
            "extra": (
                "**Fallback logic:**\n"
                "```\n"
                "IF transcript == '' or segment_count == 0  → DTMF fallback\n"
                f"IF any segment confidence < {CONFIDENCE_REPROMPT_THRESHOLD}     → flag for re-prompt\n"
                "IF error_code present                      → log + alert + fallback\n"
                "```"
            ),
        },
        "stage_c1": {
            "what_it_does": "Combines all stages that show measurable improvement: locked language, raw profanity, conservative VAD, phrase boosting, dictation mode, numeric PP.",
            "extra": (
                "| Component | From Stage | Config |\n"
                "|-----------|-----------|--------|\n"
                "| Locked language | Stage 1 | `speech_recognition_language` |\n"
                "| Profanity raw | Stage 1 | `ProfanityOption.Raw` |\n"
                "| Conservative VAD | Stage 4b | `end_silence_ms=1200` |\n"
                "| Phrase boosting | Stage 5 | `PhraseListGrammar` |\n"
                "| Dictation mode | Stage 7b | `enable_dictation()` |\n"
                "| Numeric PP | Stage 7c | `apply_numeric_pp=True` |"
            ),
        },
        "stage_c2": {
            "what_it_does": "C1 + aggressive VAD (Stage 4c) + extended vocabulary (Stage 6).",
            "extra": (
                "**Use C2 over C1 when:** Audio has very long pauses (>1.5s within a sentence) "
                "or Stage C2 word count > C1.\n\n"
                "**Stick with C1 when:** Stage C2 segment count << C1 (utterances merging)."
            ),
        },
    }

    for sid in ordered:
        if sid not in all_results:
            continue
        res  = all_results[sid]
        meta = res.get("_meta", {})
        doc  = stage_doc.get(sid, {})

        lines.append(f"### {meta.get('name', sid)}\n")
        lines.append(f"**Requirements phase:** {meta.get('phase','')}  \n")

        if doc.get("what_it_does"):
            lines.append(f"**What it does:** {doc['what_it_does']}\n")

        # Parameters Changed table
        if meta.get("parameters", {}):
            lines.append("**Parameters:**\n")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for k, v in meta["parameters"].items():
                lines.append(f"| `{k}` | `{v}` |")
            lines.append("")

        if doc.get("extra"):
            lines.append(doc["extra"])
            lines.append("")

        # Actual results from this run
        lines.append("**Actual Results from This Run:**\n")
        if res.get("segment_count") is not None:
            lines.append(f"- Segments: **{res['segment_count']}**")
        if res.get("word_count") is not None:
            lines.append(f"- Words: **{res['word_count']}**")
        if res.get("digit_token_count") is not None:
            lines.append(f"- Digit tokens: **{res['digit_token_count']}**")
        if res.get("short_word_count") is not None:
            lines.append(f"- Short words: **{res['short_word_count']}**")
        if res.get("ttft_partial_ms") is not None:
            lines.append(f"- TTFT Partial: **{res['ttft_partial_ms']} ms**")
        if res.get("ttft_final_ms") is not None:
            sla_icon = "✅" if res["ttft_final_ms"] < LATENCY_SLA_MS else "❌"
            lines.append(f"- TTFT Final: **{res['ttft_final_ms']} ms** {sla_icon}")
        if res.get("total_time_sec") is not None:
            lines.append(f"- Total time: **{res['total_time_sec']} s**")
        if res.get("confidence_avg") is not None:
            lines.append(f"- Confidence avg/min: **{res['confidence_avg']} / {res['confidence_min']}**")
        if res.get("low_conf_segments") is not None:
            lines.append(f"- Low-conf segments: **{res['low_conf_segments']}**")
        if res.get("similarity_pct") is not None:
            diff = round(100 - res["similarity_pct"], 1)
            lines.append(f"- vs Baseline: **{res['similarity_pct']}% similar** ({diff}% different)")

        # Concurrency results
        if "concurrency_results" in res:
            lines.append("\n**Concurrency Results:**\n")
            lines.append("| Sessions | Success% | Throttled | TTFT Avg | P50 | P95 | Max |")
            lines.append("|----------|----------|-----------|----------|-----|-----|-----|")
            for cr in res["concurrency_results"]:
                throttle_flag = " ⚠️" if cr.get("throttle_errors", 0) > 0 else ""
                lines.append(
                    f"| {cr.get('n_sessions')} "
                    f"| {cr.get('success_rate_pct')}% "
                    f"| {cr.get('throttle_errors')}{throttle_flag} "
                    f"| {cr.get('ttft_avg_ms')} ms "
                    f"| {cr.get('ttft_p50_ms')} ms "
                    f"| {cr.get('ttft_p95_ms')} ms "
                    f"| {cr.get('ttft_max_ms')} ms |"
                )

        # Latency runs
        if "latency_runs" in res:
            p50 = res.get("ttft_p50_ms")
            p95 = res.get("ttft_p95_ms")
            sla_pass = "✅ PASS" if res.get("sla_pass") else "❌ FAIL"
            lines.append(f"\n**Latency Statistics:** P50={p50}ms | P95={p95}ms | SLA {sla_pass}\n")
            lines.append("| Run | TTFT-P (ms) | TTFT-F (ms) | Total (s) | SLA |")
            lines.append("|-----|------------|------------|-----------|-----|")
            for lr in res["latency_runs"]:
                sla = "✅" if (lr.get("ttft_final_ms") or 9999) < LATENCY_SLA_MS else "❌"
                lines.append(
                    f"| {lr.get('run')} "
                    f"| {lr.get('ttft_partial_ms')} "
                    f"| {lr.get('ttft_final_ms')} "
                    f"| {lr.get('total_time_sec')} "
                    f"| {sla} |"
                )

        # Fallback report
        if "fallback_report" in res:
            fb = res["fallback_report"]
            lines.append(f"\n**Fallback Report:**")
            lines.append(f"- Low-confidence segments: **{fb.get('low_conf_count', 0)}**")
            lines.append(f"- Re-prompt flagged: **{fb.get('reprompt_flagged', False)}**")
            lines.append(f"- DTMF fallback triggered: **{fb.get('dtmf_fallback', False)}**")

        lines.append(f"\n**What to observe:** {meta.get('what_to_observe','')}\n")
        lines.append("---\n")

    # ── Parameters Reference ──────────────────────────────────────
    lines.append("## 4. Parameters Reference\n")
    lines.append("| Parameter | Azure Property / Method | Default | Stage C1 Value | Effect |")
    lines.append("|-----------|------------------------|---------|----------------|--------|")
    params_ref = [
        ("`locked_language`",  "`speech_config.speech_recognition_language`", "None", "detected",      "Removes auto-detect latency"),
        ("`recognition_mode`", "`enable_dictation()`",                        "conversation", "dictation", "Better native digit output"),
        ("`profanity`",        "`set_profanity(ProfanityOption.Raw)`",        "masked", "raw",          "No words censored"),
        ("`end_silence_ms`",   "`SpeechServiceConnection_EndSilenceTimeoutMs`", "800", "1200",          "Tolerate natural pauses"),
        ("`initial_silence_ms`","`SpeechServiceConnection_InitialSilenceTimeoutMs`","5000","8000",      "More time before speech"),
        ("`seg_silence_ms`",   "`Speech_SegmentationSilenceTimeoutMs`",       "600", "1000",           "Mid-sentence pause tolerance"),
        ("`phrase_list`",      "`PhraseListGrammar.addPhrase()`",             "none", "30+ entries",   "Soft vocabulary boost"),
        ("`apply_numeric_pp`", "Python post-processor",                       "False", "True",         "Word→digit (contextual)"),
        ("`output_format`",    "`OutputFormat.Detailed`",                     "Detailed", "Detailed",  "Confidence + NBest parsing"),
    ]
    for row in params_ref:
        lines.append(f"| {row[0]} | {row[1]} | `{row[2]}` | `{row[3]}` | {row[4]} |")
    lines.append("")

    # ── Observation Log (filled in from actual results) ───────────
    lines.append("---\n")
    lines.append("## 5. Results Summary Table\n")
    lines.append("> Auto-filled from this run. Use to compare improvements across stages.\n")
    lines.append("| Stage | Phase | Seg | Words | Digits | Short | TTFT-P | TTFT-F | Conf | vs BL | Key Finding |")
    lines.append("|-------|-------|-----|-------|--------|-------|--------|--------|------|-------|-------------|")

    baseline_words  = all_results.get("stage_0", {}).get("word_count", 0) or 0
    baseline_digits = all_results.get("stage_0", {}).get("digit_token_count", 0) or 0

    for sid in ordered:
        if sid not in all_results:
            continue
        r    = all_results[sid]
        meta = r.get("_meta", {})
        bl   = f"{r['similarity_pct']}%" if r.get("similarity_pct") is not None else "—"
        cf   = f"{r['confidence_avg']:.2f}" if r.get("confidence_avg") is not None else "—"
        seg  = str(r.get("segment_count", "—"))
        wc   = str(r.get("word_count", "—"))
        dig  = str(r.get("digit_token_count", "—"))
        sht  = str(r.get("short_word_count", "—"))
        tp   = str(r.get("ttft_partial_ms", "—"))
        tf   = str(r.get("ttft_final_ms", "—"))

        # Auto-generate a key finding
        finding = ""
        if sid == "stage_0":
            finding = "Baseline reference"
        elif r.get("similarity_pct") == 100:
            finding = "No change vs baseline"
        elif r.get("similarity_pct") is not None:
            diff = round(100 - r["similarity_pct"], 1)
            wc_diff = (r.get("word_count") or 0) - baseline_words
            wc_str  = f"+{wc_diff} words" if wc_diff > 0 else (f"{wc_diff} words" if wc_diff < 0 else "same words")
            dig_diff = (r.get("digit_token_count") or 0) - baseline_digits
            dig_str  = f"+{dig_diff} digits" if dig_diff > 0 else ""
            finding = f"{diff}% change; {wc_str}" + (f"; {dig_str}" if dig_str else "")
        elif "concurrency_results" in r:
            throttled = sum(c.get("throttle_errors", 0) for c in r["concurrency_results"])
            finding = "No throttle errors" if throttled == 0 else f"⚠️ {throttled} throttle error(s)"
        elif "latency_runs" in r:
            sla_pass = r.get("sla_pass")
            finding = f"P95={r.get('ttft_p95_ms')}ms — SLA {'✅' if sla_pass else '❌'}"

        lines.append(
            f"| {sid} | {meta.get('phase','')[:8]} | {seg} | {wc} | {dig} | {sht} "
            f"| {tp} | {tf} | {cf} | {bl} | {finding} |"
        )

    lines.append("")

    # ── Improvement Identification ────────────────────────────────
    lines.append("---\n")
    lines.append("## 6. Identifying Improvement at Each Stage\n")
    lines.append("| Stage | ✅ Improvement means... | ❌ No improvement means... |")
    lines.append("|-------|----------------------|--------------------------|")
    improvement_guide = [
        ("Stage 1",   "TTFT-P lower than Stage 0",               "TTFT same — network latency dominates"),
        ("Stage 1b",  "Word count ≥ Stage 1",                    "Word count < Stage 1 — 8kHz loses quality"),
        ("Stage 2",   "100% success at expected concurrency",     "Any 429 errors at expected peak"),
        ("Stage 3",   "Transcript ≈ Stage 1 quality",            "Words dropped at chunk boundaries"),
        ("Stage 4b",  "Fewer segments, same/more words",          "Same segments & words — VAD was fine"),
        ("Stage 5",   "More digit tokens, short words preserved", "Same as Stage 4b — phrases not in audio"),
        ("Stage 6",   "Mis-recognised words now correct",        "Identical to Stage 5"),
        ("Stage 7c",  "More digit tokens than 7b",               "Same as 7b — PP had nothing to convert"),
        ("Stage 8",   "Low `low_conf_segments`",                 "Many low-conf segs → noisy/stressed audio"),
        ("Stage 9",   "P95 TTFT < SLA threshold",               "P95 > SLA → consider closer Azure region"),
        ("Stage C1",  "All gains from above compounded",         "Identical to Stage 0 → audio already optimal"),
    ]
    for s, good, bad in improvement_guide:
        lines.append(f"| {s} | {good} | {bad} |")
    lines.append("")

    lines.append("**Interpreting `vs BL` (similarity to baseline):**\n")
    lines.append("| Similarity | Meaning |")
    lines.append("|-----------|---------|")
    lines.append("| `100%` | Transcript unchanged — stage had no effect on this audio |")
    lines.append("| `95–99%` | Minor wording changes (a few words) |")
    lines.append("| `85–94%` | Noticeable changes (digit conversions, recovered words) |")
    lines.append("| `< 85%` | Major changes — verify whether they are improvements |")
    lines.append("")

    # ── Production Config ─────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 7. Production Configuration (Copy-Paste)\n")
    lines.append("```python")
    lines.append("import azure.cognitiveservices.speech as speechsdk")
    lines.append("")
    lines.append('SPEECH_KEY    = "YOUR_KEY"')
    lines.append('SPEECH_REGION = "eastus"')
    lines.append("")
    lines.append("def build_production_recognizer(wav_file: str, language: str = \"en-US\"):")
    lines.append('    """Stage C1 — Production configuration."""')
    lines.append("    sc = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)")
    lines.append("")
    lines.append("    # Stage 1: Lock language, raw profanity")
    lines.append("    sc.speech_recognition_language = language")
    lines.append("    sc.set_profanity(speechsdk.ProfanityOption.Raw)")
    lines.append("")
    lines.append("    # Stage 7b: Dictation mode for better digit output")
    lines.append("    sc.enable_dictation()")
    lines.append("")
    lines.append("    # Detailed output for confidence scores (Stage 8/12)")
    lines.append("    sc.output_format = speechsdk.OutputFormat.Detailed")
    lines.append("")
    lines.append("    # Stage 4b: Conservative VAD")
    lines.append('    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,     "1200")')
    lines.append('    sc.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "8000")')
    lines.append('    sc.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,             "1000")')
    lines.append("")
    lines.append("    audio_cfg  = speechsdk.audio.AudioConfig(filename=wav_file)")
    lines.append("    recognizer = speechsdk.SpeechRecognizer(speech_config=sc, audio_config=audio_cfg)")
    lines.append("")
    lines.append("    # Stage 5: Phrase boosting")
    lines.append("    phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)")
    lines.append("    for phrase in DOMAIN_PHRASES:")
    lines.append("        phrase_list.addPhrase(phrase)")
    lines.append("")
    lines.append("    return recognizer")
    lines.append("")
    lines.append("")
    lines.append("# After recognition, apply Stage 7c numeric post-processor:")
    lines.append("# final_text = numeric_postprocess(raw_transcript, language=detected_language)")
    lines.append("```\n")

    # ── Troubleshooting ───────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 8. Troubleshooting\n")
    lines.append("| Problem | Cause | Fix |")
    lines.append("|---------|-------|-----|")
    troubleshooting = [
        ("`AuthenticationFailure`",                "Wrong SPEECH_KEY or SPEECH_REGION",       "Check Azure portal"),
        ("`FileNotFoundError`",                    "Audio file path wrong",                   "Check `INPUT_AUDIO_FILE`"),
        ("`FFmpeg conversion failed`",             "FFmpeg not installed",                    "`winget install ffmpeg`"),
        ("Empty transcript",                       "Network issue or long silence at start",  "Increase `initial_silence_ms`, check connectivity"),
        ("Transcript identical across all stages", "Audio already well-handled by Azure defaults", "Test with noisier/faster audio"),
        ("`\"to\"` converted to `\"2\"`",         "Bug in `_NEVER_CONVERT`",                 "Verify you are using latest script version"),
        ("Stage 2/10 throttle errors",             "Azure tier limit reached",                "Upgrade to S0+ or add retry-with-backoff"),
        (f"P95 TTFT > {LATENCY_SLA_MS}ms SLA",   "Network latency to Azure region",          "Switch to closer region (`westus`, `westeurope`)"),
        ("Low confidence on all segments",         "Very noisy audio",                        "Consider audio pre-processing (noise reduction)"),
        ("Stage 3 chunk boundary artifacts",       "Chunk too small",                         "Increase `chunk_ms` from 100 to 200"),
    ]
    for prob, cause, fix in troubleshooting:
        lines.append(f"| {prob} | {cause} | {fix} |")

    lines.append("")

    # ── Alerts Summary ────────────────────────────────────────────
    lines.append("---\n")
    if _alert_log:
        lines.append("## ⚠️ Alerts Triggered During This Run\n")
        lines.append("| Alert Type | Stage | Detail |")
        lines.append("|-----------|-------|--------|")
        for a in _alert_log:
            detail = ", ".join(f"{k}={v}" for k, v in a.items()
                               if k not in ("ts", "alert", "stage", "session_id"))
            lines.append(f"| `{a['alert']}` | {a.get('stage')} | {detail} |")
    else:
        lines.append("## ✅ No Alerts Triggered During This Run\n")
        lines.append("All sessions completed within SLA thresholds. No empty transcripts or errors detected.\n")

    lines.append("")
    lines.append("---\n")
    lines.append(f"*Azure STT Transcription Quality Lab — Documentation Guide  |  "
                 f"Generated: {now}  |  Audio: `{audio_file}`*\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ██  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════

ORDERED_STAGE_IDS = [
    "stage_0", "stage_1", "stage_1b", "stage_2",  "stage_3",
    "stage_4a","stage_4b","stage_4c",
    "stage_5", "stage_6",
    "stage_7a","stage_7b","stage_7c",
    "stage_8", "stage_9", "stage_10","stage_11","stage_12",
    "stage_c1","stage_c2",
]


def print_banner(name: str, sid: str, current: int, total: int):
    pct = int(current / total * 40)
    bar = "█"*pct + "░"*(40-pct)
    print(f"\n{'═'*68}")
    print(f"  [{current:2d}/{total}]  {sid.upper()}  —  {name}")
    print(f"  [{bar}]  {int(current/total*100)}%")
    print(f"{'═'*68}\n")


def main():
    parser = argparse.ArgumentParser(description="Azure STT Quality Lab — All 12 Stages")
    parser.add_argument("--file",        default=INPUT_AUDIO_FILE)
    parser.add_argument("--stage",       default="all",
                        help="all | stage_0 | stage_1 | stage_2 | … | stage_c1 | stage_c2")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Override max concurrency level for Stage 2 / 10")
    parser.add_argument("--skip-wav",    action="store_true",
                        help="Skip FFmpeg conversion (audio already 16kHz PCM WAV)")
    args = parser.parse_args()

    audio_file = args.file
    run_all    = args.stage == "all"
    run_ids    = ORDERED_STAGE_IDS if run_all else [args.stage]
    if args.concurrency:
        global CONCURRENCY_LEVELS
        CONCURRENCY_LEVELS = list(range(1, args.concurrency + 1, max(1, args.concurrency // 3)))

    print(f"\n{'═'*68}")
    print(f"  AZURE STT TRANSCRIPTION QUALITY LAB  —  ALL 12 STAGES")
    print(f"  Audio   : {audio_file}")
    print(f"  Stages  : {len(run_ids)}")
    print(f"  SLA     : {LATENCY_SLA_MS}ms TTFT")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*68}\n")

    # ── Convert audio ─────────────────────────────────────────────
    if args.skip_wav:
        wav_16k = audio_file
        wav_8k  = audio_file
        print(f"  Using as-is: {wav_16k}")
    else:
        wav_16k = convert_to_wav_16k(audio_file)
        wav_8k  = convert_to_wav_8k(audio_file)

    all_results: dict = {}

    # ── Stage 0 MUST run first ────────────────────────────────────
    temp_stages = build_all_stages("", [])
    if "stage_0" in run_ids:
        s0_cfg = temp_stages["stage_0"]
        print_banner("Baseline", "stage_0", 1, len(run_ids))
        r0 = run_file_transcription(wav_16k, s0_cfg, stage_id="stage_0")
        r0["_meta"] = s0_cfg["_meta"]
        all_results["stage_0"] = r0
        detected_language   = r0["detected_language"] or "en-US"
        baseline_transcript = r0["processed_transcript"]
        print(f"\n  ✓ Detected language: {detected_language}")
        print(f"  ✓ Words: {r0['word_count']}  Digits: {r0['digit_token_count']}")
    else:
        detected_language   = "en-US"
        baseline_transcript = ""
        print("  ⚠  Stage 0 not in run list. Defaulting detected language to en-US.\n")

    baseline_phrases = extract_baseline_phrases(baseline_transcript)
    print(f"  Extracted {len(baseline_phrases)} baseline phrases for Stage 6.\n")

    stages = build_all_stages(detected_language, baseline_phrases)
    remaining = [s for s in run_ids if s != "stage_0"]

    for idx, sid in enumerate(remaining, start=2):
        if sid not in stages:
            print(f"\n  ⚠  Unknown stage ID: {sid}  — skipping.")
            continue

        cfg  = stages[sid]
        meta = cfg["_meta"]
        print_banner(meta["name"], sid, idx, len(run_ids))
        print(f"  Parameters Changed: {meta['parameters_changed']}")
        print()

        # ── Special stage handlers ────────────────────────────────

        if sid == "stage_1b":
            result = run_file_transcription(
                wav_8k, cfg,
                stage_id=sid, baseline_transcript=baseline_transcript
            )

        elif sid == "stage_2":
            conc_levels = args.concurrency and [args.concurrency] or CONCURRENCY_LEVELS
            conc_results = []
            for n in conc_levels:
                print(f"  → Testing {n} concurrent session(s)…")
                cr = run_concurrent_sessions(wav_16k, cfg, n, sid)
                conc_results.append(cr)
            result = {
                "detected_language":    detected_language,
                "concurrency_results":  conc_results,
                "segment_count":        None,
                "word_count":           None,
                "digit_token_count":    None,
                "short_word_count":     None,
                "ttft_partial_ms":      conc_results[-1].get("ttft_avg_ms") if conc_results else None,
                "ttft_final_ms":        conc_results[-1].get("ttft_avg_ms") if conc_results else None,
                "total_time_sec":       sum(c.get("wall_time_sec",0) for c in conc_results),
                "similarity_pct":       None,
                "confidence_avg":       None,
                "confidence_min":       None,
                "low_conf_segments":    None,
                "numeric_pp_applied":   False,
                "error_info":           {},
            }

        elif sid == "stage_3":
            result = run_streaming_transcription(
                wav_16k, cfg,
                stage_id=sid, baseline_transcript=baseline_transcript
            )

        elif sid == "stage_9":
            runs = []
            ttft_f_values = []
            for run_i in range(1, 4):
                print(f"  → Run {run_i}/3…")
                r = run_file_transcription(
                    wav_16k, cfg,
                    stage_id=sid, baseline_transcript=baseline_transcript,
                    session_id=f"stage_9_run{run_i}"
                )
                runs.append({
                    "run":            run_i,
                    "ttft_partial_ms": r.get("ttft_partial_ms"),
                    "ttft_final_ms":   r.get("ttft_final_ms"),
                    "total_time_sec":  r.get("total_time_sec"),
                })
                if r.get("ttft_final_ms"):
                    ttft_f_values.append(r["ttft_final_ms"])

            tight_cfg = dict(cfg)
            tight_cfg["end_silence_ms"] = 500
            print("  → Run tight-timeout (500ms end-silence)…")
            r_tight = run_file_transcription(
                wav_16k, tight_cfg,
                stage_id=sid+"_tight", baseline_transcript=baseline_transcript
            )
            runs.append({
                "run":             "tight-500ms",
                "ttft_partial_ms": r_tight.get("ttft_partial_ms"),
                "ttft_final_ms":   r_tight.get("ttft_final_ms"),
                "total_time_sec":  r_tight.get("total_time_sec"),
            })

            result = dict(runs[-2])
            result.update({
                "latency_runs":      runs,
                "ttft_p50_ms":       _percentile(sorted(ttft_f_values), 50),
                "ttft_p95_ms":       _percentile(sorted(ttft_f_values), 95),
                "sla_pass":          all(v < LATENCY_SLA_MS for v in ttft_f_values),
                "detected_language": detected_language,
                "confidence_avg":    None,
                "confidence_min":    None,
                "low_conf_segments": None,
                "numeric_pp_applied": False,
                "error_info":        {},
            })

        elif sid == "stage_10":
            conc_levels = args.concurrency and [args.concurrency] or CONCURRENCY_LEVELS
            conc_results = []
            for n in conc_levels:
                print(f"  → Load test: {n} concurrent sessions…")
                cr = run_concurrent_sessions(wav_16k, cfg, n, sid)
                conc_results.append(cr)
            result = {
                "detected_language":    detected_language,
                "concurrency_results":  conc_results,
                "segment_count":        None,
                "word_count":           None,
                "digit_token_count":    None,
                "short_word_count":     None,
                "ttft_partial_ms":      None,
                "ttft_final_ms":        conc_results[-1].get("ttft_p95_ms") if conc_results else None,
                "total_time_sec":       sum(c.get("wall_time_sec",0) for c in conc_results),
                "similarity_pct":       None,
                "confidence_avg":       None,
                "confidence_min":       None,
                "low_conf_segments":    None,
                "numeric_pp_applied":   False,
                "error_info":           {},
            }

        elif sid == "stage_12":
            result = run_file_transcription(
                wav_16k, cfg,
                stage_id=sid, baseline_transcript=baseline_transcript
            )
            low_conf = result.get("low_conf_segments", 0) or 0
            dtmf     = result.get("segment_count", 1) == 0
            result["fallback_report"] = {
                "low_conf_count":   low_conf,
                "reprompt_flagged": low_conf > 0,
                "dtmf_fallback":    dtmf,
                "flagged_segments": [],
            }

        else:
            result = run_file_transcription(
                wav_16k, cfg,
                stage_id=sid, baseline_transcript=baseline_transcript
            )

        result["_meta"] = meta
        all_results[sid] = result

        # ── Quick stage summary ────────────────────────────────────
        print(f"\n  ── Summary ─────────────────────────────────────────────")
        if result.get("segment_count") is not None:
            print(f"  Segments       : {result['segment_count']}")
        if result.get("word_count") is not None:
            print(f"  Words          : {result['word_count']}")
        if result.get("digit_token_count") is not None:
            print(f"  Digit tokens   : {result['digit_token_count']}")
        if result.get("ttft_final_ms") is not None:
            sla_ok = "✅" if result["ttft_final_ms"] < LATENCY_SLA_MS else "❌"
            print(f"  TTFT Final     : {result['ttft_final_ms']} ms  {sla_ok} (<{LATENCY_SLA_MS}ms SLA)")
        if result.get("confidence_avg") is not None:
            print(f"  Conf Avg/Min   : {result['confidence_avg']} / {result['confidence_min']}")
        if result.get("low_conf_segments"):
            print(f"  Low-conf segs  : {result['low_conf_segments']}")
        if result.get("similarity_pct") is not None:
            print(f"  vs Baseline    : {result['similarity_pct']}% similar")
        if "concurrency_results" in result:
            for cr in result["concurrency_results"]:
                throttle = f"  ❌ {cr['throttle_errors']} THROTTLED" if cr["throttle_errors"] else ""
                print(f"  n={cr['n_sessions']:2d} → success={cr['success_rate_pct']}%  "
                      f"p95={cr['ttft_p95_ms']}ms{throttle}")
        print(f"  {'─'*55}")

    # ── Print quick comparison table ─────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'═'*80}")
    print(f"{'Stage':12s} {'Phase':10s} {'Seg':5s} {'Words':6s} {'Dig':5s} {'Sht':5s} "
          f"{'TTFT-P':8s} {'TTFT-F':8s} {'Conf':6s} {'vs BL':7s}")
    print("─"*80)
    for sid in ORDERED_STAGE_IDS:
        if sid not in all_results:
            continue
        r    = all_results[sid]
        meta = r.get("_meta", {})
        bl   = f"{r['similarity_pct']}%" if r.get("similarity_pct") is not None else "  —"
        cf   = f"{r['confidence_avg']:.2f}" if r.get("confidence_avg") is not None else "  —"
        print(
            f"{sid:12s} "
            f"{meta.get('phase','?')[:10]:10s} "
            f"{str(r.get('segment_count','?')):5s} "
            f"{str(r.get('word_count','?')):6s} "
            f"{str(r.get('digit_token_count','?')):5s} "
            f"{str(r.get('short_word_count','?')):5s} "
            f"{str(r.get('ttft_partial_ms','?')):8s} "
            f"{str(r.get('ttft_final_ms','?')):8s} "
            f"{cf:6s} "
            f"{bl:7s}"
        )

    # ══════════════════════════════════════════════════════════════
    # ██  SAVE ALL OUTPUTS
    # ══════════════════════════════════════════════════════════════

    # JSON results
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "_meta"}
        return obj
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({sid: _clean(r) for sid, r in all_results.items()}, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ results.json")

    # Metrics report
    report = generate_report(all_results, audio_file)
    with open("transcription_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ transcription_report.md")

    # Documentation guide  ← NEW
    doc_guide = generate_doc_guide(all_results, audio_file)
    with open("transcription_doc_guide.md", "w", encoding="utf-8") as f:
        f.write(doc_guide)
    print(f"  ✓ transcription_doc_guide.md  ← Documentation guide with actual results")

    # Audit log
    flush_audit_log("transcription_audit.log")

    print(f"\n{'═'*68}")
    print(f"  Done. {len(all_results)} stages completed.  Alerts: {len(_alert_log)}")
    print(f"")
    print(f"  Output files:")
    print(f"    transcription_report.md      ← Stage-by-stage metrics")
    print(f"    transcription_doc_guide.md   ← Full documentation guide (actual results filled in)")
    print(f"    results.json                 ← Machine-readable metrics")
    print(f"    transcription_audit.log      ← Structured JSON event log")
    print(f"{'═'*68}\n")
    return all_results


if __name__ == "__main__":
    main()
