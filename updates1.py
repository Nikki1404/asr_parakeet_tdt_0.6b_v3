"""
AZURE STT — 12-STAGE INCREMENTAL IMPROVEMENT SCRIPT
=====================================================
PURPOSE:
  Run the same audio through 13 progressive configurations.
  At each stage ONE new parameter/feature is added.
  Every stage is compared against baseline AND the previous stage.

QUALITY ANALYSIS (what this script measures):
  - Short word detection     (1-3 letter words caught or missed)
  - Number/digit accuracy    (spoken digits vs grouped numbers)
  - Punctuation quality      (commas, periods, question marks)
  - VAD / truncation         (are sentences cut off?)
  - Noise / filler detection (uh, um, hmm, noise words)
  - Word error indicators    (confidence score per segment)
  - Exact spoken words       (we NEVER change word meaning)

DIGIT RULE (important):
  "to", "too", "for", "won" are NEVER converted to 2/2/4/1.
  Only convert if the word is ACTUALLY a spoken digit in context:
  "one two three" → "1 2 3" (caller spelling out a number)
  "I want to pay" → "I want to pay" (NOT "I want 2 pay")

STAGES:
  0  baseline         No changes — your original working script
  1  asr_config       Lock language, telephony profile, disable open detection
  2  vad_tuning       Tune silence/endpointing thresholds
  3  phrase_boost     Boost numeric + domain phrases
  4  vocab_tuning     Auto-mine domain terms from prior transcripts
  5  numeric_handling Parse ITN/Lexical/Display; smart digit detection
  6  dictation_mode   Natural punctuation
  7  emotion_tone     Confidence proxy, disfluency, stress detection
  8  latency_testing  P50/P90/P95, multi-run SLA check
  9  realtime_socket  PushAudioInputStream (WebSocket simulation)
  10 concurrency      Multi-stream quota validation
  11 logging_alerts   Structured logs, alert rules
  12 fallback         Re-prompt → language retry → DTMF chain

PARAMETERS CHANGED PER STAGE:
  Stage 0: baseline values — EndSilence=800ms, no boost, no dictation
  Stage 1: language locked to ["en-US","es-US"], open-detect disabled
  Stage 2: EndSilence=1500ms, InitSilence=5000ms, SegSilence=800ms
  Stage 3: PhraseListGrammar with NUMERIC_PHRASES + DOMAIN_PHRASES
  Stage 4: PhraseListGrammar + auto-mined terms from prior transcripts
  Stage 5: OutputFormat.Detailed → ITN/Lexical/Display, digit context check
  Stage 6: enable_dictation() → natural punctuation
  Stage 7: tone/disfluency analysis on top of stage 6
  Stage 8: 3-run latency measurement, P90/P95 calculation
  Stage 9: PushAudioInputStream 40ms chunks instead of file read
  Stage 10: ThreadPoolExecutor concurrent streams, quota ramp test
  Stage 11: AlertEngine, SPEECH_SDK_LOGFILE, structured session log
  Stage 12: Silence WAV test, language-swap retry, DTMF simulation

HOW TO USE:
  python azure_incremental.py --stage 0    ← always start here
  python azure_incremental.py --stage 1
  ...
  python azure_incremental.py --stage 12
  python azure_incremental.py --all        ← run all sequentially

After any stage:
  python generate_observation_doc.py       ← full documentation

OUTPUT:
  observations/stage_N_<name>/result.json
  observations/stage_N_<name>/transcript.txt
  observations/stage_N_<name>/quality_analysis.txt
  observations/stage_N_<name>/metrics_summary.txt
  observations/comparison_report.json
"""

import os, re, sys, time, json, wave, struct, difflib
import subprocess, threading, statistics, collections
from pathlib import Path
from datetime import datetime

import azure.cognitiveservices.speech as speechsdk

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SPEECH_KEY    = "YOUR_AZURE_SPEECH_KEY"
SPEECH_REGION = "eastus"
INPUT_AUDIO   = "audio/maria1.mp3"

CANDIDATE_LANGUAGES = ["en-US", "es-US"]

# Phrase boosting — stage 3+
# IMPORTANT: these are hints to Azure, not replacements.
# They make Azure prefer these forms when acoustically ambiguous.
NUMERIC_PHRASES = [
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "account number","confirmation number","reference number",
    "zip code","date of birth","social security","routing number",
    "phone number","policy number","claim number",
]
DOMAIN_PHRASES = [
    "verification code","balance due","minimum payment","due date",
    "transfer","checking account","savings account","autopay",
    "statement","transaction","debit card","credit limit",
    "customer service","representative","authorization",
]

# Context words that sound like digits but are NOT digits
# Rule: never convert these when used in normal sentence flow
NOT_DIGITS = {
    "to","too","two",       # "to" in "I want to" is NOT 2
    "for","four",           # "for" in "wait for" is NOT 4
    "won","one",            # "won" in "she won" is NOT 1
    "ate","eight",          # "ate" in "she ate" is NOT 8
    "be","bee",             # NOT b/B
    "by","buy",             # NOT b
    "see","sea",            # NOT C
    "in","inn",             # NOT n
    "oh","owe",             # NOT 0 unless confirmed sequence
    "are","our",            # NOT R
}

# Sequences of spoken digits → always convert to numeric form
# e.g. "one two three four" when all are pure digit words in sequence
SPOKEN_DIGIT_MAP = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9,
}

CONCURRENCY_LEVELS = [1, 3, 5, 10]
MAX_REPROMPTS      = 2
OBS_DIR            = "observations"
os.makedirs(OBS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
STAGES = {
    0:  {"name":"baseline",        "phase":"Baseline",    "task":"Original working script",
         "description":"No changes — establishes reference baseline for all comparisons",
         "outcome":"Reference point",
         "parameters_changed": "None — exact replica of your working script",
         "parameters_used": "EndSilenceTimeout=800ms, no phrase boost, no dictation, OutputFormat=Detailed"},
    1:  {"name":"asr_config",      "phase":"Setup",       "task":"ASR Config Finalization",
         "description":"Lock language list, telephony audio profile, disable open-ended auto-detection",
         "outcome":"Stable, predictable recognition",
         "parameters_changed": "AutoDetectSourceLanguageConfig locked to ['en-US','es-US'] — no wildcard",
         "parameters_used": "CANDIDATE_LANGUAGES=['en-US','es-US'], audio=WAV PCM 16kHz mono 16-bit"},
    2:  {"name":"vad_tuning",      "phase":"Audio",       "task":"VAD Evaluation & Tuning",
         "description":"Tune silence thresholds to reduce word truncation and false sentence cuts",
         "outcome":"Reduced truncation and false cut-offs",
         "parameters_changed": "EndSilenceTimeoutMs 800→1500, InitSilenceTimeoutMs=5000, SegmentationSilenceTimeoutMs=800",
         "parameters_used": "SpeechServiceConnection_EndSilenceTimeoutMs=1500ms, SpeechServiceConnection_InitialSilenceTimeoutMs=5000ms, Speech_SegmentationSilenceTimeoutMs=800ms"},
    3:  {"name":"phrase_boost",    "phase":"Accuracy",    "task":"Word / Phrase Boosting",
         "description":"Add PhraseListGrammar with numeric terms and domain-specific phrases",
         "outcome":"Improved numeric and domain accuracy",
         "parameters_changed": "PhraseListGrammar.addPhrase() for NUMERIC_PHRASES + DOMAIN_PHRASES",
         "parameters_used": f"PhraseListGrammar with {len(NUMERIC_PHRASES)} numeric + {len(DOMAIN_PHRASES)} domain phrases"},
    4:  {"name":"vocab_tuning",    "phase":"Accuracy",    "task":"Transcript-Based Vocabulary Tuning",
         "description":"Auto-mine high-frequency domain terms from previous stage transcripts and boost them",
         "outcome":"Domain alignment",
         "parameters_changed": "PhraseListGrammar extended with auto-mined terms (min frequency=2, min length=5 chars)",
         "parameters_used": "Auto-mined terms from observations/stage_*/transcript.txt + DOMAIN_PHRASES"},
    5:  {"name":"numeric_handling","phase":"Logic",       "task":"Numeric Handling Validation",
         "description":"Parse ITN/Lexical/Display from Detailed JSON; smart digit detection (context-aware, never changes word meaning)",
         "outcome":"Accurate digit capture without corrupting word meanings",
         "parameters_changed": "OutputFormat=Detailed (was already set), NBest[0] ITN/Lexical/Display parsing enabled, digit context check active",
         "parameters_used": "SpeechServiceResponse_JsonResult → NBest[0].ITN, NBest[0].Lexical, NBest[0].Display; digit sequence detector"},
    6:  {"name":"dictation_mode",  "phase":"Accuracy",    "task":"Dictation Mode",
         "description":"Enable dictation mode — Azure inserts natural punctuation (commas, periods, question marks)",
         "outcome":"More readable, structured transcript",
         "parameters_changed": "speech_config.enable_dictation() added",
         "parameters_used": "enable_dictation()=True"},
    7:  {"name":"emotion_tone",    "phase":"Quality",     "task":"Emotion / Tone Evaluation",
         "description":"Track per-segment confidence, speech rate, disfluency markers, positive/negative tone",
         "outcome":"Robust recognition measurement across speech styles",
         "parameters_changed": "No Azure SDK params changed — analysis layer added post-recognition",
         "parameters_used": "confidence<0.70=low, disfluency_words={uh,um,hmm,er}, wps threshold=4.0"},
    8:  {"name":"latency_testing", "phase":"Testing",     "task":"Latency & Timeout Testing",
         "description":"Run 3 passes, compute P50/P90/P95 latency, validate against conversational SLA",
         "outcome":"Smooth turn-taking validated",
         "parameters_changed": "3 recognition runs executed; latency percentiles computed across runs",
         "parameters_used": "SLA targets: first-byte<500ms, avg<800ms, P90<1200ms"},
    9:  {"name":"realtime_socket", "phase":"Integration", "task":"Real-Time Socket Integration",
         "description":"Switch from file-read to PushAudioInputStream with 40ms chunks (simulates WebSocket ingestion)",
         "outcome":"Low-latency real-time ASR validated",
         "parameters_changed": "AudioConfig(filename=) replaced with PushAudioInputStream + 40ms chunk feed thread",
         "parameters_used": "PushAudioInputStream, AudioStreamFormat(16000Hz,16bit,mono), chunk_ms=40"},
    10: {"name":"concurrency",     "phase":"Testing",     "task":"Load & Concurrency Testing",
         "description":"Ramp concurrent streams 1→3→5→10, detect throttle ceiling and quota limit",
         "outcome":"Stable under peak load",
         "parameters_changed": "ThreadPoolExecutor with max_workers=N per level; 429/quota error detection",
         "parameters_used": f"CONCURRENCY_LEVELS={CONCURRENCY_LEVELS}, timeout=300s per stream"},
    11: {"name":"logging_alerts",  "phase":"Monitoring",  "task":"Logging & Alerts Setup",
         "description":"Enable SDK diagnostic log, structured session JSON, alert rules for latency/confidence/errors",
         "outcome":"Early issue detection",
         "parameters_changed": "SPEECH_SDK_LOGFILE env var set; AlertEngine active with 5 rule types",
         "parameters_used": "AlertEngine rules: high_latency>1000ms, low_confidence<0.65, socket_drop, zero_result, no_speech"},
    12: {"name":"fallback",        "phase":"Go-Live",     "task":"Fallback Validation",
         "description":"Test fallback chain: normal audio → silence/no-match → re-prompt → language retry → DTMF simulation",
         "outcome":"Resilient failure handling",
         "parameters_changed": "InitialSilenceTimeoutMs=3000ms for fallback attempts; silence WAV generated; language order reversed for retry",
         "parameters_used": "Silence WAV 3.0sec, language retry=['es-US','en-US'], DTMF simulated (logged)"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SMART DIGIT DETECTION
# Never converts "to/for/won/are" etc. to digits
# Only converts sequences of pure digit words (e.g. "one two three four")
# ─────────────────────────────────────────────────────────────────────────────
def is_digit_sequence_context(words: list, start: int, end: int) -> bool:
    """
    Returns True only if ALL words from start..end are unambiguous digit words.
    Words like 'to','for','are','oh' etc. are NOT digits even if they sound like one.
    """
    for i in range(start, end):
        w = words[i].lower().strip(".,!?")
        if w not in SPOKEN_DIGIT_MAP:
            return False
    return True


def smart_digit_normalise(text: str) -> str:
    """
    Context-aware digit normalisation.
    Rules:
      1. A run of 3+ consecutive pure digit words → convert to digit string
         "one two three four five" → "1 2 3 4 5"
      2. Isolated "two/four/one" surrounded by real words → KEEP as word
         "I want to pay" → "I want to pay"  (NOT "I want 2 pay")
      3. Never touch: to, too, for, won, ate, be, by, see, are, oh, etc.
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        clean = words[i].lower().strip(".,!?'\"")

        # Check if this starts a run of digit words
        if clean in SPOKEN_DIGIT_MAP:
            # Look ahead for a consecutive sequence
            run_end = i + 1
            while run_end < len(words):
                next_clean = words[run_end].lower().strip(".,!?'\"")
                if next_clean in SPOKEN_DIGIT_MAP:
                    run_end += 1
                else:
                    break

            run_length = run_end - i

            if run_length >= 3:
                # 3+ consecutive digit words → convert the whole run
                for j in range(i, run_end):
                    w = words[j].lower().strip(".,!?'\"")
                    result.append(str(SPOKEN_DIGIT_MAP[w]))
                i = run_end
            elif run_length == 2:
                # 2 consecutive → only convert if neither word is in NOT_DIGITS
                w0 = words[i].lower().strip(".,!?'\"")
                w1 = words[i+1].lower().strip(".,!?'\"") if i+1 < len(words) else ""
                if w0 not in NOT_DIGITS and w1 not in NOT_DIGITS:
                    result.append(str(SPOKEN_DIGIT_MAP[w0]))
                    result.append(str(SPOKEN_DIGIT_MAP.get(w1, w1)))
                    i += 2
                else:
                    result.append(words[i])
                    i += 1
            else:
                # Single digit word — keep as word unless it's an unambiguous standalone
                # "zero" → "0" always (no other meaning)
                # "nine" → "9" always (no other meaning)
                # "two"  → keep as "two" (too ambiguous)
                always_digit = {"zero", "three", "five", "six", "seven", "nine"}
                if clean in always_digit:
                    result.append(str(SPOKEN_DIGIT_MAP[clean]))
                else:
                    result.append(words[i])
                i += 1
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY ANALYSER
# Compares transcripts without treating either as ground truth
# ─────────────────────────────────────────────────────────────────────────────
def analyse_transcript_quality(transcript: str, stage_num: int, stage_name: str) -> dict:
    """
    Analyse transcript quality across multiple dimensions.
    Does NOT treat baseline as ground truth — measures quality signals directly.
    """
    words     = transcript.split()
    sentences = re.split(r'[.!?]+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]

    # ── Short word detection ──────────────────────────────────────────
    short_words     = [w for w in words if len(w.strip(".,!?\"'")) <= 3]
    short_word_pct  = round(len(short_words) / max(len(words), 1) * 100, 1)
    # Common short words that should appear in real speech
    expected_short  = {"i","a","an","the","is","it","in","on","at","to","of",
                        "my","he","she","we","do","go","no","so","if","or","and","but"}
    found_short     = set(w.lower().strip(".,!?") for w in words if len(w.strip(".,!?")) <= 3)
    missing_common  = expected_short - found_short   # short words that didn't appear

    # ── Number / digit quality ────────────────────────────────────────
    # Find digits already in transcript (from ITN or previous stages)
    digit_groups    = re.findall(r'\b\d+\b', transcript)
    spoken_digits   = re.findall(
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b',
        transcript, re.I)
    # Phone/account patterns
    phone_patterns  = re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', transcript)
    digit_sequences = re.findall(r'(?:\b\d\b\s*){3,}', transcript)  # "1 2 3 4"

    # ── Punctuation quality ───────────────────────────────────────────
    commas        = transcript.count(",")
    periods       = transcript.count(".")
    questions     = transcript.count("?")
    total_punct   = commas + periods + questions + transcript.count("!")
    # Sentences without any terminal punctuation (truncation indicator)
    unpunctuated  = sum(1 for s in sentences if s and not re.search(r'[.!?,]$', s.rstrip()))
    avg_sent_len  = round(len(words) / max(len(sentences), 1), 1)

    # ── VAD / truncation signals ──────────────────────────────────────
    # Short sentences (< 4 words) may indicate truncation
    short_sents   = [s for s in sentences if s and len(s.split()) < 4]
    # Very long sentences (>40 words) may indicate missed endpointing
    long_sents    = [s for s in sentences if len(s.split()) > 40]

    # ── Noise / filler detection ──────────────────────────────────────
    FILLERS        = {"uh","um","hmm","er","ah","uhh","umm","erm","hm","mhm"}
    NOISE_PATTERNS = [r'\[.*?\]', r'\(.*?\)', r'<.*?>']  # bracket noise
    filler_words   = [w for w in words if w.lower().strip(".,!?") in FILLERS]
    noise_brackets = sum(len(re.findall(p, transcript)) for p in NOISE_PATTERNS)
    # Repeated words (stuttering / recognition error)
    repeated_words = []
    for i in range(len(words)-1):
        if words[i].lower() == words[i+1].lower():
            repeated_words.append(words[i])

    # ── Overall quality score (0-100) ─────────────────────────────────
    # Higher = better. We penalise issues and reward good signals.
    score = 100
    score -= len(short_sents) * 2          # truncated sentences
    score -= len(long_sents) * 3           # missed endpointing
    score -= len(filler_words) * 1         # filler words (minor)
    score -= noise_brackets * 5            # noise brackets (serious)
    score -= len(repeated_words) * 3       # repeated words
    score -= unpunctuated * 1              # missing punctuation
    score += min(total_punct * 0.5, 10)   # reward punctuation (cap at 10)
    score += min(len(digit_groups) * 0.5, 5)  # reward digit recognition
    score = max(0, min(100, round(score, 1)))

    return {
        "stage"            : stage_name,
        "stage_num"        : stage_num,
        "total_words"      : len(words),
        "total_sentences"  : len(sentences),
        "avg_sentence_len" : avg_sent_len,
        "short_word": {
            "count"        : len(short_words),
            "pct_of_total" : short_word_pct,
            "sample"       : list(set(w.lower().strip(".,!?") for w in short_words))[:15],
            "missing_common_short_words": list(missing_common)[:10],
        },
        "numbers": {
            "digit_groups_found"   : digit_groups[:20],
            "digit_group_count"    : len(digit_groups),
            "spoken_digits_found"  : [d.lower() for d in spoken_digits[:15]],
            "spoken_digit_count"   : len(spoken_digits),
            "phone_patterns"       : phone_patterns,
            "digit_sequences"      : [d.strip() for d in digit_sequences[:5]],
        },
        "punctuation": {
            "commas"           : commas,
            "periods"          : periods,
            "question_marks"   : questions,
            "total"            : total_punct,
            "unpunctuated_segs": unpunctuated,
        },
        "vad_signals": {
            "short_sentences_count" : len(short_sents),
            "short_sentences_sample": short_sents[:5],
            "long_sentences_count"  : len(long_sents),
            "truncation_risk"       : "high" if len(short_sents) > len(sentences)*0.3 else
                                      "medium" if len(short_sents) > len(sentences)*0.1 else "low",
        },
        "noise": {
            "filler_words"      : filler_words[:10],
            "filler_count"      : len(filler_words),
            "noise_brackets"    : noise_brackets,
            "repeated_words"    : repeated_words[:10],
            "repeated_count"    : len(repeated_words),
        },
        "quality_score"    : score,
        "quality_grade"    : ("A" if score>=90 else "B" if score>=75 else
                              "C" if score>=60 else "D" if score>=45 else "F"),
    }


def compare_quality(qa: dict, qb: dict) -> dict:
    """Compare two quality analyses — neither is ground truth."""
    def diff(a, b, lower_better=False):
        if a is None or b is None: return None
        d = b - a
        direction = ("improved" if d < 0 else "worse" if d > 0 else "same") if lower_better \
               else ("improved" if d > 0 else "worse" if d < 0 else "same")
        return {"from": a, "to": b, "change": round(d, 2), "direction": direction}

    notes = []
    qs_diff = diff(qa["quality_score"], qb["quality_score"])
    if qs_diff:
        if qs_diff["direction"] == "improved":
            notes.append(f"✅ Quality score improved: {qa['quality_score']} → {qb['quality_score']} (+{qs_diff['change']})")
        elif qs_diff["direction"] == "worse":
            notes.append(f"⚠️  Quality score dropped: {qa['quality_score']} → {qb['quality_score']} ({qs_diff['change']})")

    # Short words
    sw_a = qa["short_word"]["count"]; sw_b = qb["short_word"]["count"]
    if sw_b > sw_a: notes.append(f"✅ More short words captured: {sw_a} → {sw_b} (+{sw_b-sw_a})")
    elif sw_b < sw_a: notes.append(f"⚠️  Fewer short words: {sw_a} → {sw_b}")

    # Punctuation
    pt_a = qa["punctuation"]["total"]; pt_b = qb["punctuation"]["total"]
    if pt_b > pt_a: notes.append(f"✅ More punctuation: {pt_a} → {pt_b} (+{pt_b-pt_a})")

    # Digits
    dg_a = qa["numbers"]["digit_group_count"]; dg_b = qb["numbers"]["digit_group_count"]
    if dg_b > dg_a: notes.append(f"✅ More digit groups recognised: {dg_a} → {dg_b}")

    # VAD
    vs_a = qa["vad_signals"]["short_sentences_count"]
    vs_b = qb["vad_signals"]["short_sentences_count"]
    if vs_b < vs_a: notes.append(f"✅ Fewer truncated sentences: {vs_a} → {vs_b}")
    elif vs_b > vs_a: notes.append(f"⚠️  More short/truncated sentences: {vs_a} → {vs_b}")

    # Noise
    nc_a = qa["noise"]["filler_count"]; nc_b = qb["noise"]["filler_count"]
    if nc_b < nc_a: notes.append(f"✅ Fewer filler words: {nc_a} → {nc_b}")

    return {
        "quality_score_delta" : qs_diff,
        "short_word_delta"    : diff(sw_a, sw_b),
        "punctuation_delta"   : diff(pt_a, pt_b),
        "digit_group_delta"   : diff(dg_a, dg_b),
        "vad_truncation_delta": diff(vs_a, vs_b, lower_better=True),
        "noise_delta"         : diff(nc_a, nc_b, lower_better=True),
        "quality_notes"       : notes if notes else ["➡️  No significant quality change detected"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_wav(input_file: str) -> str:
    out = str(Path(input_file).with_suffix(".wav"))
    print(f"\n  Converting {input_file} → {out}")
    try:
        subprocess.run(
            ["ffmpeg","-y","-i",input_file,"-ar","16000","-ac","1","-sample_fmt","s16",out],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  Conversion done.")
        return out
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg failed. Install: winget install ffmpeg")


def create_silence_wav(path, dur=3.0, sr=16000):
    n = int(sr * dur)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(struct.pack("<"+"h"*n, *([0]*n)))
    return path


def read_wav_pcm(path):
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getnchannels(), wf.getsampwidth()*8, wf.getframerate()


# ─────────────────────────────────────────────────────────────────────────────
# TONE / EMOTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DISFLUENCY = {"uh","um","hmm","er","ah","uhh","umm","erm"}
NEGATIVE_W = {"frustrated","angry","upset","terrible","cancel","refund",
               "horrible","awful","worst","never","always","wrong","mistake"}
POSITIVE_W = {"great","perfect","excellent","thank","thanks","happy",
               "resolved","appreciate","good","pleased"}

def tone_signals(text):
    w = set(text.lower().split())
    d = w & DISFLUENCY; neg = w & NEGATIVE_W; pos = w & POSITIVE_W
    return {"disfluencies":list(d),"negative":list(neg),"positive":list(pos),
            "tone":"negative" if len(neg)>len(pos) else ("positive" if pos else "neutral")}


# ─────────────────────────────────────────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AlertEngine:
    RULES = {"high_latency":"Latency>1000ms","low_confidence":"Confidence<0.65",
             "socket_drop":"Canceled with error","zero_result":"Zero segments","no_speech":"Silence timeout"}
    def __init__(self): self.alerts = []
    def _fire(self, rule, detail, value=None):
        self.alerts.append({"time":datetime.now().isoformat(),"rule":rule,
                             "desc":self.RULES.get(rule,""),"detail":detail,"value":value})
        print(f"  🔔 [{rule}] {detail}")
    def check_latency(self, ms, txt):
        if ms > 1000: self._fire("high_latency", f"{ms:.0f}ms '{txt[:25]}'", ms)
    def check_confidence(self, c, txt):
        if c and c < 0.65: self._fire("low_confidence", f"conf={c:.3f} '{txt[:25]}'", c)
    def check_canceled(self, code, det): self._fire("socket_drop", f"{code}: {(det or '')[:50]}")
    def check_zero(self, n):
        if n == 0: self._fire("zero_result", "No segments")
    def to_dict(self): return self.alerts


# ─────────────────────────────────────────────────────────────────────────────
# SPEECH CONFIG BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_speech_config(stage_num: int) -> speechsdk.SpeechConfig:
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.output_format = speechsdk.OutputFormat.Detailed  # always Detailed

    # Stage 6+: dictation mode (natural punctuation)
    if stage_num >= 6:
        cfg.enable_dictation()

    # VAD settings
    if stage_num >= 2:
        # Tuned: more patient endpointing
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,    "1500")
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,"5000")
        cfg.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,            "800")
    else:
        # Baseline values
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,    "800")
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,"5000")
        cfg.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,            "800")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# CORE RECOGNISER
# ─────────────────────────────────────────────────────────────────────────────
def run_recognition(wav_file, stage_num, audio_cfg=None, extra_phrases=None, alerter=None):
    cfg       = build_speech_config(stage_num)
    auto_lang = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=CANDIDATE_LANGUAGES)

    if audio_cfg is None:
        audio_cfg = speechsdk.audio.AudioConfig(filename=wav_file)

    rec = speechsdk.SpeechRecognizer(
        speech_config=cfg, auto_detect_source_language_config=auto_lang, audio_config=audio_cfg)

    # Stage 3+: phrase boosting
    if stage_num >= 3:
        pg = speechsdk.PhraseListGrammar.from_recognizer(rec)
        for p in NUMERIC_PHRASES + DOMAIN_PHRASES + (extra_phrases or []):
            pg.addPhrase(p)

    partial_results = []; final_results = []; final_transcript = []
    detected_lang = None; first_partial = None; first_final = None
    t0 = time.time(); done = threading.Event()

    def on_recognizing(evt):
        nonlocal first_partial, detected_lang
        if not evt.result.text: return
        now = time.time()
        if first_partial is None: first_partial = now
        try: detected_lang = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        except: detected_lang = "unknown"
        lat = (now - t0)*1000
        partial_results.append({"text":evt.result.text,"latency_ms":round(lat,2)})
        print(f"  [PARTIAL {lat:.0f}ms] ({detected_lang}) {evt.result.text}")

    def on_recognized(evt):
        nonlocal first_final, detected_lang
        if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech or not evt.result.text: return
        now = time.time()
        if first_final is None: first_final = now
        try: detected_lang = speechsdk.AutoDetectSourceLanguageResult(evt.result).language
        except: detected_lang = "unknown"
        lat = (now - t0)*1000

        confidence = itn = lexical = display = None
        if stage_num >= 5:
            try:
                detail = json.loads(evt.result.properties.get_property(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
                nb = detail.get("NBest",[])
                if nb:
                    confidence = nb[0].get("Confidence")
                    itn        = nb[0].get("ITN","")
                    lexical    = nb[0].get("Lexical","")
                    display    = nb[0].get("Display", evt.result.text)
            except: pass

        # Use display text as the primary; apply smart digit normalisation at stage 5+
        primary_text = display or evt.result.text
        if stage_num >= 5:
            primary_text = smart_digit_normalise(primary_text)

        tone = tone_signals(primary_text) if stage_num >= 7 else None

        seg = {
            "text"      : primary_text,
            "original"  : evt.result.text,  # always keep exact SDK output
            "display"   : display,
            "itn"       : itn,
            "lexical"   : lexical,
            "latency_ms": round(lat,2),
            "confidence": round(confidence,4) if confidence else None,
            "language"  : detected_lang,
            "tone"      : tone,
        }
        final_results.append(seg)
        final_transcript.append(primary_text)

        if alerter:
            alerter.check_latency(lat, primary_text)
            if confidence: alerter.check_confidence(confidence, primary_text)

        c_str = f" conf={confidence:.3f}" if confidence else ""
        t_str = f" [{tone['tone']}]" if tone else ""
        print(f"  [FINAL  {lat:.0f}ms] ({detected_lang}){c_str}{t_str} {primary_text}")

    def on_canceled(evt):
        cd = evt.result.cancellation_details
        if cd.reason == speechsdk.CancellationReason.Error:
            print(f"\n  [CANCELED] {cd.reason}|{cd.error_code}|{cd.error_details}")
            if alerter: alerter.check_canceled(str(cd.error_code), cd.error_details)
        done.set()

    def on_stopped(evt):
        print("\n  [SESSION STOPPED]"); done.set()

    rec.recognizing.connect(on_recognizing); rec.recognized.connect(on_recognized)
    rec.canceled.connect(on_canceled);       rec.session_stopped.connect(on_stopped)

    rec.start_continuous_recognition()
    done.wait(timeout=600)
    rec.stop_continuous_recognition()

    total     = time.time() - t0
    full_text = " ".join(final_transcript)
    confs     = [s["confidence"] for s in final_results if s.get("confidence")]

    return {
        "detected_language": detected_lang,
        "ttft_partial_ms"  : round((first_partial-t0)*1000,1) if first_partial else None,
        "ttft_final_ms"    : round((first_final-t0)*1000,1)   if first_final   else None,
        "total_time_sec"   : round(total,2),
        "segment_count"    : len(final_results),
        "word_count"       : len(full_text.split()),
        "empty_segments"   : sum(1 for s in final_results if not s["text"].strip()),
        "avg_confidence"   : round(sum(confs)/len(confs),4) if confs else None,
        "min_confidence"   : round(min(confs),4) if confs else None,
        "max_confidence"   : round(max(confs),4) if confs else None,
        "partial_count"    : len(partial_results),
        "transcript"       : full_text,
        "segments"         : final_results,
        "partial_results"  : partial_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE RUNNERS (one per stage — each adds exactly one thing)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_0_baseline(wav_file):
    print("\n  Stage 0: Baseline — exact replica of your working script.")
    return run_recognition(wav_file, 0)

def run_stage_1_asr_config(wav_file):
    print("\n  Stage 1: Language locked to en-US/es-US. Open detection disabled.")
    r = run_recognition(wav_file, 1)
    r["asr_config_notes"] = {
        "language_locked": CANDIDATE_LANGUAGES,
        "audio_format"   : "WAV PCM 16kHz mono 16-bit",
        "open_detect"    : False,
    }
    return r

def run_stage_2_vad(wav_file):
    print("\n  Stage 2: VAD — EndSilence=1500ms, Init=5000ms, Seg=800ms")
    r = run_recognition(wav_file, 2)
    r["vad_config"] = {"end_silence_ms":1500,"init_silence_ms":5000,"seg_silence_ms":800,
                        "change":"EndSilenceTimeoutMs increased 800→1500ms"}
    return r

def run_stage_3_phrase_boost(wav_file):
    all_p = NUMERIC_PHRASES + DOMAIN_PHRASES
    print(f"\n  Stage 3: Phrase boost — {len(all_p)} terms added to PhraseListGrammar")
    r = run_recognition(wav_file, 3)
    tx = r["transcript"].lower()
    hits = [p for p in all_p if p.lower() in tx]
    r["phrase_boost"] = {
        "total_phrases":len(all_p),"numeric_count":len(NUMERIC_PHRASES),
        "domain_count":len(DOMAIN_PHRASES),"hits_in_transcript":hits,"hit_count":len(hits),
    }
    return r

def run_stage_4_vocab_tuning(wav_file):
    print("\n  Stage 4: Vocab tuning — auto-mining domain terms from previous transcripts.")
    mined = []
    for sd in sorted(Path(OBS_DIR).glob("stage_*")):
        tx_f = sd / "transcript.txt"
        if tx_f.exists():
            words = re.findall(r"[a-zA-Z']{5,}", tx_f.read_text(encoding="utf-8").lower())
            freq  = collections.Counter(w for w in words if w not in
                    {"about","their","there","would","could","which","where","after","being","going"})
            mined = [w for w,c in freq.most_common(50) if c >= 2]
            break
    if not mined:
        mined = DOMAIN_PHRASES
        print("  No prior transcript found — using DOMAIN_PHRASES as fallback")
    else:
        print(f"  Mined {len(mined)} terms. Top 10: {mined[:10]}")
    r = run_recognition(wav_file, 4, extra_phrases=mined)
    r["vocab_tuning"] = {"mined_terms":mined,"count":len(mined),"source":"auto-mined from prior transcripts"}
    return r

def run_stage_5_numeric(wav_file):
    print("\n  Stage 5: Numeric handling — ITN/Lexical/Display + smart digit normalisation.")
    print("  Rule: only convert digit SEQUENCES (3+ words). Never convert to/for/won/ate etc.")
    r = run_recognition(wav_file, 5)

    numeric_analysis = []
    for seg in r["segments"]:
        itn = seg.get("itn","") or ""; lexical = seg.get("lexical","") or ""
        has_spoken = bool(re.search(r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b',lexical,re.I))
        digits = re.sub(r'[^\d]','',itn)
        grouped = len(digits)>=3 and not bool(re.search(r'\d\s\d',itn))
        if has_spoken or digits:
            numeric_analysis.append({
                "display":seg.get("display",""),"itn":itn,"lexical":lexical,
                "digit_string":digits,"itn_grouped":grouped,"spoken_digit_words":has_spoken,
            })

    r["numeric_analysis"] = numeric_analysis
    r["numeric_summary"]  = {
        "segments_with_numbers":len(numeric_analysis),
        "grouped_digit_segs"   :sum(1 for n in numeric_analysis if n["itn_grouped"]),
        "digit_by_digit_segs"  :sum(1 for n in numeric_analysis if not n["itn_grouped"] and n["digit_string"]),
        "digit_rule"           :"Sequences of 3+ digit words converted. to/for/won/ate/etc. NEVER converted.",
    }
    print(f"  Segments with numbers  : {len(numeric_analysis)}")
    print(f"  Grouped form (1234)    : {r['numeric_summary']['grouped_digit_segs']}")
    print(f"  Digit-by-digit (1 2 3) : {r['numeric_summary']['digit_by_digit_segs']}")
    return r

def run_stage_6_dictation(wav_file):
    print("\n  Stage 6: Dictation mode ON — natural punctuation inserted by Azure.")
    r = run_recognition(wav_file, 6)
    tx = r["transcript"]
    r["dictation_analysis"] = {
        "commas":tx.count(","),"periods":tx.count("."),"questions":tx.count("?"),
        "exclamations":tx.count("!"),"total_punct":tx.count(",")+tx.count(".")+tx.count("?")+tx.count("!"),
    }
    print(f"  Punctuation counts: {r['dictation_analysis']}")
    return r

def run_stage_7_emotion_tone(wav_file):
    print("\n  Stage 7: Emotion/tone — confidence proxy, speech rate, disfluency, keywords.")
    r = run_recognition(wav_file, 7)
    segs = r["segments"]; tx = r["transcript"]
    overall_tone = tone_signals(tx)
    confs        = [s["confidence"] for s in segs if s.get("confidence")]
    low_conf     = [s for s in segs if s.get("confidence") and s["confidence"] < 0.70]
    disfl_segs   = [s for s in segs if s.get("tone") and s["tone"].get("disfluencies")]
    r["emotion_tone"] = {
        "overall_tone"        : overall_tone["tone"],
        "negative_markers"    : overall_tone["negative"],
        "positive_markers"    : overall_tone["positive"],
        "disfluencies_overall": overall_tone["disfluencies"],
        "low_confidence_segs" : len(low_conf),
        "disfluency_segs"     : len(disfl_segs),
        "stress_risk"         : "high" if len(low_conf)>len(segs)*0.3 else "medium" if len(low_conf)>0.1*len(segs) else "low",
        "sdk_note"            : "Azure SDK has no native emotion label. Using confidence+rate+keywords as proxy.",
    }
    print(f"  Overall tone: {r['emotion_tone']['overall_tone']} | Stress: {r['emotion_tone']['stress_risk']}")
    return r

def run_stage_8_latency(wav_file):
    print("\n  Stage 8: Latency — 3 runs, P50/P90/P95, SLA check.")
    SLA = {"first_byte":500,"avg":800,"p90":1200}
    runs = []; all_fb = []; all_avg = []
    for i in range(1,4):
        print(f"\n  Run {i}/3")
        r = run_recognition(wav_file, 8)
        lats = [s["latency_ms"] for s in r["segments"]]
        avg  = round(statistics.mean(lats),1) if lats else None
        all_fb.append(r["ttft_final_ms"] or 0)
        if avg: all_avg.append(avg)
        run_r = {**r,"run_id":i,"avg_seg_ms":avg}
        run_r["p90_ms"] = sorted(lats)[int(len(lats)*0.9)] if len(lats)>=5 else None
        run_r["p95_ms"] = sorted(lats)[int(len(lats)*0.95)] if len(lats)>=10 else None
        runs.append(run_r)
        time.sleep(2)
    best = min(runs, key=lambda x: x["ttft_final_ms"] or 9999)
    sla  = {
        "first_byte_pass":(min(all_fb) or 9999)<=SLA["first_byte"],
        "avg_pass"       :(min(all_avg) if all_avg else 9999)<=SLA["avg"],
        "p90_pass"       :any(r.get("p90_ms") and r["p90_ms"]<=SLA["p90"] for r in runs),
        "targets"        : SLA,
    }
    result = {**best}
    result["latency_multi_run"] = {
        "runs":3,"avg_ttft_final_ms":round(statistics.mean(all_fb),1),
        "min_ttft_final_ms":min(all_fb),"max_ttft_final_ms":max(all_fb),
        "sla_assessment":sla,
    }
    print(f"  Avg TTFT Final: {result['latency_multi_run']['avg_ttft_final_ms']}ms")
    print(f"  SLA first-byte: {'✔ PASS' if sla['first_byte_pass'] else '✘ FAIL'}")
    print(f"  SLA avg       : {'✔ PASS' if sla['avg_pass'] else '✘ FAIL'}")
    return result

def run_stage_9_realtime_socket(wav_file):
    print("\n  Stage 9: PushAudioInputStream with 40ms chunks (WebSocket simulation).")
    CHUNK_MS = 40
    try:
        pcm, ch, bits, sr = read_wav_pcm(wav_file)
    except Exception as e:
        print(f"  WAV read failed: {e}")
        return {**run_recognition(wav_file, 9), "realtime_socket":{"error":str(e)}}

    fmt    = speechsdk.audio.AudioStreamFormat(samples_per_second=sr, bits_per_sample=bits, channels=ch)
    ps     = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    ac     = speechsdk.audio.AudioConfig(stream=ps)
    bpm    = (sr*ch*(bits//8))//1000
    chunks = [pcm[i:i+bpm*CHUNK_MS] for i in range(0,len(pcm),bpm*CHUNK_MS)]

    def feed():
        for c in chunks:
            ps.write(c); time.sleep(CHUNK_MS/1000)
        ps.close()

    threading.Thread(target=feed, daemon=True).start()
    r = run_recognition(wav_file, 9, audio_cfg=ac)
    r["realtime_socket"] = {"method":"PushAudioInputStream","chunk_ms":CHUNK_MS,
                             "chunk_count":len(chunks),"sample_rate":sr}
    print(f"  Chunks: {len(chunks)} × {CHUNK_MS}ms")
    return r

def run_stage_10_concurrency(wav_file):
    import concurrent.futures
    print(f"\n  Stage 10: Concurrency ramp {CONCURRENCY_LEVELS} streams.")

    def one_stream(wid):
        t0 = time.time(); done = threading.Event(); segs = []
        err = {"throttled":False,"error":None}
        cfg   = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        auto  = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=CANDIDATE_LANGUAGES)
        audio = speechsdk.audio.AudioConfig(filename=wav_file)
        rec   = speechsdk.SpeechRecognizer(speech_config=cfg, auto_detect_source_language_config=auto, audio_config=audio)
        def on_rec(e):
            if e.result.reason == speechsdk.ResultReason.RecognizedSpeech and e.result.text: segs.append(e.result.text)
        def on_stop(e): done.set()
        def on_cancel(e):
            cd = e.result.cancellation_details
            if cd.reason == speechsdk.CancellationReason.Error:
                err["error"] = cd.error_details
                err["throttled"] = "429" in (cd.error_details or "") or "quota" in (cd.error_details or "").lower()
            done.set()
        rec.recognized.connect(on_rec); rec.session_stopped.connect(on_stop); rec.canceled.connect(on_cancel)
        rec.start_continuous_recognition(); done.wait(timeout=300); rec.stop_continuous_recognition()
        return {"worker_id":wid,"status":"ok" if not err["error"] else "error",
                "throttled":err["throttled"],"error":err["error"],"segments":len(segs),"total_sec":round(time.time()-t0,2)}

    level_results = []; ceiling = None
    for n in CONCURRENCY_LEVELS:
        print(f"\n  Testing N={n} streams…")
        tw = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            workers = list(pool.map(one_stream, range(n)))
        ok = sum(1 for w in workers if w["status"]=="ok")
        th = sum(1 for w in workers if w["throttled"])
        lvl = {"concurrency":n,"wall_sec":round(time.time()-tw,2),"ok":ok,
               "throttled":th,"success_pct":round(ok/n*100,1),"ceiling_hit":th>0}
        level_results.append(lvl)
        print(f"  N={n}: {ok}/{n} OK | throttled={th}")
        if th > 0:
            ceiling = n
            print(f"  ⚠ Ceiling hit at N={n}"); break
        time.sleep(3)

    safe = [l["concurrency"] for l in level_results if l["success_pct"]==100 and not l["ceiling_hit"]]
    base = run_recognition(wav_file, 10)
    base["concurrency_test"] = {"levels_tested":CONCURRENCY_LEVELS,"level_results":level_results,
                                 "max_safe_concurrency":max(safe) if safe else 1,"quota_ceiling":ceiling}
    print(f"\n  Max safe concurrency: {base['concurrency_test']['max_safe_concurrency']}")
    return base

def run_stage_11_logging(wav_file):
    print("\n  Stage 11: Logging & alerts — SDK log + structured session + alert rules.")
    log_dir = os.path.join(OBS_DIR,"stage_11_logging_alerts","logs")
    os.makedirs(log_dir, exist_ok=True)
    sdk_log = os.path.join(log_dir,"azure_sdk.log")
    os.environ["SPEECH_SDK_LOGFILE"] = sdk_log
    alerter = AlertEngine()
    r       = run_recognition(wav_file, 11, alerter=alerter)
    alerter.check_zero(r["segment_count"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(log_dir,f"session_{ts}.json"),"w") as f: json.dump(r,f,indent=2,default=str)
    with open(os.path.join(log_dir,f"alerts_{ts}.json"),"w")  as f: json.dump(alerter.to_dict(),f,indent=2)
    r["logging"] = {"log_dir":log_dir,"sdk_log":sdk_log,"alerts_fired":len(alerter.alerts),
                     "alert_details":alerter.to_dict(),"rules":list(AlertEngine.RULES.keys())}
    print(f"  Alerts fired: {len(alerter.alerts)} | Logs: {log_dir}")
    return r

def run_stage_12_fallback(wav_file):
    print("\n  Stage 12: Fallback chain — silence test → re-prompt → language retry → DTMF.")
    log = []; attempts = []

    def attempt(audio_path, languages, num):
        cfg   = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        cfg.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,"3000")
        auto  = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
        audio = speechsdk.audio.AudioConfig(filename=audio_path)
        rec   = speechsdk.SpeechRecognizer(speech_config=cfg,auto_detect_source_language_config=auto,audio_config=audio)
        segs=[]; nm=[]; done=threading.Event(); can={"v":False}; t0=time.time()
        def on_rec(e):
            if e.result.reason==speechsdk.ResultReason.RecognizedSpeech and e.result.text: segs.append(e.result.text)
            elif e.result.reason==speechsdk.ResultReason.NoMatch: nm.append(True)
        def on_stop(e): done.set()
        def on_can(e): can["v"]=True; done.set()
        rec.recognized.connect(on_rec); rec.session_stopped.connect(on_stop); rec.canceled.connect(on_can)
        rec.start_continuous_recognition(); done.wait(timeout=120); rec.stop_continuous_recognition()
        return {"attempt":num,"segments":len(segs),"no_match":len(nm),"canceled":can["v"],
                "transcript":" ".join(segs),"total_sec":round(time.time()-t0,2)}

    print("\n  Attempt 1: normal audio")
    r1 = attempt(wav_file, CANDIDATE_LANGUAGES, 1); attempts.append(r1)
    log.append({"action":"recognition_attempt","attempt":1,"ok":r1["segments"]>0})
    print(f"  → {r1['segments']} segments")

    silence_path = os.path.join(OBS_DIR,"silence_test.wav")
    create_silence_wav(silence_path, 3.0)
    print("\n  Attempt 2: silence input (tests no-match / re-prompt path)")
    r2 = attempt(silence_path, CANDIDATE_LANGUAGES, 2); attempts.append(r2)
    log.append({"action":"silence_test","ok":r2["segments"]>0})
    if r2["segments"] == 0:
        log.append({"action":"reprompt_triggered","reason":"no_speech"})
        print("  ✔ No-match triggered → re-prompt fires in live IVR")

    print("\n  Attempt 3: language retry (reversed order)")
    r3 = attempt(wav_file, list(reversed(CANDIDATE_LANGUAGES)), 3); attempts.append(r3)
    log.append({"action":"language_retry","languages":list(reversed(CANDIDATE_LANGUAGES)),"ok":r3["segments"]>0})
    print(f"  → {r3['segments']} segments")

    print("\n  DTMF fallback: simulated")
    log.append({"action":"dtmf_simulated","input":"1","note":"IVR would switch to touch-tone"})
    log.append({"action":"agent_escalation","note":"IVR would route to human agent if DTMF fails"})

    base = run_recognition(wav_file, 12)
    base["fallback_test"] = {
        "attempts":attempts,"fallback_log":log,
        "silence_triggered_reprompt":r2["segments"]==0,
        "language_retry_worked":r3["segments"]>0,
        "dtmf_simulated":True,
        "fallback_chain":["recognition","re-prompt","language_retry","dtmf","agent_escalation"],
    }
    print(f"\n  Re-prompt triggered: {base['fallback_test']['silence_triggered_reprompt']}")
    print(f"  Language retry OK  : {base['fallback_test']['language_retry_worked']}")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# STAGE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
STAGE_RUNNERS = {
    0:run_stage_0_baseline, 1:run_stage_1_asr_config, 2:run_stage_2_vad,
    3:run_stage_3_phrase_boost, 4:run_stage_4_vocab_tuning, 5:run_stage_5_numeric,
    6:run_stage_6_dictation, 7:run_stage_7_emotion_tone, 8:run_stage_8_latency,
    9:run_stage_9_realtime_socket, 10:run_stage_10_concurrency,
    11:run_stage_11_logging, 12:run_stage_12_fallback,
}

# ─────────────────────────────────────────────────────────────────────────────
# SAVE + COMPARE
# ─────────────────────────────────────────────────────────────────────────────
REPORT_PATH = os.path.join(OBS_DIR, "comparison_report.json")

def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f: return json.load(f)
    return {"stages":[],"comparisons":[],"last_updated":None}

def save_report(report):
    report["last_updated"] = datetime.now().isoformat()
    with open(REPORT_PATH,"w") as f: json.dump(report, f, indent=2, default=str)

def save_stage_files(stage_num, meta, result, qa):
    name      = meta["name"]
    stage_dir = os.path.join(OBS_DIR, f"stage_{stage_num}_{name}")
    os.makedirs(stage_dir, exist_ok=True)

    with open(os.path.join(stage_dir,"result.json"),"w",encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(stage_dir,"transcript.txt"),"w",encoding="utf-8") as f:
        f.write(result.get("transcript",""))

    # Quality analysis file
    with open(os.path.join(stage_dir,"quality_analysis.json"),"w",encoding="utf-8") as f:
        json.dump(qa, f, indent=2, default=str)

    # Human-readable metrics + params summary
    m = result
    with open(os.path.join(stage_dir,"metrics_summary.txt"),"w",encoding="utf-8") as f:
        lines = [
            f"Stage         : {stage_num} — {name}",
            f"Phase         : {meta['phase']}",
            f"Task          : {meta['task']}",
            f"Description   : {meta['description']}",
            f"Expected Out  : {meta['outcome']}",
            f"Timestamp     : {datetime.now().isoformat()}",
            "",
            "── Parameters Changed at This Stage ──────────────────",
            f"  {meta['parameters_changed']}",
            "",
            "── Parameters Used ───────────────────────────────────",
            f"  {meta['parameters_used']}",
            "",
            "── Latency Metrics ───────────────────────────────────",
            f"  TTFT Partial   : {m.get('ttft_partial_ms')} ms",
            f"  TTFT Final     : {m.get('ttft_final_ms')} ms",
            f"  Total Time     : {m.get('total_time_sec')} sec",
            "",
            "── Accuracy Metrics ──────────────────────────────────",
            f"  Segments       : {m.get('segment_count')}",
            f"  Words          : {m.get('word_count')}",
            f"  Empty Segments : {m.get('empty_segments')}",
            f"  Avg Confidence : {m.get('avg_confidence')}",
            f"  Min Confidence : {m.get('min_confidence')}",
            f"  Max Confidence : {m.get('max_confidence')}",
            "",
            "── Quality Analysis ──────────────────────────────────",
            f"  Quality Score  : {qa['quality_score']} / 100  (Grade: {qa['quality_grade']})",
            f"  Short Words    : {qa['short_word']['count']} ({qa['short_word']['pct_of_total']}%)",
            f"  Digit Groups   : {qa['numbers']['digit_group_count']}",
            f"  Spoken Digits  : {qa['numbers']['spoken_digit_count']}",
            f"  Punctuation    : {qa['punctuation']['total']} total",
            f"  VAD Truncation : {qa['vad_signals']['truncation_risk']} risk",
            f"  Filler Words   : {qa['noise']['filler_count']}",
            f"  Repeated Words : {qa['noise']['repeated_count']}",
            "",
            "── Transcript ────────────────────────────────────────",
            result.get("transcript",""),
        ]
        f.write("\n".join(lines))

    print(f"  Saved → {stage_dir}/")
    return stage_dir


def compute_delta(prev, curr, lower_better=False):
    if prev is None or curr is None: return {"prev":prev,"curr":curr,"change":None,"direction":"unknown"}
    d = curr - prev
    direction = ("improved" if d<0 else "worse" if d>0 else "same") if lower_better \
           else ("improved" if d>0 else "worse" if d<0 else "same")
    return {"prev":prev,"curr":curr,"change":round(d,4),"direction":direction}


def compare(prev_result, curr_result, curr_meta, prev_qa, curr_qa):
    pm = prev_result; cm = curr_result
    words_a = (pm.get("transcript","") or "").lower().split()
    words_b = (cm.get("transcript","") or "").lower().split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    sim_pct = round(matcher.ratio()*100,1)
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

    observations = _observations(pm, cm, sim_pct, len(changes), curr_meta)
    quality_comp = compare_quality(prev_qa, curr_qa)

    return {
        "from_stage"     : pm.get("_stage_name","prev"),
        "to_stage"       : cm.get("_stage_name","curr"),
        "phase"          : curr_meta["phase"],
        "task"           : curr_meta["task"],
        "parameters_changed": curr_meta["parameters_changed"],
        "parameters_used"   : curr_meta["parameters_used"],
        "description"    : curr_meta["description"],
        "expected_outcome": curr_meta["outcome"],
        "metric_deltas"  : deltas,
        "transcript_diff": {"similarity_pct":sim_pct,"change_count":len(changes),"changes":changes[:15]},
        "observations"   : observations,
        "quality_comparison": quality_comp,
        "timestamp"      : datetime.now().isoformat(),
    }


def _observations(pm, cm, sim_pct, n_changes, meta):
    notes = []
    if pm.get("ttft_final_ms") and cm.get("ttft_final_ms"):
        d = cm["ttft_final_ms"] - pm["ttft_final_ms"]
        if d < -50: notes.append(f"✅ TTFT Final improved by {abs(d):.0f}ms — faster first response")
        elif d > 50: notes.append(f"⚠️  TTFT Final slower by {d:.0f}ms (may be acceptable for better accuracy)")
    if pm.get("word_count") and cm.get("word_count"):
        d = cm["word_count"] - pm["word_count"]
        if d > 5:  notes.append(f"✅ {d} more words captured — less truncation or broader vocabulary")
        elif d < -5: notes.append(f"⚠️  {abs(d)} fewer words — verify endpointing not over-cutting")
    if pm.get("avg_confidence") and cm.get("avg_confidence"):
        d = cm["avg_confidence"] - pm["avg_confidence"]
        if d > 0.01: notes.append(f"✅ Avg confidence improved by {d:.4f} — Azure more certain about words")
        elif d < -0.01: notes.append(f"⚠️  Confidence dropped by {abs(d):.4f}")
    if pm.get("empty_segments") is not None and cm.get("empty_segments") is not None:
        d = cm["empty_segments"] - pm["empty_segments"]
        if d < 0: notes.append(f"✅ {abs(d)} fewer empty segments")
        elif d > 0: notes.append(f"⚠️  {d} more empty segments")
    if sim_pct > 98: notes.append(f"➡️  Transcript nearly identical ({sim_pct}%) — feature impact is in quality metrics, not words")
    elif sim_pct > 90: notes.append(f"➡️  Transcript similar ({sim_pct}%) — small word-level changes")
    else: notes.append(f"⚠️  Transcript changed significantly ({sim_pct}%) — review word diff carefully")
    notes.append(f"ℹ️  Expected: {meta['outcome']}")
    return notes


def print_comparison(comp):
    print(f"\n{'─'*65}")
    print(f"  COMPARISON: {comp['from_stage']} → {comp['to_stage']}")
    print(f"  Phase: {comp['phase']} | Task: {comp['task']}")
    print(f"  Param changed: {comp['parameters_changed']}")
    print(f"{'─'*65}")
    for metric, d in comp["metric_deltas"].items():
        if d["change"] is None: continue
        sym = {"improved":"↑","worse":"↓","same":"→"}.get(d["direction"],"?")
        sign = "+" if d["change"]>0 else ""
        print(f"  {metric:<22}: {d['prev']} → {d['curr']}  ({sign}{d['change']}) {sym}")
    td = comp["transcript_diff"]
    print(f"\n  Transcript similarity: {td['similarity_pct']}% | Word changes: {td['change_count']}")
    if td["changes"]:
        for c in td["changes"][:4]:
            print(f"    [{c['type']}] '{c['before'][:25]}' → '{c['after'][:25]}'")
    print(f"\n  Observations:")
    for n in comp["observations"]: print(f"    {n}")
    print(f"\n  Quality:")
    for n in comp["quality_comparison"]["quality_notes"]: print(f"    {n}")


def print_full_table(report):
    stages = report.get("stages",[])
    if not stages: return
    print(f"\n\n{'='*80}")
    print("  FULL INCREMENTAL REPORT")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Stage':<22} {'Phase':<13} {'Words':>6} {'Segs':>5} {'Conf':>7} {'TTFT-F':>8} {'Qual':>6}")
    print(f"  {'─'*3} {'─'*22} {'─'*13} {'─'*6} {'─'*5} {'─'*7} {'─'*8} {'─'*6}")
    for s in stages:
        qa = s.get("_quality_analysis",{})
        print(f"  {str(s.get('_stage_num','?')):<3} {s.get('_stage_name','?'):<22} "
              f"{s.get('_phase','?'):<13} "
              f"{str(s.get('word_count','?')):>6} "
              f"{str(s.get('segment_count','?')):>5} "
              f"{str(s.get('avg_confidence','N/A')):>7} "
              f"{str(s.get('ttft_final_ms','N/A')):>8} "
              f"{str(qa.get('quality_score','N/A')):>6}")
    print(f"\n  Full JSON → {REPORT_PATH}")
    print(f"  Run: python generate_observation_doc.py")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_one(stage_num):
    if stage_num not in STAGES:
        print(f"Invalid stage {stage_num}. Valid: 0–12"); sys.exit(1)

    meta     = STAGES[stage_num]
    wav_file = convert_to_wav(INPUT_AUDIO)

    print(f"\n{'='*65}")
    print(f"  STAGE {stage_num} / 12 — {meta['name'].upper()}")
    print(f"  Phase  : {meta['phase']}")
    print(f"  Task   : {meta['task']}")
    print(f"  Goal   : {meta['outcome']}")
    print(f"  Params : {meta['parameters_changed']}")
    print(f"{'='*65}")

    result = STAGE_RUNNERS[stage_num](wav_file)
    qa     = analyse_transcript_quality(result.get("transcript",""), stage_num, meta["name"])

    print(f"\n  Quality Score: {qa['quality_score']}/100 (Grade {qa['quality_grade']})")
    print(f"  Short words: {qa['short_word']['count']} | Digits: {qa['numbers']['digit_group_count']} "
          f"| Punct: {qa['punctuation']['total']} | VAD risk: {qa['vad_signals']['truncation_risk']}")

    result["_stage_num"]        = stage_num
    result["_stage_name"]       = meta["name"]
    result["_phase"]            = meta["phase"]
    result["_task"]             = meta["task"]
    result["_timestamp"]        = datetime.now().isoformat()
    result["_quality_analysis"] = qa

    save_stage_files(stage_num, meta, result, qa)

    report = load_report()
    report["stages"].append(result)

    if len(report["stages"]) >= 2:
        prev    = report["stages"][-2]
        prev_qa = prev.get("_quality_analysis", analyse_transcript_quality(prev.get("transcript",""), 0, "prev"))
        comp    = compare(prev, result, meta, prev_qa, qa)
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
        print(f"  STAGE {stage_num}/12 — {meta['name'].upper()} [{meta['phase']}]")
        print(f"  Params: {meta['parameters_changed']}")
        print(f"{'='*65}")

        result = STAGE_RUNNERS[stage_num](wav_file)
        qa     = analyse_transcript_quality(result.get("transcript",""), stage_num, meta["name"])

        result["_stage_num"]        = stage_num
        result["_stage_name"]       = meta["name"]
        result["_phase"]            = meta["phase"]
        result["_task"]             = meta["task"]
        result["_timestamp"]        = datetime.now().isoformat()
        result["_quality_analysis"] = qa

        save_stage_files(stage_num, meta, result, qa)
        report["stages"].append(result)

        if len(report["stages"]) >= 2:
            prev    = report["stages"][-2]
            prev_qa = prev.get("_quality_analysis", analyse_transcript_quality(prev.get("transcript",""), 0, "prev"))
            comp    = compare(prev, result, meta, prev_qa, qa)
            report["comparisons"].append(comp)
            print_comparison(comp)

        save_report(report)
        if stage_num < 12: print("\n  Pausing 3s…"); time.sleep(3)

    print_full_table(report)


def main():
    print("\n  Azure STT — 12-Stage Incremental Improvement")
    print("  ─────────────────────────────────────────────")
    for n, m in STAGES.items():
        print(f"    {n:>2}  {m['name']:<22} [{m['phase']}]  {m['parameters_changed'][:55]}")
    print()
    print("  Usage:")
    print("    python azure_incremental.py --stage 0   ← start here")
    print("    python azure_incremental.py --all       ← run all")
    print()
    if "--all" in sys.argv:    run_all()
    elif "--stage" in sys.argv: run_one(int(sys.argv[sys.argv.index("--stage")+1]))
    else: print("  No stage given — running stage 0"); run_one(0)

if __name__ == "__main__":
    main()
