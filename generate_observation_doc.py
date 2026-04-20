"""
OBSERVATION DOC GENERATOR
=========================
Reads observations/comparison_report.json and generates a formatted
documentation report.

Enhanced version:
- Supports old and new report formats
- Shows stage parameter changes / config changes
- Shows transcript quality metrics
- Shows comparison metrics and observations
- Better aligned to the modified azure_incremental.py script

Run:
    python generate_observation_doc.py

Output:
    observations/OBSERVATION_REPORT.md
    observations/OBSERVATION_REPORT.txt
"""

import os
import json
from datetime import datetime

REPORT_PATH = os.path.join("observations", "comparison_report.json")
MD_OUT = os.path.join("observations", "OBSERVATION_REPORT.md")
TXT_OUT = os.path.join("observations", "OBSERVATION_REPORT.txt")


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_report() -> dict:
    if not os.path.exists(REPORT_PATH):
        print(f"No report found at {REPORT_PATH}")
        print("Run azure_incremental.py first.")
        raise SystemExit(1)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FIELD RESOLVERS
# ─────────────────────────────────────────────────────────────────────────────

def get_stage_num(stage: dict, fallback_index: int) -> str:
    val = stage.get("_stage_num")
    if val is not None:
        return str(val)
    return str(fallback_index)


def get_stage_name(stage: dict, fallback_index: int) -> str:
    if stage.get("_stage_name"):
        return stage["_stage_name"]
    if stage.get("stage"):
        return stage["stage"]
    return f"stage_{fallback_index}"


def get_phase(stage: dict) -> str:
    return stage.get("_phase", "")


def get_task(stage: dict) -> str:
    return stage.get("_task", "")


def get_description(stage: dict) -> str:
    if stage.get("stage_description"):
        return stage["stage_description"]
    if stage.get("_task"):
        phase_str = f" [{stage['_phase']}]" if stage.get("_phase") else ""
        return f"{stage['_task']}{phase_str}"
    return f"Detected language: {stage.get('detected_language', 'Unknown')}"


def get_metrics(stage: dict) -> dict:
    if "metrics" in stage and isinstance(stage["metrics"], dict):
        return stage["metrics"]
    return stage


def get_comp_key(stage: dict, fallback_index: int) -> str:
    return get_stage_name(stage, fallback_index)


def fmt(val, suffix="") -> str:
    return "N/A" if val is None else f"{val}{suffix}"


def direction_symbol(d: str) -> str:
    return {
        "improved": "✅",
        "worse": "⚠️",
        "same": "➡️",
        "unknown": "❓",
    }.get(d, "❓")


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER CHANGE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def get_stage_parameter_changes(stage: dict) -> list[str]:
    """
    Extracts stage-specific changes/configs from result JSON.
    This is the main enhancement requested.
    """
    out = []

    stage_num = stage.get("_stage_num")
    stage_name = stage.get("_stage_name", "")

    # Common fields
    if stage.get("detected_language") is not None:
        out.append(f"Detected language result: {stage.get('detected_language')}")

    if stage_num == 0:
        out.append("Baseline run — no improvement features enabled")
        out.append("Default Azure config used for reference measurement")

    if "asr_config_notes" in stage:
        cfg = stage["asr_config_notes"]
        out.append(f"Language candidates locked to: {cfg.get('language_locked')}")
        out.append(f"Audio format fixed to: {cfg.get('audio_format')}")
        out.append(f"Auto-detect locked: {cfg.get('auto_detect_locked')}")
        out.append(f"Open-ended detection disabled: {cfg.get('open_ended_detect')}")

    if "vad_config" in stage:
        cfg = stage["vad_config"]
        out.append(f"End silence timeout set to: {cfg.get('end_silence_ms')} ms")
        out.append(f"Initial silence timeout set to: {cfg.get('init_silence_ms')} ms")
        out.append(f"Segmentation silence timeout set to: {cfg.get('seg_silence_ms')} ms")
        if cfg.get("note"):
            out.append(f"VAD note: {cfg.get('note')}")

    if "phrase_boost" in stage:
        pb = stage["phrase_boost"]
        out.append(f"Phrase boosting enabled")
        out.append(f"Total boosted phrases: {pb.get('total_phrases')}")
        out.append(f"Phrase hits found in transcript: {pb.get('hit_count')}")
        hits = pb.get("hits_in_transcript") or []
        if hits:
            out.append(f"Matched boosted phrases: {', '.join(hits[:10])}")

    if "vocab_tuning" in stage:
        vt = stage["vocab_tuning"]
        out.append("Transcript-based vocabulary tuning enabled")
        out.append(f"Mined vocabulary term count: {vt.get('mined_term_count')}")
        terms = vt.get("mined_terms") or []
        if terms:
            out.append(f"Top mined terms: {', '.join(terms[:10])}")

    if "numeric_analysis" in stage:
        out.append("Detailed numeric analysis enabled")
        q = stage.get("quality_scores", {}).get("numeric_quality", {})
        out.append(f"Numeric context detected: {q.get('numeric_context_detected')}")
        out.append(f"Grouped numeric candidates: {q.get('grouped_candidate_count')}")
        amb = q.get("ambiguous_unsafe_cases") or []
        out.append(f"Ambiguous numeric-risk cases: {len(amb)}")

    if "dictation_analysis" in stage:
        da = stage["dictation_analysis"]
        out.append("Dictation mode enabled")
        out.append(f"Commas inserted: {da.get('commas')}")
        out.append(f"Periods inserted: {da.get('periods')}")
        out.append(f"Questions inserted: {da.get('questions')}")
        out.append(f"Total punctuation count: {da.get('total_punct')}")

    if "emotion_tone" in stage:
        et = stage["emotion_tone"]
        out.append("Emotion/tone proxy analysis enabled")
        out.append(f"Overall tone proxy: {et.get('overall_tone')}")
        out.append(f"Stress risk proxy: {et.get('stress_risk')}")
        out.append(f"Low-confidence segments: {et.get('low_confidence_segs')}")
        out.append(f"Disfluency segments: {et.get('disfluency_segs')}")

    if "latency_multi_run" in stage:
        lt = stage["latency_multi_run"]
        out.append("Multi-run latency testing enabled")
        out.append(f"Latency runs performed: {lt.get('runs')}")
        out.append(f"Average TTFT final: {lt.get('avg_ttft_final_ms')} ms")
        out.append(f"Minimum TTFT final: {lt.get('min_ttft_final_ms')} ms")
        out.append(f"Maximum TTFT final: {lt.get('max_ttft_final_ms')} ms")
        sla = lt.get("sla_assessment", {})
        if sla:
            out.append(f"SLA first-byte pass: {sla.get('first_byte_pass')}")
            out.append(f"SLA average-latency pass: {sla.get('avg_pass')}")
            out.append(f"SLA P90 pass: {sla.get('p90_pass')}")

    if "realtime_socket" in stage:
        rs = stage["realtime_socket"]
        out.append(f"Streaming mode enabled via: {rs.get('method')}")
        out.append(f"Chunk size used: {rs.get('chunk_ms')} ms")
        out.append(f"Chunk count: {rs.get('chunk_count')}")
        out.append(f"Sample rate: {rs.get('sample_rate')}")

    if "concurrency_test" in stage:
        ct = stage["concurrency_test"]
        out.append("Concurrency/load testing enabled")
        out.append(f"Levels tested: {ct.get('levels_tested')}")
        out.append(f"Max safe concurrency: {ct.get('max_safe_concurrency')}")
        out.append(f"Quota ceiling detected at: {ct.get('quota_ceiling')}")

    if "logging" in stage:
        lg = stage["logging"]
        out.append("Structured logging and alert rules enabled")
        out.append(f"Alerts fired: {lg.get('alerts_fired')}")
        out.append(f"SDK log path: {lg.get('sdk_log_path')}")
        out.append(f"Rules active: {lg.get('rules_active')}")

    if "fallback_test" in stage:
        fb = stage["fallback_test"]
        out.append("Fallback validation enabled")
        out.append(f"Fallback chain: {fb.get('fallback_chain')}")
        out.append(f"Silence triggered re-prompt: {fb.get('silence_triggered_reprompt')}")
        out.append(f"Language retry worked: {fb.get('language_retry_worked')}")
        out.append(f"DTMF simulated: {fb.get('dtmf_simulated')}")

    # Quality score summary
    q = stage.get("quality_scores", {})
    if q:
        out.append("Transcript quality scoring enabled")
        out.append(f"Overall quality score: {stage.get('overall_quality_score')}")
        out.append(f"Punctuation quality score: {q.get('punctuation_quality', {}).get('score')}")
        out.append(f"Numeric quality score: {q.get('numeric_quality', {}).get('score')}")
        out.append(f"Domain quality score: {q.get('domain_quality', {}).get('score')}")
        out.append(f"VAD quality score: {q.get('vad_quality', {}).get('score')}")
        out.append(f"Short-word quality score: {q.get('short_word_quality', {}).get('score')}")
        out.append(f"Readability quality score: {q.get('readability_quality', {}).get('score')}")

    if not out:
        out.append(f"No explicit stage parameter changes captured for `{stage_name}`")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown(report: dict) -> str:
    stages = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines = []

    lines += [
        "# Azure Speech-to-Text — Incremental Improvement Observations",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Stages run:** {len(stages)}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This document records observations at each stage of the Azure STT",
        "incremental improvement process. Each stage adds one feature or test",
        "configuration and compares the resulting metrics, quality scores, and transcript.",
        "",
        "Baseline is used as a reference point only and not as ground truth.",
        "",
        "---",
        "",
    ]

    for i, stage in enumerate(stages):
        m = get_metrics(stage)
        feat = stage.get("features", {})
        stage_num = get_stage_num(stage, i)
        stage_name = get_stage_name(stage, i)
        phase = get_phase(stage)
        task = get_task(stage)
        description = get_description(stage)
        comp_key = get_comp_key(stage, i)
        param_changes = get_stage_parameter_changes(stage)

        comp = next((c for c in comparisons if c.get("to_stage") == comp_key), None)

        lines
