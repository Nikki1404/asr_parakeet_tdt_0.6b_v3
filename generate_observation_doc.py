"""
OBSERVATION DOC GENERATOR
=========================
Reads observations/comparison_report.json and generates:

    observations/OBSERVATION_REPORT.md
    observations/OBSERVATION_REPORT.txt

This version supports:
- old simple report format
- current 12-stage incremental Azure script
- quality-score aware reporting
- stage parameter / config change reporting

Run:
    python generate_observation_doc.py
"""

import os
import json
from datetime import datetime

REPORT_PATH = os.path.join("observations", "comparison_report.json")
MD_OUT = os.path.join("observations", "OBSERVATION_REPORT.md")
TXT_OUT = os.path.join("observations", "OBSERVATION_REPORT.txt")


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT STAGE PARAMETER MAP
# These describe what was intentionally changed/tested at each stage.
# ─────────────────────────────────────────────────────────────────────────────
STAGE_PARAMETER_MAP = {
    "baseline": {
        "azure_parameters_tested": [
            "SpeechConfig(subscription, region)",
            "OutputFormat = Detailed",
            "AutoDetectSourceLanguageConfig(en-US, es-US)",
            "WAV PCM 16kHz mono 16-bit input",
        ],
        "changes_made": [
            "No optimization feature enabled",
            "Used as baseline reference only",
        ],
    },
    "asr_config": {
        "azure_parameters_tested": [
            "AutoDetectSourceLanguageConfig(languages=['en-US','es-US'])",
            "Fixed WAV PCM 16kHz mono 16-bit audio format",
            "SpeechConfig output_format = Detailed",
        ],
        "changes_made": [
            "Locked candidate locales",
            "Standardized input audio format",
            "Avoided open-ended language detection",
        ],
    },
    "vad_tuning": {
        "azure_parameters_tested": [
            "SpeechServiceConnection_EndSilenceTimeoutMs = 1500",
            "SpeechServiceConnection_InitialSilenceTimeoutMs = 5000",
            "Speech_SegmentationSilenceTimeoutMs = 800",
        ],
        "changes_made": [
            "Increased end silence timeout",
            "Tested segmentation and initial silence thresholds",
            "Evaluated truncation / false cutoff behavior",
        ],
    },
    "phrase_boost": {
        "azure_parameters_tested": [
            "PhraseListGrammar.from_recognizer(...)",
            "Added numeric phrases",
            "Added finance / domain phrases",
        ],
        "changes_made": [
            "Boosted digits and domain vocabulary",
            "Tested phrase hit improvement in transcript",
        ],
    },
    "vocab_tuning": {
        "azure_parameters_tested": [
            "PhraseListGrammar with mined transcript terms",
        ],
        "changes_made": [
            "Extracted repeated useful words from prior transcripts",
            "Added mined vocabulary as boost phrases",
        ],
    },
    "numeric_handling": {
        "azure_parameters_tested": [
            "Detailed JSON parsing from SpeechServiceResponse_JsonResult",
            "ITN / Lexical / Display field analysis",
        ],
        "changes_made": [
            "Evaluated number rendering quality",
            "Used context-aware digit analysis",
            "Prevented blind conversions like to -> 2",
        ],
    },
    "dictation_mode": {
        "azure_parameters_tested": [
            "SpeechConfig.enable_dictation()",
        ],
        "changes_made": [
            "Enabled dictation mode",
            "Evaluated punctuation and readability improvement",
        ],
    },
    "emotion_tone": {
        "azure_parameters_tested": [
            "Confidence from NBest JSON",
            "Segment word rate estimation",
            "Tone/disfluency keyword proxy analysis",
        ],
        "changes_made": [
            "No transcript rewriting",
            "Quality proxy from confidence/rate/disfluency markers",
        ],
    },
    "latency_testing": {
        "azure_parameters_tested": [
            "3 repeated recognition runs",
            "TTFT Partial",
            "TTFT Final",
            "avg / p90 / p95 latency estimation",
        ],
        "changes_made": [
            "Measured latency stability across runs",
            "Checked SLA-style thresholds",
        ],
    },
    "realtime_socket": {
        "azure_parameters_tested": [
            "PushAudioInputStream",
            "AudioStreamFormat(samples_per_second, bits_per_sample, channels)",
            "40 ms chunk streaming",
        ],
        "changes_made": [
            "Simulated real-time streaming ingestion",
            "Compared with file-based recognition",
        ],
    },
    "concurrency": {
        "azure_parameters_tested": [
            "Parallel SpeechRecognizer sessions",
            "Concurrency levels [1, 3, 5, 10]",
            "Throttle/quota detection",
        ],
        "changes_made": [
            "Measured multi-stream stability",
            "Checked probable concurrency ceiling",
        ],
    },
    "logging_alerts": {
        "azure_parameters_tested": [
            "SPEECH_SDK_LOGFILE",
            "Structured session JSON logging",
            "Alert rules for latency/confidence/cancel/error",
        ],
        "changes_made": [
            "Enabled diagnostic logging",
            "Generated alert artifacts",
        ],
    },
    "fallback": {
        "azure_parameters_tested": [
            "Reduced InitialSilenceTimeoutMs = 3000 for fallback test",
            "Language retry with reversed candidate order",
            "Silence file no-speech path",
        ],
        "changes_made": [
            "Simulated re-prompt flow",
            "Simulated language retry",
            "Simulated DTMF / agent escalation fallback",
        ],
    },
}


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
    return str(val) if val is not None else str(fallback_index)


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


def get_quality(stage: dict) -> dict:
    return stage.get("quality_scores", {}) if isinstance(stage.get("quality_scores"), dict) else {}


def get_stage_parameters(stage: dict, stage_name: str) -> dict:
    """
    Returns parameter/config summary for a stage.
    First uses saved stage data if present, then falls back to default map.
    """
    out = {
        "azure_parameters_tested": [],
        "changes_made": [],
    }

    defaults = STAGE_PARAMETER_MAP.get(stage_name, {})
    out["azure_parameters_tested"].extend(defaults.get("azure_parameters_tested", []))
    out["changes_made"].extend(defaults.get("changes_made", []))

    # pull from result payload if present
    if stage.get("asr_config_notes"):
        for k, v in stage["asr_config_notes"].items():
            out["changes_made"].append(f"{k} = {v}")

    if stage.get("vad_config"):
        for k, v in stage["vad_config"].items():
            out["changes_made"].append(f"{k} = {v}")

    if stage.get("phrase_boost"):
        pb = stage["phrase_boost"]
        out["changes_made"].append(
            f"Phrase boosting active: total_phrases={pb.get('total_phrases')}, hits={pb.get('hit_count')}"
        )

    if stage.get("vocab_tuning"):
        vt = stage["vocab_tuning"]
        out["changes_made"].append(
            f"Mined vocabulary terms added: {vt.get('mined_term_count')}"
        )

    if stage.get("dictation_analysis"):
        out["changes_made"].append(
            f"Dictation punctuation counts: {stage['dictation_analysis']}"
        )

    if stage.get("realtime_socket"):
        rs = stage["realtime_socket"]
        out["changes_made"].append(
            f"Push stream config: chunk_ms={rs.get('chunk_ms')}, chunk_count={rs.get('chunk_count')}, sample_rate={rs.get('sample_rate')}"
        )

    if stage.get("concurrency_test"):
        ct = stage["concurrency_test"]
        out["changes_made"].append(
            f"Concurrency tested: levels={ct.get('levels_tested')}, max_safe={ct.get('max_safe_concurrency')}, ceiling={ct.get('quota_ceiling')}"
        )

    if stage.get("logging"):
        lg = stage["logging"]
        out["changes_made"].append(
            f"Logging enabled: alerts_fired={lg.get('alerts_fired')}, sdk_log={lg.get('sdk_log_path')}"
        )

    if stage.get("fallback_test"):
        ft = stage["fallback_test"]
        out["changes_made"].append(
            f"Fallback chain tested: reprompt={ft.get('silence_triggered_reprompt')}, lang_retry={ft.get('language_retry_worked')}, dtmf={ft.get('dtmf_simulated')}"
        )

    # dedupe preserving order
    out["azure_parameters_tested"] = list(dict.fromkeys(out["azure_parameters_tested"]))
    out["changes_made"] = list(dict.fromkeys(out["changes_made"]))
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
        "incremental improvement process. Each stage adds one feature or configuration",
        "on top of the previous and compares resulting metrics, transcript quality,",
        "and stage-specific behavior.",
        "",
        "**Important interpretation note:** baseline is treated as a reference point,",
        "not as ground truth. Quality improvements are evaluated independently using",
        "punctuation, numeric handling, domain phrase quality, VAD quality, short-word",
        "risk, readability, and overall quality score.",
        "",
        "---",
        "",
    ]

    for i, stage in enumerate(stages):
        m = get_metrics(stage)
        q = get_quality(stage)
        stage_num = get_stage_num(stage, i)
        stage_name = get_stage_name(stage, i)
        phase = get_phase(stage)
        task = get_task(stage)
        description = get_description(stage)
        comp_key = get_comp_key(stage, i)
        params = get_stage_parameters(stage, stage_name)

        comp = next((c for c in comparisons if c.get("to_stage") == comp_key), None)

        lines += [f"## Stage {stage_num}: `{stage_name}`", ""]

        if phase or task:
            parts = []
            if phase:
                parts.append(f"**Phase:** {phase}")
            if task:
                parts.append(f"**Task:** {task}")
            lines.append("  &nbsp;|&nbsp;  ".join(parts))
            lines.append("")

        lines += [f"**What was added:** {description}", ""]

        lines += ["### Parameters / configuration tested", ""]
        if params["azure_parameters_tested"]:
            lines.append("**Azure / SDK parameters**")
            lines.append("")
            for p in params["azure_parameters_tested"]:
                lines.append(f"- `{p}`")
            lines.append("")
        if params["changes_made"]:
            lines.append("**Changes applied / tested**")
            lines.append("")
            for c in params["changes_made"]:
                lines.append(f"- {c}")
            lines.append("")
        if not params["azure_parameters_tested"] and not params["changes_made"]:
            lines.append("- No stage parameter metadata available")
            lines.append("")

        lines += [
            "### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Stage Number | **{stage_num}** |",
            f"| Stage Name | **`{stage_name}`** |",
            f"| Phase | {fmt(phase or None)} |",
            f"| Task | {fmt(task or None)} |",
            f"| Detected Language | {fmt(stage.get('detected_language'))} |",
            f"| TTFT Partial | {fmt(m.get('ttft_partial_ms'), ' ms')} |",
            f"| TTFT Final | {fmt(m.get('ttft_final_ms'), ' ms')} |",
            f"| Total Time | {fmt(m.get('total_time_sec'), ' sec')} |",
            f"| Segments | {fmt(m.get('segment_count'))} |",
            f"| Words | {fmt(m.get('word_count'))} |",
            f"| Empty Segments | {fmt(m.get('empty_segments'))} |",
            f"| Avg Confidence | {fmt(m.get('avg_confidence'))} |",
            f"| Min Confidence | {fmt(m.get('min_confidence'))} |",
            f"| Max Confidence | {fmt(m.get('max_confidence'))} |",
            f"| Partial Count | {fmt(m.get('partial_count'))} |",
            "",
        ]

        if q:
            lines += [
                "### Quality scoring",
                "",
                "| Quality Metric | Score |",
                "|---------------|-------|",
                f"| Overall Quality | {fmt(stage.get('overall_quality_score'))} |",
                f"| Punctuation Quality | {fmt(q.get('punctuation_quality', {}).get('score'))} |",
                f"| Numeric Quality | {fmt(q.get('numeric_quality', {}).get('score'))} |",
                f"| Domain Phrase Quality | {fmt(q.get('domain_quality', {}).get('score'))} |",
                f"| VAD Quality | {fmt(q.get('vad_quality', {}).get('score'))} |",
                f"| Short Word Quality | {fmt(q.get('short_word_quality', {}).get('score'))} |",
                f"| Readability Quality | {fmt(q.get('readability_quality', {}).get('score'))} |",
                "",
            ]

        transcript = stage.get("transcript", "(empty)")
        lines += [
            "### Transcript",
            "",
            "```",
            transcript[:800],
            "```" if len(transcript) <= 800 else "...(truncated)\n```",
            "",
        ]

        if q:
            lines += ["### Quality details", ""]
            pq = q.get("punctuation_quality", {})
            nq = q.get("numeric_quality", {})
            dq = q.get("domain_quality", {})
            vq = q.get("vad_quality", {})
            swq = q.get("short_word_quality", {})
            rq = q.get("readability_quality", {})

            lines += [
                f"- **Punctuation notes:** {pq.get('notes', [])}",
                f"- **Numeric notes:** {nq.get('notes', [])}",
                f"- **Numeric context detected:** {nq.get('numeric_context_detected')}",
                f"- **Spoken digit sequences found:** {len(nq.get('spoken_digit_sequences', []))}",
                f"- **Domain phrase hits:** {dq.get('hits', [])}",
                f"- **VAD notes:** {vq.get('notes', [])}",
                f"- **Short-word ambiguous count:** {swq.get('ambiguous_count')}",
                f"- **Readability notes:** {rq.get('notes', [])}",
                "",
            ]

        if comp:
            from_stage = comp.get("from_stage", "previous")
            to_stage = comp.get("to_stage", stage_name)

            lines += [
                f"### Change vs Previous Stage (`{from_stage}` → `{to_stage}`)",
                "",
                "| Metric | Before | After | Change | Signal |",
                "|--------|--------|-------|--------|--------|",
            ]

            for metric, d in comp.get("metric_deltas", {}).items():
                if d.get("change") is None:
                    continue
                sign = "+" if d["change"] > 0 else ""
                sym = direction_symbol(d.get("direction"))
                lines.append(
                    f"| {metric} | {fmt(d.get('prev'))} | {fmt(d.get('curr'))} | {sign}{d.get('change')} | {sym} {d.get('direction')} |"
                )

            lines.append("")

            td = comp.get("transcript_diff", {})
            lines += [
                f"**Transcript similarity vs previous:** {fmt(td.get('similarity_pct'), '%')}",
                f"**Word-level changes:** {fmt(td.get('change_count'))}",
                "",
            ]

            changes = td.get("changes", [])
            if changes:
                lines += ["**Sample word changes:**", ""]
                for c in changes[:10]:
                    before = c.get("before") or "(nothing)"
                    after = c.get("after") or "(nothing)"
                    lines.append(f"- `[{c.get('type')}]` `{before}` → `{after}`")
                lines.append("")

            obs = comp.get("observations") or comp.get("observation_notes") or []
            lines += ["### Observations", ""]
            for note in obs:
                lines.append(f"- {note}")
            lines.append("")
        else:
            lines += [
                "### Observations",
                "",
                "- This is the baseline/reference stage.",
                "- Future stages are compared against it, but it is not treated as ground truth.",
                "",
            ]

        lines += ["---", ""]

    if len(stages) >= 2:
        baseline = stages[0]
        latest = stages[-1]
        bm = get_metrics(baseline)
        lm = get_metrics(latest)
        bq = get_quality(baseline)
        lq = get_quality(latest)

        b_num = get_stage_num(baseline, 0)
        b_name = get_stage_name(baseline, 0)
        l_num = get_stage_num(latest, len(stages) - 1)
        l_name = get_stage_name(latest, len(stages) - 1)

        lines += [
            "## Net Gain: Baseline → Latest Stage",
            "",
            f"Comparing **Stage {b_num}: `{b_name}`** → **Stage {l_num}: `{l_name}`**",
            "",
            "| Metric | Baseline | Latest | Net Change |",
            "|--------|----------|--------|------------|",
        ]

        for metric in [
            "word_count", "segment_count", "avg_confidence",
            "ttft_final_ms", "ttft_partial_ms", "empty_segments",
            "total_time_sec", "overall_quality_score"
        ]:
            bv = baseline.get(metric) if metric == "overall_quality_score" else bm.get(metric)
            lv = latest.get(metric) if metric == "overall_quality_score" else lm.get(metric)
            if bv is None and lv is None:
                continue
            diff = round(lv - bv, 4) if (bv is not None and lv is not None) else None
            sign = "+" if (diff or 0) > 0 else ""
            lines.append(f"| {metric} | {fmt(bv)} | {fmt(lv)} | {sign}{fmt(diff)} |")

        if bq or lq:
            lines += [
                "",
                "### Net quality score comparison",
                "",
                "| Quality Metric | Baseline | Latest | Net Change |",
                "|---------------|----------|--------|------------|",
            ]
            quality_keys = [
                ("Punctuation Quality", "punctuation_quality"),
                ("Numeric Quality", "numeric_quality"),
                ("Domain Phrase Quality", "domain_quality"),
                ("VAD Quality", "vad_quality"),
                ("Short Word Quality", "short_word_quality"),
                ("Readability Quality", "readability_quality"),
            ]
            for label, key in quality_keys:
                bv = bq.get(key, {}).get("score")
                lv = lq.get(key, {}).get("score")
                if bv is None and lv is None:
                    continue
                diff = round(lv - bv, 4) if (bv is not None and lv is not None) else None
                sign = "+" if (diff or 0) > 0 else ""
                lines.append(f"| {label} | {fmt(bv)} | {fmt(lv)} | {sign}{fmt(diff)} |")

        lines += ["", "---", ""]

    lines += [
        "## Stage Progression Summary",
        "",
        "| # | Stage Name | Phase | Words | Segs | Overall Quality | Avg Conf | TTFT Final (ms) | Total Time (s) |",
        "|---|-----------|-------|-------|------|-----------------|----------|-----------------|----------------|",
    ]

    for idx, stage in enumerate(stages):
        m = get_metrics(stage)
        stage_num = get_stage_num(stage, idx)
        stage_name = get_stage_name(stage, idx)
        phase = get_phase(stage)

        lines.append(
            f"| {stage_num} | `{stage_name}` | {fmt(phase or None)} | {fmt(m.get('word_count'))} | {fmt(m.get('segment_count'))} | {fmt(stage.get('overall_quality_score'))} | {fmt(m.get('avg_confidence'))} | {fmt(m.get('ttft_final_ms'))} | {fmt(m.get('total_time_sec'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Metric Reference",
        "",
        "| Metric | Description | Better direction |",
        "|--------|-------------|-----------------|",
        "| TTFT Partial | Time to first interim result | Lower |",
        "| TTFT Final | Time to first committed segment | Lower |",
        "| Avg Confidence | Azure certainty score 0–1 | Higher |",
        "| Word Count | Total words captured | Usually higher |",
        "| Empty Segments | Segments with no text | Lower |",
        "| Total Time | End-to-end processing time | Lower |",
        "| Overall Quality | Composite transcript quality score | Higher |",
        "| Numeric Quality | Number rendering / numeric-context quality | Higher |",
        "| VAD Quality | Segmentation / truncation quality | Higher |",
        "",
        f"*Generated by `generate_observation_doc.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN TEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_plain_text(report: dict) -> str:
    stages = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines = []

    lines += [
        "AZURE STT — INCREMENTAL IMPROVEMENT OBSERVATIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Stages run: {len(stages)}",
        "=" * 80,
        "",
    ]

    for i, stage in enumerate(stages):
        m = get_metrics(stage)
        q = get_quality(stage)
        stage_num = get_stage_num(stage, i)
        stage_name = get_stage_name(stage, i)
        phase = get_phase(stage)
        task = get_task(stage)
        description = get_description(stage)
        comp_key = get_comp_key(stage, i)
        params = get_stage_parameters(stage, stage_name)

        comp = next((c for c in comparisons if c.get("to_stage") == comp_key), None)

        lines += [
            f"STAGE {stage_num}: {stage_name.upper()}",
            f"  Phase        : {phase or 'N/A'}",
            f"  Task         : {task or 'N/A'}",
            f"  Description  : {description}",
            "",
            "  PARAMETERS / CHANGES TESTED:",
        ]

        for p in params["azure_parameters_tested"]:
            lines.append(f"    Azure: {p}")
        for c in params["changes_made"]:
            lines.append(f"    Change: {c}")
        if not params["azure_parameters_tested"] and not params["changes_made"]:
            lines.append("    No parameter metadata available")

        lines += [
            "",
            f"  TTFT Partial : {fmt(m.get('ttft_partial_ms'), ' ms')}",
            f"  TTFT Final   : {fmt(m.get('ttft_final_ms'), ' ms')}",
            f"  Total Time   : {fmt(m.get('total_time_sec'), ' sec')}",
            f"  Segments     : {fmt(m.get('segment_count'))}",
            f"  Words        : {fmt(m.get('word_count'))}",
            f"  Avg Conf     : {fmt(m.get('avg_confidence'))}",
            f"  Empty Segs   : {fmt(m.get('empty_segments'))}",
            f"  Overall Qual : {fmt(stage.get('overall_quality_score'))}",
        ]

        if q:
            lines += [
                f"  Punctuation  : {fmt(q.get('punctuation_quality', {}).get('score'))}",
                f"  Numeric      : {fmt(q.get('numeric_quality', {}).get('score'))}",
                f"  Domain       : {fmt(q.get('domain_quality', {}).get('score'))}",
                f"  VAD          : {fmt(q.get('vad_quality', {}).get('score'))}",
                f"  Short Word   : {fmt(q.get('short_word_quality', {}).get('score'))}",
                f"  Readability  : {fmt(q.get('readability_quality', {}).get('score'))}",
            ]

        lines.append("")

        if comp:
            from_stage = comp.get("from_stage", "previous")
            to_stage = comp.get("to_stage", stage_name)

            lines.append(f"  CHANGES VS PREVIOUS STAGE ({from_stage} → {to_stage}):")
            for metric, d in comp.get("metric_deltas", {}).items():
                if d.get("change") is None:
                    continue
                arrow = {
                    "improved": "↑",
                    "worse": "↓",
                    "same": "→",
                }.get(d.get("direction"), "?")
                sign = "+" if d["change"] > 0 else ""
                lines.append(
                    f"    {metric:<26}: {fmt(d.get('prev'))} → {fmt(d.get('curr'))} ({sign}{d.get('change')}) {arrow} {d.get('direction')}"
                )

            obs = comp.get("observations") or comp.get("observation_notes") or []
            lines += ["", "  OBSERVATIONS:"]
            for note in obs:
                lines.append(f"    {note}")
        else:
            lines += [
                "  OBSERVATIONS:",
                "    Baseline/reference stage only.",
            ]

        lines += ["", "-" * 80, ""]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    report = load_report()

    md_content = build_markdown(report)
    txt_content = build_plain_text(report)

    os.makedirs("observations", exist_ok=True)

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print("Observation report generated:")
    print(f"  Markdown → {MD_OUT}")
    print(f"  Text     → {TXT_OUT}")
    print()
    print(txt_content[:2000])
    if len(txt_content) > 2000:
        print("...(see full file for rest)")


if __name__ == "__main__":
    main()
