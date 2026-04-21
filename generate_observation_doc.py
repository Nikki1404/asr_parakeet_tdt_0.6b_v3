"""
OBSERVATION DOC GENERATOR
===========================
Reads observations/comparison_report.json and prints a
formatted documentation report you can copy into Confluence,
Notion, Google Docs, or send to your manager.

Supports BOTH formats:

FORMAT A (old / simple)
------------------------
{
  "stages": [
    {
      "stage": "baseline",
      "stage_description": "...",
      "metrics": {...},
      "features": {...},
      "transcript": "..."
    }
  ]
}

FORMAT B (current 12-stage incremental script)
-----------------------------------------------
{
  "stages": [
    {
      "_stage_num":  0,
      "_stage_name": "baseline",
      "_phase":      "Baseline",
      "_task":       "Original working script",
      "detected_language": "en-US",
      "ttft_partial_ms": 2505.8,
      "ttft_final_ms":   3656.8,
      "total_time_sec":  442.96,
      "segment_count":   83,
      "word_count":      1240,
      "empty_segments":  0,
      "avg_confidence":  null,
      "transcript": "..."
    }
  ]
}

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
MD_OUT      = os.path.join("observations", "OBSERVATION_REPORT.md")
TXT_OUT     = os.path.join("observations", "OBSERVATION_REPORT.txt")


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_report() -> dict:
    if not os.path.exists(REPORT_PATH):
        print(f"No report found at {REPORT_PATH}")
        print("Run azure_incremental.py first.")
        exit(1)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FIELD RESOLVERS
# Handles Format A (_stage_name absent) and Format B (_stage_name present)
# ─────────────────────────────────────────────────────────────────────────────

def get_stage_num(stage: dict, fallback_index: int) -> str:
    """
    Reads _stage_num from JSON (Format B).
    Falls back to the list position index (Format A).
    """
    val = stage.get("_stage_num")
    if val is not None:
        return str(val)
    return str(fallback_index)


def get_stage_name(stage: dict, fallback_index: int) -> str:
    """
    Priority:
      1. _stage_name  ← Format B key (12-stage script)
      2. stage        ← Format A key (old script)
      3. stage_N      ← positional fallback
    """
    if stage.get("_stage_name"):
        return stage["_stage_name"]
    if stage.get("stage"):
        return stage["stage"]
    return f"stage_{fallback_index}"


def get_phase(stage: dict) -> str:
    """_phase from Format B, empty string for Format A."""
    return stage.get("_phase", "")


def get_task(stage: dict) -> str:
    """_task from Format B, empty string for Format A."""
    return stage.get("_task", "")


def get_description(stage: dict) -> str:
    """
    Priority:
      1. stage_description  (Format A)
      2. _task + _phase     (Format B)
      3. detected_language  (last-resort fallback)
    """
    if stage.get("stage_description"):
        return stage["stage_description"]
    if stage.get("_task"):
        phase_str = f" [{stage['_phase']}]" if stage.get("_phase") else ""
        return f"{stage['_task']}{phase_str}"
    return f"Detected language: {stage.get('detected_language', 'Unknown')}"


def get_metrics(stage: dict) -> dict:
    """
    Format A → stage["metrics"] (nested dict)
    Format B → metrics are flat keys directly on the stage dict
    """
    if "metrics" in stage and isinstance(stage["metrics"], dict):
        return stage["metrics"]
    return stage


def get_comp_key(stage: dict, fallback_index: int) -> str:
    """Key that comparison entries store in their 'to_stage' field."""
    return get_stage_name(stage, fallback_index)


def fmt(val, suffix="") -> str:
    return "N/A" if val is None else f"{val}{suffix}"


def direction_symbol(d: str) -> str:
    return {
        "improved": "✅",
        "worse":    "⚠️",
        "same":     "➡️",
        "unknown":  "❓",
    }.get(d, "❓")


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown(report: dict) -> str:
    stages      = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines       = []

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
        "incremental improvement process. Each stage adds one feature on top",
        "of the previous and compares the resulting metrics and transcript.",
        "",
        "---",
        "",
    ]

    # ── Per-stage sections ────────────────────────────────────────────
    for i, stage in enumerate(stages):
        m           = get_metrics(stage)
        feat        = stage.get("features", {})
        stage_num   = get_stage_num(stage, i)
        stage_name  = get_stage_name(stage, i)
        phase       = get_phase(stage)
        task        = get_task(stage)
        description = get_description(stage)
        comp_key    = get_comp_key(stage, i)

        comp = next(
            (c for c in comparisons if c.get("to_stage") == comp_key),
            None
        )

        # ── Header ───────────────────────────────────────────────────
        lines += [f"## Stage {stage_num}: `{stage_name}`", ""]

        if phase or task:
            parts = []
            if phase: parts.append(f"**Phase:** {phase}")
            if task:  parts.append(f"**Task:** {task}")
            lines.append("  &nbsp;|&nbsp;  ".join(parts))
            lines.append("")

        lines += [f"**What was added:** {description}", ""]

        # ── Features ─────────────────────────────────────────────────
        lines += ["### Features active at this stage", ""]
        if feat:
            for k, v in feat.items():
                tick = "✅" if v else "⬜"
                lines.append(f"- {tick} `{k}`")
        else:
            lines.append("- No feature metadata available")

        # ── Metrics ───────────────────────────────────────────────────
        lines += [
            "",
            "### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Stage Number      | **{stage_num}** |",
            f"| Stage Name        | **`{stage_name}`** |",
            f"| Phase             | {fmt(phase or None)} |",
            f"| Task              | {fmt(task or None)} |",
            f"| Detected Language | {fmt(stage.get('detected_language'))} |",
            f"| TTFT Partial      | {fmt(m.get('ttft_partial_ms'), ' ms')} |",
            f"| TTFT Final        | {fmt(m.get('ttft_final_ms'), ' ms')} |",
            f"| Total Time        | {fmt(m.get('total_time_sec'), ' sec')} |",
            f"| Segments          | {fmt(m.get('segment_count'))} |",
            f"| Words             | {fmt(m.get('word_count'))} |",
            f"| Empty Segments    | {fmt(m.get('empty_segments'))} |",
            f"| Avg Confidence    | {fmt(m.get('avg_confidence'))} |",
            f"| Min Confidence    | {fmt(m.get('min_confidence'))} |",
            f"| Max Confidence    | {fmt(m.get('max_confidence'))} |",
            f"| Partial Count     | {fmt(m.get('partial_count'))} |",
            "",
        ]

        # ── Transcript ────────────────────────────────────────────────
        transcript = stage.get("transcript", "(empty)")
        lines += [
            "### Transcript",
            "",
            "```",
            transcript[:800],
            "```" if len(transcript) <= 800 else "...(truncated)\n```",
            "",
        ]

        # ── Comparison vs previous ────────────────────────────────────
        if comp:
            from_stage = comp.get("from_stage", "previous")
            to_stage   = comp.get("to_stage",   stage_name)

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
                sym  = direction_symbol(d.get("direction"))
                lines.append(
                    f"| {metric} | {fmt(d.get('prev'))} | {fmt(d.get('curr'))} "
                    f"| {sign}{d.get('change')} | {sym} {d.get('direction')} |"
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
                    after  = c.get("after")  or "(nothing)"
                    lines.append(f"- `[{c.get('type')}]` `{before}` → `{after}`")
                lines.append("")

            # Support both key names from different script versions
            obs = comp.get("observations") or comp.get("observation_notes") or []
            lines += ["### Observations", ""]
            for note in obs:
                lines.append(f"- {note}")
            lines.append("")

        else:
            lines += [
                "### Observations",
                "",
                "- This is the baseline stage — no previous stage to compare against.",
                "- All future stages will be measured against this transcript and metrics.",
                "",
            ]

        lines += ["---", ""]

    # ── Net gain summary ──────────────────────────────────────────────
    if len(stages) >= 2:
        baseline = stages[0]
        latest   = stages[-1]
        bm       = get_metrics(baseline)
        lm       = get_metrics(latest)

        b_num  = get_stage_num(baseline, 0)
        b_name = get_stage_name(baseline, 0)
        l_num  = get_stage_num(latest, len(stages) - 1)
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
            "word_count", "segment_count", "avg_confidence", "min_confidence",
            "ttft_final_ms", "ttft_partial_ms", "empty_segments", "total_time_sec",
        ]:
            bv   = bm.get(metric)
            lv   = lm.get(metric)
            if bv is None and lv is None:
                continue
            diff = round(lv - bv, 4) if (bv is not None and lv is not None) else None
            sign = "+" if (diff or 0) > 0 else ""
            lines.append(
                f"| {metric} | {fmt(bv)} | {fmt(lv)} | {sign}{fmt(diff)} |"
            )

        lines += ["", "---", ""]

    # ── Progression table ─────────────────────────────────────────────
    lines += [
        "## Stage Progression Summary",
        "",
        "| # | Stage Name | Phase | Task | Words | Segs | Avg Conf | TTFT Final (ms) | Total Time (s) |",
        "|---|-----------|-------|------|-------|------|----------|-----------------|----------------|",
    ]

    for idx, stage in enumerate(stages):
        m          = get_metrics(stage)
        stage_num  = get_stage_num(stage, idx)
        stage_name = get_stage_name(stage, idx)
        phase      = get_phase(stage)
        task       = get_task(stage)

        lines.append(
            f"| {stage_num} "
            f"| `{stage_name}` "
            f"| {fmt(phase or None)} "
            f"| {fmt(task or None)} "
            f"| {fmt(m.get('word_count'))} "
            f"| {fmt(m.get('segment_count'))} "
            f"| {fmt(m.get('avg_confidence'))} "
            f"| {fmt(m.get('ttft_final_ms'))} "
            f"| {fmt(m.get('total_time_sec'))} |"
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
        "| Word Count | Total words in transcript | Higher (more captured) |",
        "| Empty Segments | Segments with no text | Lower |",
        "| Total Time | End-to-end processing time | Lower |",
        "",
        "### Confidence Interpretation",
        "| Range | Signal |",
        "|-------|--------|",
        "| > 0.90 | Excellent — clear, calm speech |",
        "| 0.75–0.90 | Good — minor uncertainty |",
        "| 0.65–0.75 | Moderate — possible misrecognition |",
        "| < 0.65 | Low — stress, noise, or wrong language |",
        "",
        f"*Generated by `generate_observation_doc.py` on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN TEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_plain_text(report: dict) -> str:
    stages      = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines       = []

    lines += [
        "AZURE STT — INCREMENTAL IMPROVEMENT OBSERVATIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Stages run: {len(stages)}",
        "=" * 65,
        "",
    ]

    for i, stage in enumerate(stages):
        m           = get_metrics(stage)
        stage_num   = get_stage_num(stage, i)
        stage_name  = get_stage_name(stage, i)
        phase       = get_phase(stage)
        task        = get_task(stage)
        description = get_description(stage)
        comp_key    = get_comp_key(stage, i)

        comp = next(
            (c for c in comparisons if c.get("to_stage") == comp_key),
            None
        )

        lines += [
            f"STAGE {stage_num}: {stage_name.upper()}",
            f"  Phase        : {phase or 'N/A'}",
            f"  Task         : {task or 'N/A'}",
            f"  Description  : {description}",
            "",
            f"  TTFT Partial : {fmt(m.get('ttft_partial_ms'), ' ms')}",
            f"  TTFT Final   : {fmt(m.get('ttft_final_ms'), ' ms')}",
            f"  Total Time   : {fmt(m.get('total_time_sec'), ' sec')}",
            f"  Segments     : {fmt(m.get('segment_count'))}",
            f"  Words        : {fmt(m.get('word_count'))}",
            f"  Avg Conf     : {fmt(m.get('avg_confidence'))}",
            f"  Empty Segs   : {fmt(m.get('empty_segments'))}",
            "",
        ]

        if comp:
            from_stage = comp.get("from_stage", "previous")
            to_stage   = comp.get("to_stage",   stage_name)

            lines.append(
                f"  CHANGES VS PREVIOUS STAGE ({from_stage} → {to_stage}):"
            )

            for metric, d in comp.get("metric_deltas", {}).items():
                if d.get("change") is None:
                    continue
                arrow = {
                    "improved": "↑",
                    "worse":    "↓",
                    "same":     "→",
                }.get(d.get("direction"), "?")
                sign = "+" if d["change"] > 0 else ""
                lines.append(
                    f"    {metric:<22}: "
                    f"{fmt(d.get('prev'))} → {fmt(d.get('curr'))} "
                    f"({sign}{d.get('change')}) {arrow} {d.get('direction')}"
                )

            obs = comp.get("observations") or comp.get("observation_notes") or []
            lines += ["", "  OBSERVATIONS:"]
            for note in obs:
                lines.append(f"    {note}")

        else:
            lines += [
                "  OBSERVATIONS:",
                "    Baseline stage — all future stages compared to this.",
            ]

        lines += ["", "-" * 65, ""]

    # ── Net gain ──────────────────────────────────────────────────────
    if len(stages) >= 2:
        baseline = stages[0]
        latest   = stages[-1]
        bm       = get_metrics(baseline)
        lm       = get_metrics(latest)

        b_label = f"Stage {get_stage_num(baseline, 0)}: {get_stage_name(baseline, 0)}"
        l_label = f"Stage {get_stage_num(latest, len(stages)-1)}: {get_stage_name(latest, len(stages)-1)}"

        lines += [f"NET GAIN: {b_label}  →  {l_label}", ""]

        for metric in ["word_count", "avg_confidence", "ttft_final_ms", "empty_segments"]:
            bv = bm.get(metric)
            lv = lm.get(metric)
            if bv is None or lv is None:
                continue
            diff = round(lv - bv, 4)
            sign = "+" if diff > 0 else ""
            lines.append(f"  {metric:<22}: {bv} → {lv} ({sign}{diff})")

        lines += ["", "=" * 65]

    # ── Progression table ─────────────────────────────────────────────
    lines += ["", "STAGE PROGRESSION SUMMARY", ""]
    lines.append(
        f"  {'#':<4} {'Stage Name':<22} {'Phase':<14} "
        f"{'Words':>6} {'Segs':>5} {'Conf':>7} {'TTFT-F':>8} {'Time':>8}"
    )
    lines.append(
        f"  {'─'*4} {'─'*22} {'─'*14} "
        f"{'─'*6} {'─'*5} {'─'*7} {'─'*8} {'─'*8}"
    )

    for idx, stage in enumerate(stages):
        m          = get_metrics(stage)
        stage_num  = get_stage_num(stage, idx)
        stage_name = get_stage_name(stage, idx)
        phase      = get_phase(stage)

        lines.append(
            f"  {stage_num:<4} {stage_name:<22} {fmt(phase or None):<14} "
            f"{str(m.get('word_count','N/A')):>6} "
            f"{str(m.get('segment_count','N/A')):>5} "
            f"{str(m.get('avg_confidence','N/A')):>7} "
            f"{str(m.get('ttft_final_ms','N/A')):>8} "
            f"{str(m.get('total_time_sec','N/A')):>8}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    report = load_report()

    md_content  = build_markdown(report)
    txt_content = build_plain_text(report)

    os.makedirs("observations", exist_ok=True)

    with open(MD_OUT,  "w", encoding="utf-8") as f:
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
