"""
OBSERVATION DOC GENERATOR
===========================
Reads observations/comparison_report.json and prints a
formatted documentation report you can copy into Confluence,
Notion, Google Docs, or send to your manager.

Supports BOTH formats:

FORMAT A (old expected)
-----------------------
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

FORMAT B (your current JSON)
----------------------------
{
  "stages": [
    {
      "detected_language": "en-US",
      "ttft_partial_ms": 2505.8,
      "ttft_final_ms": 3656.8,
      "total_time_sec": 442.96,
      "segment_count": 83,
      "word_count": 1240,
      "empty_segments": 0,
      "avg_confidence": null,
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
MD_OUT = os.path.join("observations", "OBSERVATION_REPORT.md")
TXT_OUT = os.path.join("observations", "OBSERVATION_REPORT.txt")


def load_report():
    if not os.path.exists(REPORT_PATH):
        print(f"No report found at {REPORT_PATH}")
        print("Run azure_incremental.py first.")
        exit(1)

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val}{suffix}"


def direction_symbol(d):
    return {
        "improved": "✅",
        "worse": "⚠️",
        "same": "➡️",
        "unknown": "❓"
    }.get(d, "❓")


def get_metrics(stage):
    """
    Supports both:
    Format A -> stage["metrics"]
    Format B -> metrics directly inside stage
    """
    if "metrics" in stage and isinstance(stage["metrics"], dict):
        return stage["metrics"]
    return stage


def get_stage_name(stage, index):
    return stage.get("stage", f"stage_{index}")


def get_stage_description(stage):
    return stage.get(
        "stage_description",
        f"Detected language: {stage.get('detected_language', 'Unknown')}"
    )


def build_markdown(report):
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
        "incremental improvement process. Each stage adds one feature on top",
        "of the previous and compares the resulting metrics and transcript.",
        "",
        "---",
        "",
    ]

    # Stage-by-stage detail
    for i, stage in enumerate(stages):
        m = get_metrics(stage)
        feat = stage.get("features", {})
        stage_name = get_stage_name(stage, i)
        stage_desc = get_stage_description(stage)

        comp = next(
            (
                c for c in comparisons
                if c.get("to_stage") == stage.get("stage")
            ),
            None
        )

        lines += [
            f"## Stage {i}: `{stage_name}`",
            "",
            f"**What was added:** {stage_desc}",
            "",
            "### Features active at this stage",
            "",
        ]

        if feat:
            for k, v in feat.items():
                tick = "✅" if v else "⬜"
                lines.append(f"- {tick} `{k}`")
        else:
            lines.append("- No feature metadata available")

        lines += [
            "",
            "### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Detected Language | {fmt(stage.get('detected_language'))} |",
            f"| TTFT Partial | {fmt(m.get('ttft_partial_ms'), ' ms')} |",
            f"| TTFT Final   | {fmt(m.get('ttft_final_ms'), ' ms')} |",
            f"| Total Time   | {fmt(m.get('total_time_sec'), ' sec')} |",
            f"| Segments     | {fmt(m.get('segment_count'))} |",
            f"| Words        | {fmt(m.get('word_count'))} |",
            f"| Empty Segs   | {fmt(m.get('empty_segments'))} |",
            f"| Avg Confidence | {fmt(m.get('avg_confidence'))} |",
            f"| Min Confidence | {fmt(m.get('min_confidence'))} |",
            f"| Max Confidence | {fmt(m.get('max_confidence'))} |",
            f"| Partial Count | {fmt(m.get('partial_count'))} |",
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

        if comp:
            lines += [
                "### Change vs Previous Stage",
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
                    f"| {metric} | {fmt(d.get('prev'))} | {fmt(d.get('curr'))} "
                    f"| {sign}{d.get('change')} | {sym} {d.get('direction')} |"
                )

            lines += [""]

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

                    lines.append(
                        f"- `[{c.get('type')}]` `{before}` → `{after}`"
                    )

                lines.append("")

            lines += ["### Observations", ""]

            for note in comp.get("observation_notes", []):
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

    # Net gain summary
    if len(stages) >= 2:
        baseline = stages[0]
        latest = stages[-1]

        bm = get_metrics(baseline)
        lm = get_metrics(latest)

        baseline_name = get_stage_name(baseline, 0)
        latest_name = get_stage_name(latest, len(stages) - 1)

        lines += [
            "## Net Gain: Baseline → Latest Stage",
            "",
            f"Comparing **{baseline_name}** → **{latest_name}**",
            "",
            "| Metric | Baseline | Latest | Net Change |",
            "|--------|----------|--------|------------|",
        ]

        for metric in [
            "word_count",
            "segment_count",
            "avg_confidence",
            "min_confidence",
            "ttft_final_ms",
            "ttft_partial_ms",
            "empty_segments",
            "total_time_sec"
        ]:
            bv = bm.get(metric)
            lv = lm.get(metric)

            if bv is None and lv is None:
                continue

            diff = None
            if bv is not None and lv is not None:
                diff = round(lv - bv, 4)

            sign = "+" if (diff or 0) > 0 else ""

            lines.append(
                f"| {metric} | {fmt(bv)} | {fmt(lv)} | {sign}{fmt(diff)} |"
            )

        lines += ["", "---", ""]

    # Stage progression summary
    lines += [
        "## Stage Progression Summary",
        "",
        "| Stage | Words | Segs | Avg Conf | TTFT Final (ms) | Total Time (s) |",
        "|-------|-------|------|----------|-----------------|----------------|",
    ]

    for idx, stage in enumerate(stages):
        m = get_metrics(stage)
        stage_name = get_stage_name(stage, idx)

        lines.append(
            f"| {stage_name} "
            f"| {fmt(m.get('word_count'))} "
            f"| {fmt(m.get('segment_count'))} "
            f"| {fmt(m.get('avg_confidence'))} "
            f"| {fmt(m.get('ttft_final_ms'))} "
            f"| {fmt(m.get('total_time_sec'))} |"
        )

    return "\n".join(lines)


def build_plain_text(report):
    stages = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines = []

    lines += [
        "AZURE STT — INCREMENTAL IMPROVEMENT OBSERVATIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Stages run: {len(stages)}",
        "=" * 60,
        "",
    ]

    for i, stage in enumerate(stages):
        m = get_metrics(stage)
        stage_name = get_stage_name(stage, i)
        stage_desc = get_stage_description(stage)

        comp = next(
            (
                c for c in comparisons
                if c.get("to_stage") == stage.get("stage")
            ),
            None
        )

        lines += [
            f"STAGE {i}: {stage_name.upper()}",
            f"  What was added: {stage_desc}",
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
            lines.append("  CHANGES VS PREVIOUS STAGE:")

            for metric, d in comp.get("metric_deltas", {}).items():
                if d.get("change") is None:
                    continue

                arrow = {
                    "improved": "↑",
                    "worse": "↓",
                    "same": "→"
                }.get(d.get("direction"), "?")

                sign = "+" if d["change"] > 0 else ""

                lines.append(
                    f"    {metric:<22}: "
                    f"{fmt(d.get('prev'))} → {fmt(d.get('curr'))} "
                    f"({sign}{d.get('change')}) {arrow}"
                )

            lines += ["", "  OBSERVATIONS:"]

            for note in comp.get("observation_notes", []):
                lines.append(f"    {note}")

        else:
            lines += [
                "  OBSERVATIONS:",
                "    Baseline stage — all future stages compared to this."
            ]

        lines += ["", "-" * 60, ""]

    if len(stages) >= 2:
        baseline = stages[0]
        latest = stages[-1]

        bm = get_metrics(baseline)
        lm = get_metrics(latest)

        lines += [
            f"NET GAIN: {get_stage_name(baseline, 0)} → "
            f"{get_stage_name(latest, len(stages)-1)}",
            "",
        ]

        for metric in [
            "word_count",
            "avg_confidence",
            "ttft_final_ms",
            "empty_segments"
        ]:
            bv = bm.get(metric)
            lv = lm.get(metric)

            if bv is None or lv is None:
                continue

            diff = round(lv - bv, 4)
            sign = "+" if diff > 0 else ""

            lines.append(
                f"  {metric:<22}: {bv} → {lv} ({sign}{diff})"
            )

        lines += ["", "=" * 60]

    return "\n".join(lines)


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
