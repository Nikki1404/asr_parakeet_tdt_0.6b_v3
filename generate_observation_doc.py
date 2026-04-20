"""
OBSERVATION DOC GENERATOR
===========================
Reads observations/comparison_report.json and prints a
formatted documentation report you can copy into Confluence,
Notion, Google Docs, or send to your manager.

Run after every stage (or at the end of all stages):
  python generate_observation_doc.py

Output:
  observations/OBSERVATION_REPORT.md   ← Markdown doc
  observations/OBSERVATION_REPORT.txt  ← Plain text doc
"""

import os
import json
from datetime import datetime

REPORT_PATH = os.path.join("observations", "comparison_report.json")
MD_OUT      = os.path.join("observations", "OBSERVATION_REPORT.md")
TXT_OUT     = os.path.join("observations", "OBSERVATION_REPORT.txt")


def load_report() -> dict:
    if not os.path.exists(REPORT_PATH):
        print(f"No report found at {REPORT_PATH}")
        print("Run azure_incremental.py first.")
        exit(1)
    with open(REPORT_PATH) as f:
        return json.load(f)


def fmt(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val}{suffix}"


def direction_symbol(d: str) -> str:
    return {"improved": "✅", "worse": "⚠️", "same": "➡️", "unknown": "❓"}.get(d, "❓")


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

    # ── Stage-by-stage detail ──────────────────────────────────────────
    for i, stage in enumerate(stages):
        m    = stage["metrics"]
        feat = stage.get("features", {})
        comp = next((c for c in comparisons if c["to_stage"] == stage["stage"]), None)

        lines += [
            f"## Stage {i}: `{stage['stage']}`",
            "",
            f"**What was added:** {stage['stage_description']}",
            "",
            "### Features active at this stage",
            "",
        ]

        for k, v in feat.items():
            tick = "✅" if v else "⬜"
            lines.append(f"- {tick} `{k}`")

        lines += [
            "",
            "### Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| TTFT Partial | {fmt(m.get('ttft_partial_ms'), ' ms')} |",
            f"| TTFT Final   | {fmt(m.get('ttft_final_ms'), ' ms')} |",
            f"| Total Time   | {fmt(m.get('total_time_sec'), ' sec')} |",
            f"| Segments     | {fmt(m.get('segment_count'))} |",
            f"| Words        | {fmt(m.get('word_count'))} |",
            f"| Empty Segs   | {fmt(m.get('empty_segments'))} |",
            f"| Avg Confidence | {fmt(m.get('avg_confidence'))} |",
            f"| Min Confidence | {fmt(m.get('min_confidence'))} |",
            f"| Max Confidence | {fmt(m.get('max_confidence'))} |",
            "",
        ]

        lines += [
            "### Transcript",
            "",
            "```",
            stage.get("transcript", "(empty)")[:800],
            "```" if len(stage.get("transcript", "")) <= 800 else "…(truncated)\n```",
            "",
        ]

        if comp:
            lines += [
                "### Change vs Previous Stage",
                "",
                "| Metric | Before | After | Change | Signal |",
                "|--------|--------|-------|--------|--------|",
            ]
            for metric, d in comp["metric_deltas"].items():
                if d["change"] is None:
                    continue
                sign = "+" if d["change"] > 0 else ""
                sym  = direction_symbol(d["direction"])
                lines.append(
                    f"| {metric} | {fmt(d['prev'])} | {fmt(d['curr'])} "
                    f"| {sign}{d['change']} | {sym} {d['direction']} |"
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
                    lines.append(f"- `[{c['type']}]` `{c['before'] or '(nothing)'}` → `{c['after'] or '(nothing)'}`")
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

    # ── Net gain summary table ─────────────────────────────────────────
    if len(stages) >= 2:
        baseline = stages[0]
        latest   = stages[-1]
        bm       = baseline["metrics"]
        lm       = latest["metrics"]

        lines += [
            "## Net Gain: Baseline → Latest Stage",
            "",
            f"Comparing **{baseline['stage']}** → **{latest['stage']}**",
            "",
            "| Metric | Baseline | Latest | Net Change |",
            "|--------|----------|--------|------------|",
        ]

        for metric in ["word_count", "segment_count", "avg_confidence",
                        "min_confidence", "ttft_final_ms", "ttft_partial_ms",
                        "empty_segments", "total_time_sec"]:
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

    # ── Stage progression table ────────────────────────────────────────
    lines += [
        "## Stage Progression Summary",
        "",
        "| Stage | Words | Segs | Avg Conf | TTFT Final (ms) | Total Time (s) |",
        "|-------|-------|------|----------|-----------------|----------------|",
    ]
    for stage in stages:
        m = stage["metrics"]
        lines.append(
            f"| {stage['stage']} "
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
        "## How to Read This Document",
        "",
        "- **TTFT Partial** — time to first interim (partial) transcript. Lower = more real-time feel.",
        "- **TTFT Final** — time to first committed segment. Critical for IVR turn-taking.",
        "- **Avg Confidence** — Azure's certainty score (0–1). Higher = more accurate recognition.",
        "- **Empty Segments** — segments with no recognised text. Ideally zero.",
        "- **Transcript similarity** — how similar the transcript is to the previous stage (word-level).",
        "- **Word-level changes** — actual words that changed between stages.",
        "",
        "### Confidence Interpretation",
        "| Range | Signal |",
        "|-------|--------|",
        "| > 0.90 | Excellent — clear, confident recognition |",
        "| 0.75–0.90 | Good — minor uncertainty |",
        "| 0.65–0.75 | Moderate — possible misrecognition |",
        "| < 0.65 | Low — stressed speech, noise, or wrong language |",
        "",
        f"*Report generated by generate_observation_doc.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


def build_plain_text(report: dict) -> str:
    """Plain text version for copy-paste into email or Slack."""
    stages      = report.get("stages", [])
    comparisons = report.get("comparisons", [])
    lines       = []

    lines += [
        "AZURE STT — INCREMENTAL IMPROVEMENT OBSERVATIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Stages run: {len(stages)}",
        "=" * 60,
        "",
    ]

    for i, stage in enumerate(stages):
        m    = stage["metrics"]
        comp = next((c for c in comparisons if c["to_stage"] == stage["stage"]), None)

        lines += [
            f"STAGE {i}: {stage['stage'].upper()}",
            f"  What was added: {stage['stage_description']}",
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
            for metric, d in comp["metric_deltas"].items():
                if d["change"] is None:
                    continue
                arrow = {"improved": "↑", "worse": "↓", "same": "→"}.get(d["direction"], "?")
                sign  = "+" if d["change"] > 0 else ""
                lines.append(f"    {metric:<22}: {fmt(d['prev'])} → {fmt(d['curr'])}  ({sign}{d['change']})  {arrow}")

            lines += ["", "  OBSERVATIONS:"]
            for note in comp.get("observation_notes", []):
                lines.append(f"    {note}")
        else:
            lines += ["  OBSERVATIONS:", "    Baseline stage — all future stages compared to this."]

        lines += ["", "-" * 60, ""]

    if len(stages) >= 2:
        baseline = stages[0]
        latest   = stages[-1]
        bm       = baseline["metrics"]
        lm       = latest["metrics"]
        lines += [
            f"NET GAIN: {baseline['stage']} → {latest['stage']}",
            "",
        ]
        for metric in ["word_count", "avg_confidence", "ttft_final_ms", "empty_segments"]:
            bv = bm.get(metric)
            lv = lm.get(metric)
            if bv is None or lv is None:
                continue
            diff = round(lv - bv, 4)
            sign = "+" if diff > 0 else ""
            lines.append(f"  {metric:<22}: {bv} → {lv}  ({sign}{diff})")
        lines += ["", "=" * 60]

    return "\n".join(lines)


def main():
    report = load_report()

    md_content  = build_markdown(report)
    txt_content = build_plain_text(report)

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(f"Observation report generated:")
    print(f"  Markdown → {MD_OUT}")
    print(f"  Text     → {TXT_OUT}")
    print()
    print(txt_content[:2000])
    if len(txt_content) > 2000:
        print("…(see full file for rest)")


if __name__ == "__main__":
    main()
