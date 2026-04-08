import argparse
import asyncio
import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import websockets
import pandas as pd
from jiwer import wer


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

SEND_CHUNK_SEC = 30
SEND_CHUNK_SAMPLES = SAMPLE_RATE * SEND_CHUNK_SEC
SEND_CHUNK_BYTES = SEND_CHUNK_SAMPLES * 2

SEND_DELAY_MS = 250

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FOLDER = Path("transcription_results_new")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# =========================
# AUDIO LOADER
# =========================
def load_wav_as_pcm16(filepath):
    print(f"Loading audio -> {filepath.name}")

    audio, sr = librosa.load(
        str(filepath),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration_sec = len(audio) / SAMPLE_RATE

    pcm16 = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

    print(
        f"Loaded {filepath.name} | "
        f"Duration: {duration_sec:.1f} sec"
    )

    return pcm16, duration_sec


# =========================
# SAVE RESULTS
# =========================
def save_results(filepath, transcript_text, result_json):
    output_dir = OUTPUT_FOLDER / filepath.stem
    output_dir.mkdir(exist_ok=True)

    transcript_path = output_dir / f"{filepath.stem}_transcript.txt"
    latency_path = output_dir / f"{filepath.stem}_latency.json"

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"SAVED -> {filepath.name}")


# =========================
# REFERENCE TEXT
# =========================
def load_reference_text(filepath):
    txt_path = filepath.with_suffix(".txt")

    if txt_path.exists():
        return txt_path.read_text(
            encoding="utf-8"
        ).strip()

    return ""


# =========================
# EXCEL REPORT
# =========================
def generate_excel_report(rows):
    excel_path = OUTPUT_FOLDER / "final_report.xlsx"

    df = pd.DataFrame(rows)

    df.to_excel(
        excel_path,
        index=False,
        engine="openpyxl"
    )

    print(f"EXCEL SAVED -> {excel_path}")


# =========================
# TRANSCRIBE ONE FILE
# =========================
async def transcribe_file(filepath, host, port):
    uri = f"ws://{host}:{port}/ws"

    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_wav_as_pcm16(filepath)

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    first_chunk_time = None
    send_start_time = None
    send_end_time = None

    response_num = 0

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**24
        ) as ws:

            async def sender():
                nonlocal first_chunk_time
                nonlocal send_start_time
                nonlocal send_end_time

                send_start_time = time.time()

                offset = 0
                chunk_num = 0

                while offset < len(pcm):
                    chunk = pcm[
                        offset: offset + SEND_CHUNK_BYTES
                    ]
                    offset += SEND_CHUNK_BYTES
                    chunk_num += 1

                    now = time.time()

                    if first_chunk_time is None:
                        first_chunk_time = now

                    await ws.send(chunk)

                    print(
                        f"{filepath.name} | "
                        f"SENT CHUNK {chunk_num}"
                    )

                    await asyncio.sleep(
                        SEND_DELAY_MS / 1000
                    )

                send_end_time = time.time()

                await asyncio.sleep(1.0)

                await ws.send(
                    json.dumps({"cmd": "flush"})
                )

            async def receiver():
                nonlocal first_response_time
                nonlocal first_final_time
                nonlocal response_num

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(),
                            timeout=120
                        )

                        now = time.time()

                        msg = json.loads(raw)

                        text = msg.get("text", "")
                        msg_type = msg.get("type", "")

                        is_final = (
                            msg_type == "transcript"
                        )

                        response_num += 1

                        elapsed_ms = (
                            now - start_time
                        ) * 1000

                        latency_from_send_start = (
                            (now - send_start_time) * 1000
                            if send_start_time else None
                        )

                        latency_from_first_chunk = (
                            (now - first_chunk_time) * 1000
                            if first_chunk_time else None
                        )

                        latency_from_send_end = (
                            (now - send_end_time) * 1000
                            if send_end_time else None
                        )

                        if first_response_time is None:
                            first_response_time = elapsed_ms

                        if is_final and first_final_time is None:
                            first_final_time = elapsed_ms

                        words = len(text.split()) if text else 0
                        chars = len(text)

                        latencies.append({
                            "response_num": response_num,
                            "latency_from_start_ms": round(elapsed_ms, 4),
                            "latency_from_send_start_ms": round(latency_from_send_start, 4) if latency_from_send_start else None,
                            "latency_from_first_chunk_ms": round(latency_from_first_chunk, 4) if latency_from_first_chunk else None,
                            "latency_from_send_end_ms": round(latency_from_send_end, 4) if latency_from_send_end else None,
                            "is_final": is_final,
                            "words": words,
                            "char_count": chars
                        })

                        if text and is_final:
                            final_transcript_parts.append(text)

                            print(
                                f"{filepath.name} | "
                                f"FINAL {response_num}"
                            )

                    except asyncio.TimeoutError:
                        print(
                            f"{filepath.name} | timeout"
                        )
                        break

                    except websockets.exceptions.ConnectionClosed:
                        print(
                            f"{filepath.name} | closed"
                        )
                        break

            await asyncio.gather(
                sender(),
                receiver()
            )

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return None

    total_time = time.time() - start_time

    transcript_text = "\n".join(
        final_transcript_parts
    )

    latency_values_start = [
        x["latency_from_send_start_ms"]
        for x in latencies
        if x["latency_from_send_start_ms"] is not None
    ]

    latency_values_end = [
        x["latency_from_send_end_ms"]
        for x in latencies
        if x["latency_from_send_end_ms"] is not None
    ]

    result_json = {
        "audio_file": str(filepath),
        "audio_duration_sec": duration_sec,
        "total_processing_time_sec": round(total_time, 4),
        "timestamp": datetime.now().isoformat(),
        "model": "parakeet-tdt-0.6b-v3",
        "language": "multi",
        "timing_metrics": {
            "connection_time_sec": 0.0,
            "send_duration_sec": round(
                send_end_time - send_start_time, 4
            ) if send_end_time else None,
            "first_byte_latency_sec": round(
                first_response_time / 1000, 4
            ) if first_response_time else None,
            "first_response_latency_sec": round(
                first_response_time / 1000, 4
            ) if first_response_time else None,
            "first_final_latency_sec": round(
                first_final_time / 1000, 4
            ) if first_final_time else None,
            "time_to_first_chunk_sec": round(
                first_chunk_time - start_time, 4
            ) if first_chunk_time else None
        },
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(
                1 for x in latencies if x["is_final"]
            ),
            "interim_responses": sum(
                1 for x in latencies if not x["is_final"]
            ),
            "total_words": sum(
                x["words"] for x in latencies
            ),
            "total_characters": sum(
                x["char_count"] for x in latencies
            ),
            "avg_latency_from_send_start_ms": round(
                statistics.mean(latency_values_start), 4
            ) if latency_values_start else None,
            "avg_latency_from_send_end_ms": round(
                statistics.mean(latency_values_end), 4
            ) if latency_values_end else None
        }
    }

    save_results(
        filepath,
        transcript_text,
        result_json
    )

    reference_text = load_reference_text(filepath)

    calculated_wer = None

    if reference_text and transcript_text:
        try:
            calculated_wer = round(
                wer(
                    reference_text.lower(),
                    transcript_text.lower()
                ) * 100,
                2
            )
        except Exception:
            calculated_wer = None

    return {
        "file_name": filepath.name,
        "reference_txt": reference_text,
        "ttft_ms": (
            result_json["timing_metrics"]["first_final_latency_sec"] * 1000
            if result_json["timing_metrics"]["first_final_latency_sec"]
            else None
        ),
        "ttfb_ms": (
            result_json["timing_metrics"]["first_response_latency_sec"] * 1000
            if result_json["timing_metrics"]["first_response_latency_sec"]
            else None
        ),
        "avg_latency_ms": result_json["summary"]["avg_latency_from_send_start_ms"],
        "wer": calculated_wer,
        "total_time_sec": total_time,
        "transcription": transcript_text
    }


# =========================
# BATCH RUNNER
# =========================
async def run_batch(host, port):
    files = sorted(INPUT_FOLDER.glob("*.wav"))

    total_files = len(files)

    print("=" * 80)
    print(f"TOTAL FILES = {total_files}")
    print("=" * 80)

    report_rows = []

    for idx, file in enumerate(files, start=1):
        print(f"\n[{idx}/{total_files}] {file.name}")

        row = await transcribe_file(
            filepath=file,
            host=host,
            port=port
        )

        if row:
            report_rows.append(row)

    generate_excel_report(report_rows)

    print("\nALL FILES COMPLETED")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8001)

    return parser.parse_args()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_batch(
            args.host,
            args.port
        )
    )


[2/6] 0a530b75-d3b1-4533-8ca9-f405d41b445e.wav

STARTING -> 0a530b75-d3b1-4533-8ca9-f405d41b445e.wav
Loading audio -> 0a530b75-d3b1-4533-8ca9-f405d41b445e.wav
Loaded 0a530b75-d3b1-4533-8ca9-f405d41b445e.wav | Duration: 14.0 sec
0a530b75-d3b1-4533-8ca9-f405d41b445e.wav | SENT CHUNK 1
0a530b75-d3b1-4533-8ca9-f405d41b445e.wav | FINAL 1
 with self.send_context():
  File "/usr/lib/python3.11/contextlib.py", line 204, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/re_nikitav/parakeet_test/env/lib/python3.11/site-packages/websockets/asyncio/connection.py", line 957, in send_context
    raise self.protocol.close_exc from original_exc
websockets.exceptions.ConnectionClosedError: received 1011 (internal error) keepalive ping timeout; then sent 1011 (internal error) keepalive ping timeout
