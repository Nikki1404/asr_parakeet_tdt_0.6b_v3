import argparse
import time
import statistics
import traceback
import json
import wave
from pathlib import Path
from datetime import datetime

import riva.client
import pandas as pd
from jiwer import wer


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

# FIXED: 250 ms chunks instead of 30 sec
CHUNK_MS = 250
SEND_CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FOLDER = Path("transcription_results_grpc")
OUTPUT_FOLDER.mkdir(exist_ok=True)

LANGUAGE = "es-US"


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
def transcribe_file(filepath, host, port):
    print(f"\nSTARTING -> {filepath.name}")

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    first_chunk_time = None
    send_start_time = None
    send_end_time = None

    response_num = 0
    final_transcript_parts = []
    latencies = []

    try:
        auth = riva.client.Auth(
            uri=f"{host}:{port}"
        )

        asr_service = riva.client.ASRService(auth)

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=LANGUAGE,
                max_alternatives=1,
                enable_automatic_punctuation=True
            ),
            interim_results=True
        )

        # PCM ONLY
        with wave.open(str(filepath), "rb") as wf:
            duration_sec = (
                wf.getnframes() / SAMPLE_RATE
            )

            pcm = wf.readframes(
                wf.getnframes()
            )

        send_start_time = time.time()

        chunks = []

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

            chunks.append(chunk)

        send_end_time = time.time()

        print(f"TOTAL CHUNKS -> {len(chunks)}")

        responses = asr_service.streaming_response_generator(
            audio_chunks=chunks,
            streaming_config=config
        )

        for response in responses:
            now = time.time()

            for result in response.results:
                if not result.alternatives:
                    continue

                text = (
                    result.alternatives[0]
                    .transcript
                    .strip()
                )

                is_final = result.is_final

                response_num += 1

                elapsed_ms = (
                    now - start_time
                ) * 1000

                latency_from_send_start = (
                    (now - send_start_time) * 1000
                )

                latency_from_first_chunk = (
                    (now - first_chunk_time) * 1000
                )

                latency_from_send_end = (
                    (now - send_end_time) * 1000
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
                    "latency_from_send_start_ms": round(latency_from_send_start, 4),
                    "latency_from_first_chunk_ms": round(latency_from_first_chunk, 4),
                    "latency_from_send_end_ms": round(latency_from_send_end, 4),
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

        total_time = time.time() - start_time

        transcript_text = "\n".join(
            final_transcript_parts
        )

        latency_values_start = [
            x["latency_from_send_start_ms"]
            for x in latencies
        ]

        latency_values_end = [
            x["latency_from_send_end_ms"]
            for x in latencies
        ]

        result_json = {
            "audio_file": str(filepath),
            "audio_duration_sec": duration_sec,
            "total_processing_time_sec": round(total_time, 4),
            "timestamp": datetime.now().isoformat(),
            "model": "parakeet-ctc-0.6b-es",
            "language": LANGUAGE,
            "timing_metrics": {
                "send_duration_sec": round(
                    send_end_time - send_start_time, 4
                ),
                "first_byte_latency_sec": round(
                    first_response_time / 1000, 4
                ) if first_response_time else None,
                "first_response_latency_sec": round(
                    first_response_time / 1000, 4
                ) if first_response_time else None,
                "first_final_latency_sec": round(
                    first_final_time / 1000, 4
                ) if first_final_time else None
            },
            "latencies": latencies,
            "summary": {
                "total_responses": len(latencies),
                "final_responses": sum(
                    1 for x in latencies if x["is_final"]
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
            calculated_wer = round(
                wer(
                    reference_text.lower(),
                    transcript_text.lower()
                ) * 100,
                2
            )

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

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return None


# =========================
# BATCH RUNNER
# =========================
def run_batch(host, port):
    files = sorted(INPUT_FOLDER.glob("*.wav"))

    report_rows = []

    for idx, file in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {file.name}")

        row = transcribe_file(
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
    parser.add_argument("--port", type=int, default=50051)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch(
        args.host,
        args.port
    )
