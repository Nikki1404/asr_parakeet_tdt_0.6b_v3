import argparse
import json
import time
import statistics
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import pandas as pd
from jiwer import wer

import riva.client


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000

# 80 ms chunks — smaller = lower latency over gRPC streaming
SEND_CHUNK_MS = 80
SEND_CHUNK_SAMPLES = SAMPLE_RATE * SEND_CHUNK_MS // 1000
SEND_CHUNK_BYTES = SEND_CHUNK_SAMPLES * 2  # int16 → 2 bytes/sample

INPUT_FOLDER = Path(
    "/home/re_nikitav/parakeet-asr-multilingual/audio_samples"
)

OUTPUT_FOLDER = Path("transcription_results_grpc")
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
        return txt_path.read_text(encoding="utf-8").strip()

    return ""


# =========================
# EXCEL REPORT
# =========================
def generate_excel_report(rows):
    excel_path = OUTPUT_FOLDER / "final_report.xlsx"

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False, engine="openpyxl")

    print(f"EXCEL SAVED -> {excel_path}")


# =========================
# CHUNK GENERATOR
# Yields PCM bytes in small chunks; timestamps first-chunk send time.
# =========================
def chunk_generator(pcm, chunk_size, timing):
    """
    Generator that yields audio chunks and records timing.
    timing is a dict mutated in place:
      - send_start_time
      - first_chunk_time
      - send_end_time
    """
    offset = 0
    chunk_num = 0

    timing["send_start_time"] = time.time()

    while offset < len(pcm):
        chunk = pcm[offset: offset + chunk_size]
        offset += chunk_size
        chunk_num += 1

        now = time.time()
        if timing.get("first_chunk_time") is None:
            timing["first_chunk_time"] = now

        yield chunk

    timing["send_end_time"] = time.time()


# =========================
# TRANSCRIBE ONE FILE
# =========================
def transcribe_file(filepath, server, language):
    print(f"\nSTARTING -> {filepath.name}")

    pcm, duration_sec = load_wav_as_pcm16(filepath)

    # riva.client setup
    auth = riva.client.Auth(uri=server)
    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=True,
    )

    final_transcript_parts = []
    latencies = []

    start_time = time.time()

    first_response_time = None
    first_final_time = None
    response_num = 0

    # timing is mutated by chunk_generator
    timing = {
        "send_start_time": None,
        "first_chunk_time": None,
        "send_end_time": None,
    }

    try:
        chunks = chunk_generator(pcm, SEND_CHUNK_BYTES, timing)

        responses = asr_service.streaming_response_generator(
            audio_chunks=chunks,
            streaming_config=config,
        )

        for response in responses:
            now = time.time()
            elapsed_ms = (now - start_time) * 1000

            for result in response.results:
                if not result.alternatives:
                    continue

                text = result.alternatives[0].transcript
                is_final = result.is_final

                response_num += 1

                latency_from_send_start = (
                    (now - timing["send_start_time"]) * 1000
                    if timing["send_start_time"] else None
                )

                latency_from_first_chunk = (
                    (now - timing["first_chunk_time"]) * 1000
                    if timing["first_chunk_time"] else None
                )

                latency_from_send_end = (
                    (now - timing["send_end_time"]) * 1000
                    if timing["send_end_time"] else None
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
                    "char_count": chars,
                    "text_preview": text[:80] if text else "",
                })

                if is_final and text:
                    final_transcript_parts.append(text)
                    print(
                        f"{filepath.name} | FINAL #{response_num} | "
                        f"{words} words | latency {latency_from_send_start:.0f} ms"
                        if latency_from_send_start else
                        f"{filepath.name} | FINAL #{response_num} | {words} words"
                    )
                else:
                    print(
                        f"{filepath.name} | interim #{response_num} | {text[:60]!r}"
                    )

    except Exception as e:
        print(f"FAILED -> {filepath.name}")
        print(str(e))
        traceback.print_exc()
        return None

    total_time = time.time() - start_time

    transcript_text = "\n".join(final_transcript_parts)

    send_start_time = timing["send_start_time"]
    send_end_time = timing["send_end_time"]
    first_chunk_time = timing["first_chunk_time"]

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
        "language": language,
        "timing_metrics": {
            "send_duration_sec": round(
                send_end_time - send_start_time, 4
            ) if send_end_time and send_start_time else None,
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
            ) if first_chunk_time and start_time else None,
        },
        "latencies": latencies,
        "summary": {
            "total_responses": len(latencies),
            "final_responses": sum(1 for x in latencies if x["is_final"]),
            "interim_responses": sum(1 for x in latencies if not x["is_final"]),
            "total_words": sum(x["words"] for x in latencies),
            "total_characters": sum(x["char_count"] for x in latencies),
            "avg_latency_from_send_start_ms": round(
                statistics.mean(latency_values_start), 4
            ) if latency_values_start else None,
            "avg_latency_from_send_end_ms": round(
                statistics.mean(latency_values_end), 4
            ) if latency_values_end else None,
        },
    }

    save_results(filepath, transcript_text, result_json)

    reference_text = load_reference_text(filepath)
    calculated_wer = None

    if reference_text and transcript_text:
        try:
            calculated_wer = round(
                wer(reference_text.lower(), transcript_text.lower()) * 100,
                2
            )
        except Exception:
            pass

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
        "transcription": transcript_text,
    }


# =========================
# BATCH RUNNER
# =========================
def run_batch(server, language):
    files = sorted(INPUT_FOLDER.glob("*.wav"))

    if not files:
        print(f"No .wav files found in {INPUT_FOLDER}")
        return

    report_rows = []

    for idx, file in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {file.name}")

        row = transcribe_file(
            filepath=file,
            server=server,
            language=language,
        )

        if row:
            report_rows.append(row)

    generate_excel_report(report_rows)

    print("\nALL FILES COMPLETED")


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ASR transcription via Riva gRPC"
    )
    parser.add_argument(
        "--server",
        default="192.168.4.62:50051",
        help="gRPC server address (default: 192.168.4.62:50051)",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Language code (default: en-US)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch(
        server=args.server,
        language=args.language,
    )


python transcribe_grpc.py --server 192.168.4.62:50051 --language en-US
